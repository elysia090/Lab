// glk_core_v1_3_1.cpp
// "GrayLUT Core" v1.3.1 — Competitive Programming Edition
// - Calm naming, branch-light hot path, optional witness check (off by default)
// - 24-byte Entry, cache-friendly, O(0) execution style
// Build: g++ -O3 -pipe -static -s -o glk_core_v1_3_1 glk_core_v1_3_1.cpp
// If -static is disallowed on your OJ: drop -static.
//
// Tunables (change via -D…)
#ifndef KEY_BITS
#define KEY_BITS 18              // 2^18 = 262,144 entries per LUT bank
#endif
#ifndef TENANT_BITS
#define TENANT_BITS 2
#endif
#ifndef WITNESS_ON
#define WITNESS_ON 0             // 0: off (CP default); 1: enable tag verification
#endif
#ifndef URN_RING_LEN
#define URN_RING_LEN 64
#endif

#include <bits/stdc++.h>
using namespace std;

static constexpr int KEY_SPACE = 1 << KEY_BITS;
static constexpr uint32_t KEY_MASK = KEY_SPACE - 1;

#if defined(__GNUC__) || defined(__clang__)
#  define LIKELY(x)   (__builtin_expect(!!(x),1))
#  define UNLIKELY(x) (__builtin_expect(!!(x),0))
#else
#  define LIKELY(x)   (x)
#  define UNLIKELY(x) (x)
#endif

// ---------- Small types & enums ----------
struct Key { uint32_t v; };
enum EffectId : uint8_t { EffToggle=0, EffConstAdd=1, EffIdSwap=2, EffHold=7 };
enum LutKind  : uint8_t { BK_Robin=0, BK_Lattice=1, BK_Path=2, BK_Cech=3, BK_Urn=4 };
enum BudgetCat: uint8_t { BC_Robin=0, BC_Price=1, BC_Cech=2, BC_Hodge=3 };
enum BudgetLv : uint8_t { BL_Hard=0, BL_Soft=1, BL_Learn=2 };

struct Mono {
  static constexpr uint16_t PRICE_NONNEG = 1u<<0;
  static constexpr uint16_t CECH_DEC     = 1u<<1;
  static constexpr uint16_t LATTICE_MONO = 1u<<2;
  static constexpr uint16_t ROBIN_CONTR  = 1u<<3;
  static constexpr uint16_t PATH_SAFE    = 1u<<4;
};

// ---------- Δ event ----------
struct Delta {
  uint8_t tenant;    // 0..(2^TENANT_BITS-1)
  uint8_t kind;      // LutKind
  uint8_t load_u, kappa, delta_tau, slack; // 0..255
  uint8_t mask4, coh4, motif;              // small fields
  uint16_t cell_i, cell_j;                 // ≤ ~1023 typical
  uint8_t cong_bin;                        // 0..3
};

// ---------- Quantization (3-bit Gray) ----------
struct Quant {
  uint8_t q_load[256], q_kappa[256], q_dt[256], q_slack[256];
  static inline uint8_t gray3(uint8_t x){ return (x ^ (x>>1)) & 7; }
  static inline uint8_t q3(uint8_t x){
    return (x<=3)?0:(x<=7)?1:(x<=15)?2:(x<=31)?3:(x<=63)?4:(x<=95)?5:(x<=159)?6:7;
  }
  Quant(){
    for(int i=0;i<256;i++){
      q_load[i]=gray3(q3(i)); q_kappa[i]=gray3(q3(i));
      q_dt[i]=gray3(q3(i));   q_slack[i]=gray3(q3(i));
    }
  }
};

// ---------- Entry ABI v1.3.1 (24B aligned) ----------
// [0..7]   tag (FNV-1a-64 over key||header) — only checked if WITNESS_ON
// [8..11]  witness_id (kept as data; ignored when WITNESS_ON=0)
// [12]     op_id (low 3b = EffectId)
// [13]     eps_cost (cat<<2 | level)
// [14..15] mono_code (Mono flags)
// [16..17] step (mix:6 | coeff:10)
// [18..22] target_40
// [23]     pad
struct Entry {
  uint64_t tag;
  uint32_t witness_id;
  uint8_t  op_id;
  uint8_t  eps_cost;
  uint16_t mono_code;
  uint16_t step;
  uint8_t  tgt40[5];
  uint8_t  pad;

  inline EffectId effect() const {
    uint8_t e = op_id & 7u;
    return e==0?EffToggle: e==1?EffConstAdd: e==2?EffIdSwap: EffHold;
  }
  inline uint64_t target_u64() const {
    return uint64_t(tgt40[0]) |
          (uint64_t(tgt40[1])<<8) |
          (uint64_t(tgt40[2])<<16)|
          (uint64_t(tgt40[3])<<24)|
          (uint64_t(tgt40[4])<<32);
  }
  inline uint16_t coeff_id() const { return step & 0x03FFu; }
  inline void header_bytes(uint8_t out[15]) const {
    out[0]=op_id; memcpy(out+1,tgt40,5);
    out[6]=uint8_t(step); out[7]=uint8_t(step>>8);
    out[8]=uint8_t(mono_code); out[9]=uint8_t(mono_code>>8);
    out[10]=eps_cost;
    out[11]=uint8_t(witness_id); out[12]=uint8_t(witness_id>>8);
    out[13]=uint8_t(witness_id>>16); out[14]=uint8_t(witness_id>>24);
  }
};
static_assert(sizeof(Entry)==24, "Entry must be 24 bytes");

static inline Entry safe_default(){
  Entry e{}; e.op_id=EffHold; e.witness_id=0xFFFF'FFFFu; return e;
}

// ---------- Fast tag (FNV-1a-64) ----------
static inline uint64_t fnv1a64(const uint8_t* p, size_t n){
  uint64_t h=1469598103934665603ull;
  for(size_t i=0;i<n;i++){ h^=p[i]; h*=1099511628211ull; }
  return h;
}

// ---------- ε-Budget ----------
struct Budget {
  uint32_t hard_cap[4], soft_cap[4], learn_cap[4];
  uint32_t hard[4]{}, soft[4]{}, learn[4]{};
  inline bool consume(uint8_t eps_cost){
    const int cat=(eps_cost>>2)&3, lvl=eps_cost&3;
    if(lvl==0){ uint32_t v=++hard[cat]; return LIKELY(v<=hard_cap[cat]); }
    if(lvl==1){ uint32_t v=++soft[cat]; return LIKELY(v<=soft_cap[cat]); }
    uint32_t v=++learn[cat]; return LIKELY(v<=learn_cap[cat]);
  }
};

// ---------- State (flat, CP-friendly) ----------
struct State {
  vector<int32_t> phi, nu, w;
  uint32_t coh_bits{0};
  vector<uint32_t> path_ids;
  // Power-of-two fast-path masks
  uint32_t phi_mask=0, w_mask=0, path_mask=0;
  // Monotone/safety flags
  uint16_t flags = Mono::PRICE_NONNEG | Mono::CECH_DEC |
                   Mono::LATTICE_MONO | Mono::ROBIN_CONTR |
                   Mono::PATH_SAFE;

  static inline bool is_pow2(size_t n){ return n && ((n&(n-1))==0); }

  State(int nv,int ne,int np):phi(nv,0),nu(ne,0),w(ne,0),path_ids(np,0){
    if(is_pow2(phi.size()))  phi_mask = uint32_t(phi.size()-1);
    if(is_pow2(w.size()))    w_mask   = uint32_t(w.size()-1);
    if(is_pow2(path_ids.size())) path_mask = uint32_t(path_ids.size()-1);
  }

  inline size_t idx_w(uint32_t raw) const {
    return LIKELY(w_mask)? (raw & w_mask) : (raw % w.size());
  }
  inline size_t idx_phi(uint32_t raw) const {
    return LIKELY(phi_mask)? (raw & phi_mask) : (raw % phi.size());
  }
  inline size_t idx_path(uint32_t raw) const {
    return LIKELY(path_mask)? (raw & path_mask) : (raw % path_ids.size());
  }
};

// ---------- LUT bank ----------
struct Bank {
  vector<Entry> robin, lattice, path, cech, urn;
  Bank(){
    Entry def=safe_default();
    robin.assign(KEY_SPACE,def);
    lattice.assign(KEY_SPACE,def);
    path.assign(KEY_SPACE,def);
    cech.assign(KEY_SPACE,def);
    urn.assign(KEY_SPACE,def);
  }
  inline const Entry& get(uint8_t kind, Key k) const {
    const size_t i = k.v & KEY_MASK;
    switch(kind){
      case BK_Robin:   return robin[i];
      case BK_Lattice: return lattice[i];
      case BK_Path:    return path[i];
      case BK_Cech:    return cech[i];
      default:         return urn[i];
    }
  }
  inline void set(uint8_t kind, Key k, const Entry& e){
    const size_t i = k.v & KEY_MASK;
    switch(kind){
      case BK_Robin:   robin[i]=e; break;
      case BK_Lattice: lattice[i]=e; break;
      case BK_Path:    path[i]=e; break;
      case BK_Cech:    cech[i]=e; break;
      default:         urn[i]=e; break;
    }
  }
};

// ---------- Quantizer ----------
struct Quantizer {
  Quant qt;
  inline Key pack(const Delta& d) const {
    const uint32_t t  = (uint32_t(d.tenant) & ((1u<<TENANT_BITS)-1)) << (KEY_BITS - TENANT_BITS);
    const uint32_t u  = qt.q_load[d.load_u] & 7u;
    const uint32_t k  = qt.q_kappa[d.kappa] & 7u;
    const uint32_t dt = qt.q_dt[d.delta_tau] & 7u;
    const uint32_t s  = qt.q_slack[d.slack] & 7u;
    const uint32_t m  = d.motif & 15u;
    const uint32_t c  = d.cong_bin & 3u;
    const uint32_t low = (u<<18)|(k<<15)|(dt<<12)|(s<<9)|(m<<5)|(c<<3);
    return Key{ (t|low) & KEY_MASK };
  }
};

// ---------- ConstAdd lookup table ----------
static constexpr int8_t DELTA[32] = {
  -16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15
};
static inline int32_t constadd(uint16_t coeff){ return DELTA[coeff & 31u]; }

// ---------- Core ----------
struct Core {
  Quantizer qz;
  Bank bank;
  State st;
  Budget bud;

  // Tag salt (unused if WITNESS_ON=0)
  uint64_t tag_k0=0x03030508d15ca11aull, tag_k1=0x0badcafedeadbeefull;

  // Metrics (optional)
  uint64_t m_total=0, m_applied=0, m_fallback=0, m_mac_fail=0, m_budget_fail=0, m_mono_fail=0;

  Core(int nv,int ne,int np)
    : st(nv,ne,np),
      bud{ {UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX},
           {1000000,1000000,1000000,1000000},
           {10000000,10000000,10000000,10000000} } {}

  inline Key key(const Delta& d) const { return qz.pack(d); }

  inline uint64_t make_tag(Key k, const Entry& e) const {
#if WITNESS_ON
    uint8_t hdr[15]; e.header_bytes(hdr);
    uint8_t buf[4+15];
    buf[0]=uint8_t(k.v); buf[1]=uint8_t(k.v>>8); buf[2]=uint8_t(k.v>>16); buf[3]=uint8_t(k.v>>24);
    memcpy(buf+4,hdr,15);
    uint64_t h = fnv1a64(buf,sizeof buf);
    // light salt mix; not cryptographic (that’s fine for OJ / integrity hint)
    h ^= tag_k0; h += (h<<7) ^ tag_k1;
    return h;
#else
    (void)k; (void)e; return 0;
#endif
  }

  inline void install(uint8_t kind, Key k, Entry e){
#if WITNESS_ON
    e.tag = make_tag(k,e);
#endif
    bank.set(kind,k,e);
  }

  // ---- effect application (branch-lean) ----
  inline void apply(const Entry& e, const Delta& d){
    const uint64_t tgt = e.target_u64();
    switch(e.effect()){
      case EffToggle: {
        st.coh_bits ^= uint32_t(tgt & 0xFFFF'FFFFull);
      } break;
      case EffConstAdd: {
        const int32_t delta = constadd(e.coeff_id());
        const size_t  idx   = st.idx_w(uint32_t(tgt & 0xFFFFull));
        // w += delta (saturating)
        int64_t t0 = int64_t(st.w[idx]) + delta;
        st.w[idx] = (t0<INT32_MIN)?INT32_MIN:(t0>INT32_MAX?INT32_MAX:int32_t(t0));
        // nu += delta; clamp ≥0 if flagged
        if(LIKELY(e.mono_code & Mono::PRICE_NONNEG)){
          int64_t t1 = int64_t(st.nu[idx]) + delta;
          st.nu[idx] = (t1<0)?0:(t1>INT32_MAX?INT32_MAX:int32_t(t1));
        }
        // phi mixer on two vertices
        const size_t v0 = st.idx_phi(uint32_t(tgt>>16));
        const size_t v1 = st.idx_phi(uint32_t(tgt>>24));
        int64_t p0 = int64_t(st.phi[v0]) + delta;
        int64_t p1 = int64_t(st.phi[v1]) - delta;
        st.phi[v0] = (p0<INT32_MIN)?INT32_MIN:(p0>INT32_MAX?INT32_MAX:int32_t(p0));
        st.phi[v1] = (p1<INT32_MIN)?INT32_MIN:(p1>INT32_MAX?INT32_MAX:int32_t(p1));
      } break;
      case EffIdSwap: {
        const size_t pid = st.idx_path(uint32_t(tgt));
        const uint32_t newid = (uint32_t(d.cell_i)<<10) | uint32_t(d.cell_j);
        if(newid < st.path_ids[pid]) st.path_ids[pid]=newid; // idempotent, deterministic
      } break;
      default: /* hold */ break;
    }
  }

  // ---- single event (O(0) gates + constant effect) ----
  inline void step(const Delta& d){
    ++m_total;
    const Key k = key(d);
    const Entry& e = bank.get(d.kind, k);

    bool macok=true, budok=true, mono=true;
#if WITNESS_ON
    macok = (e.tag == make_tag(k,e));
    if(UNLIKELY(!macok)) ++m_mac_fail;
#endif
    budok = bud.consume(e.eps_cost);
    if(UNLIKELY(!budok)) ++m_budget_fail;

    mono = ((st.flags & e.mono_code) == e.mono_code);
    if(UNLIKELY(!mono)) ++m_mono_fail;

    if(LIKELY(macok && budok && mono)){ apply(e,d); ++m_applied; }
    else { apply(safe_default(), d); ++m_fallback; }
  }

  inline void run_batch(const vector<Delta>& ds){
    for(const auto& d: ds) step(d);
  }

  inline void print_kpi() const {
    auto pct=[&](uint64_t x){ return (m_total? (100.0*x/m_total):0.0); };
    fprintf(stderr,"[GrayLUT Core v1.3.1] total=%llu applied=%llu(%.2f%%) fallback=%llu(%.2f%%) mac=%llu bud=%llu mono=%llu\n",
      (unsigned long long)m_total,(unsigned long long)m_applied,pct(m_applied),
      (unsigned long long)m_fallback,pct(m_fallback),
      (unsigned long long)m_mac_fail,(unsigned long long)m_budget_fail,(unsigned long long)m_mono_fail);
  }
};

// ---------- Minimal demo main (replace for your task) ----------
int main(){
  // Choose sizes. If you can, make them powers of two for &-mask fast path.
  const int NV = 4096, NE = 8192, NP = 2048; // all powers of two → fastest
  Core core(NV, NE, NP);

  // Build one demo entry at a key derived from a sample delta
  Delta d{0, BK_Robin, 124,42,17,88, 3,5,7, 12,34,1};
  Key key = core.key(d);

  Entry e{}; // Robin: ConstAdd, simple positive delta
  e.op_id    = EffConstAdd;
  e.eps_cost = (BC_Robin<<2) | BL_Hard;
  e.mono_code= Mono::PRICE_NONNEG | Mono::ROBIN_CONTR;
  e.step     = 0x0021;                    // coeff → DELTA[1] = +(-16+33-16?) => +1 (see table)
  e.tgt40[0]=1; e.tgt40[1]=2; e.tgt40[2]=3; e.tgt40[3]=4; e.tgt40[4]=5;
  core.install(BK_Robin, key, e);

  // Run a small batch to exercise the hot path
  int Q=10000;
  vector<Delta> ds(Q, d);
  core.run_batch(ds);

  // Print a stable value for OJ sanity (customize per problem)
  cout << core.st.w[0] << '\n';
  // core.print_kpi(); // dev only
  return 0;
}
