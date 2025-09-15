// "GrayLUT Core" v1.3.3 — hyper-tuned (safe key layout + cache mix)
// Build: g++ -O3 -pipe -flto -march=native -mtune=native -DNDEBUG -o glk_core_v1_3_3 glk_core_v1_3_3.cpp

#ifndef KEY_BITS
#define KEY_BITS 20  // ← デフォを20に。TENANT_BITS=2を活かす（18ならTENANT_BITS=0に）
#endif
#ifndef TENANT_BITS
#define TENANT_BITS 2
#endif
#ifndef WITNESS_ON
#define WITNESS_ON 0
#endif
#ifndef HOTCACHE_ON
#define HOTCACHE_ON 1
#endif
#ifndef PREFETCH_ON
#define PREFETCH_ON 1
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

// ---- small types/enums
struct Key{ uint32_t v; };
enum EffectId: uint8_t { EffToggle=0, EffConstAdd=1, EffIdSwap=2, EffHold=7 };
enum LutKind : uint8_t { BK_Robin=0, BK_Lattice=1, BK_Path=2, BK_Cech=3, BK_Urn=4 };
enum BudgetCat: uint8_t { BC_Robin=0, BC_Price=1, BC_Cech=2, BC_Hodge=3 };
enum BudgetLv : uint8_t { BL_Hard=0, BL_Soft=1, BL_Learn=2 };
struct Mono{ static constexpr uint16_t PRICE_NONNEG=1u<<0, CECH_DEC=1u<<1, LATTICE_MONO=1u<<2, ROBIN_CONTR=1u<<3, PATH_SAFE=1u<<4; };

// ---- Δ
struct Delta{
  uint8_t tenant, kind;
  uint8_t load_u, kappa, delta_tau, slack;
  uint8_t mask4, coh4, motif;
  uint16_t cell_i, cell_j;
  uint8_t cong_bin;
};

// ---- quantization (3-bit Gray)
struct Quant{
  uint8_t q_load[256], q_kappa[256], q_dt[256], q_slack[256];
  static inline uint8_t g3(uint8_t x){ return (x^(x>>1)) & 7; }
  static inline uint8_t q3(uint8_t x){
    return (x<=3)?0:(x<=7)?1:(x<=15)?2:(x<=31)?3:(x<=63)?4:(x<=95)?5:(x<=159)?6:7;
  }
  Quant(){ for(int i=0;i<256;i++){ q_load[i]=g3(q3(i)); q_kappa[i]=g3(q3(i)); q_dt[i]=g3(q3(i)); q_slack[i]=g3(q3(i)); } }
};

// ---- Entry ABI (24B)
struct Entry{
  uint64_t tag;
  uint32_t witness_id;
  uint8_t  op_id, eps_cost;
  uint16_t mono_code, step;
  uint8_t  tgt40[5], pad;
  inline EffectId effect() const { uint8_t e=op_id&7u; return e==0?EffToggle:e==1?EffConstAdd:e==2?EffIdSwap:EffHold; }
  inline uint64_t tgt() const {
    return uint64_t(tgt40[0]) | (uint64_t(tgt40[1])<<8) | (uint64_t(tgt40[2])<<16) |
           (uint64_t(tgt40[3])<<24) | (uint64_t(tgt40[4])<<32);
  }
  inline uint16_t coeff() const { return step & 0x03FFu; }
  inline void hdr(uint8_t out[15]) const {
    out[0]=op_id; memcpy(out+1,tgt40,5);
    out[6]=uint8_t(step); out[7]=uint8_t(step>>8);
    out[8]=uint8_t(mono_code); out[9]=uint8_t(mono_code>>8);
    out[10]=eps_cost;
    out[11]=uint8_t(witness_id); out[12]=uint8_t(witness_id>>8);
    out[13]=uint8_t(witness_id>>16); out[14]=uint8_t(witness_id>>24);
  }
};
static_assert(sizeof(Entry)==24,"Entry must be 24 bytes");
static inline Entry safe_default(){ Entry e{}; e.op_id=EffHold; e.witness_id=0xFFFF'FFFFu; return e; }

// ---- tag (FNV-1a-64)
static inline uint64_t fnv1a(const uint8_t* p, size_t n){ uint64_t h=1469598103934665603ull; for(size_t i=0;i<n;i++){ h^=p[i]; h*=1099511628211ull; } return h; }

// ---- Budget
struct Budget{
  uint32_t hard_cap[4], soft_cap[4], learn_cap[4];
  uint32_t hard[4]{}, soft[4]{}, learn[4]{};
  inline bool use(uint8_t eps){
    const int c=(eps>>2)&3, l=eps&3;
    if(l==0){ uint32_t v=++hard[c]; return LIKELY(v<=hard_cap[c]); }
    if(l==1){ uint32_t v=++soft[c]; return LIKELY(v<=soft_cap[c]); }
    uint32_t v=++learn[c]; return LIKELY(v<=learn_cap[c]);
  }
};

// ---- State
struct State{
  vector<int32_t> phi, nu, w;
  vector<uint32_t> path;
  uint32_t coh_bits{0};
  uint16_t flags = Mono::PRICE_NONNEG|Mono::CECH_DEC|Mono::LATTICE_MONO|Mono::ROBIN_CONTR|Mono::PATH_SAFE;
  uint32_t m_phi=0, m_w=0, m_path=0;
  static inline bool p2(size_t n){ return n && ((n&(n-1))==0); }
  State(int nv,int ne,int np):phi(nv,0),nu(ne,0),w(ne,0),path(np,0){
    if(p2(phi.size())) m_phi=phi.size()-1;
    if(p2(w.size()))   m_w  =w.size()-1;
    if(p2(path.size()))m_path=path.size()-1;
  }
  inline size_t iw(uint32_t r)   const { return LIKELY(m_w)?   (r & m_w)   : (r % w.size()); }
  inline size_t iphi(uint32_t r) const { return LIKELY(m_phi)? (r & m_phi) : (r % phi.size()); }
  inline size_t ipath(uint32_t r)const { return LIKELY(m_path)?(r & m_path): (r % path.size()); }
};

// ---- Bank (dense arrays + raw pointer access)
struct Bank{
  vector<Entry> robin,lattice,path,cech,urn;
  Bank(){ Entry d=safe_default(); robin.assign(KEY_SPACE,d); lattice.assign(KEY_SPACE,d); path.assign(KEY_SPACE,d); cech.assign(KEY_SPACE,d); urn.assign(KEY_SPACE,d); }
  inline const Entry& get(uint8_t kind, Key k) const {
    const size_t i = k.v & KEY_MASK;
    switch(kind){ case BK_Robin: return robin[i]; case BK_Lattice: return lattice[i]; case BK_Path: return path[i]; case BK_Cech: return cech[i]; default: return urn[i]; }
  }
  inline void set(uint8_t kind, Key k, const Entry& e){
    const size_t i = k.v & KEY_MASK;
    switch(kind){ case BK_Robin: robin[i]=e; break; case BK_Lattice: lattice[i]=e; break; case BK_Path: path[i]=e; break; case BK_Cech: cech[i]=e; break; default: urn[i]=e; break; }
  }
  inline const Entry* base(uint8_t kind) const {
    switch(kind){ case BK_Robin: return robin.data(); case BK_Lattice: return lattice.data(); case BK_Path: return path.data(); case BK_Cech: return cech.data(); default: return urn.data(); }
  }
};

// ---- Quantizer
struct Quantizer{
  Quant qt;

  // bit layout safety
  static constexpr uint32_t U=3, K=3, DT=3, S=3, M=4, C=2;
  static constexpr uint32_t LOW = U+K+DT+S+M+C; // 18
  static_assert(LOW <= (uint32_t)KEY_BITS, "LOW fields overflow KEY_BITS");
  static_assert(LOW + TENANT_BITS <= (uint32_t)KEY_BITS, "TENANT bits overflow KEY_BITS");

  inline Key key(const Delta& d) const {
    uint32_t pos = 0;

    const uint32_t c  = (uint32_t(d.cong_bin) & ((1u<<C)-1))                 << pos; pos+=C;
    const uint32_t m  = (uint32_t(d.motif)    & ((1u<<M)-1))                 << pos; pos+=M;
    const uint32_t s  = (uint32_t(qt.q_slack[d.slack])     & ((1u<<S)-1))    << pos; pos+=S;
    const uint32_t dt = (uint32_t(qt.q_dt[d.delta_tau])    & ((1u<<DT)-1))   << pos; pos+=DT;
    const uint32_t k  = (uint32_t(qt.q_kappa[d.kappa])     & ((1u<<K)-1))    << pos; pos+=K;
    const uint32_t u  = (uint32_t(qt.q_load[d.load_u])     & ((1u<<U)-1))    << pos; pos+=U;

    uint32_t low = u|k|dt|s|m|c;

    uint32_t t = 0;
    if constexpr (TENANT_BITS>0){
      const uint32_t TPOS = KEY_BITS - TENANT_BITS;
      t = (uint32_t(d.tenant) & ((1u<<TENANT_BITS)-1)) << TPOS;
    }
    return Key{ (t | low) & KEY_MASK };
  }
};

// ---- ConstAdd table
static constexpr int8_t DELTA[32]={ -16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
static inline int32_t cadd(uint16_t c){ return DELTA[c & 31u]; }

// ---- HotCache (direct-mapped 1K) with better mixing
#if HOTCACHE_ON
static inline uint32_t rotl32(uint32_t x, unsigned r){ return (x<<r) | (x>>(32-r)); }
struct HotCache{
  static constexpr size_t S=1024, M=S-1;
  Entry  e[S];
  uint32_t tag[S]{}; // kind-mix ^ key
  inline size_t idx(uint8_t kind, Key k) const {
    // mix: multiplicative + rotate; low bitsも撹拌
    uint32_t h = (k.v * 0x9E3779B1u) ^ rotl32(uint32_t(kind)*0x85ebca6bu, 11);
    h ^= (h>>16);
    return h & M;
  }
  inline const Entry& get(const Bank& b, uint8_t kind, Key k){
    const uint32_t tg = rotl32(k.v, 7) ^ (uint32_t(kind)*0x9E3779B1u);
    size_t i = idx(kind,k);
    if(LIKELY(tag[i]==tg)) return e[i];
    const Entry& ref=b.get(kind,k);
    e[i]=ref; tag[i]=tg; return e[i];
  }
};
#endif

// ---- Core
struct Core{
  Quantizer qz; Bank bank; State st; Budget bud;
  uint64_t k0=0x03030508d15ca11aull, k1=0x0badcafedeadbeefull; // salt
#if HOTCACHE_ON
  HotCache hc;
#endif
  // metrics
  uint64_t m_total=0,m_apply=0,m_fallback=0,m_mac=0,m_bud=0,m_mono=0;

  Core(int nv,int ne,int np)
    : st(nv,ne,np),
      bud{ {UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX},
           {1000000,1000000,1000000,1000000},
           {10000000,10000000,10000000,10000000} } {}

  inline Key key(const Delta& d) const { return qz.key(d); }

  inline uint64_t tag_make(Key k, const Entry& e) const {
#if WITNESS_ON
    uint8_t hdr[15]; e.hdr(hdr);
    uint8_t kbuf[4] = { uint8_t(k.v), uint8_t(k.v>>8), uint8_t(k.v>>16), uint8_t(k.v>>24) };
    // incremental FNV: kbuf → hdr
    uint64_t h=1469598103934665603ull;
    for(int i=0;i<4;i++){ h^=kbuf[i]; h*=1099511628211ull; }
    for(int i=0;i<15;i++){ h^=hdr[i];  h*=1099511628211ull; }
    h^=k0; h+=(h<<7)^k1; return h;
#else
    (void)k; (void)e; return 0;
#endif
  }

  inline void install(uint8_t kind, Key k, Entry e){
#if WITNESS_ON
    e.tag = tag_make(k,e);
#endif
    bank.set(kind,k,e);
  }

  inline void apply(const Entry& e, const Delta& d){
    const uint64_t t = e.tgt();
    switch(e.effect()){
      case EffToggle:{
        st.coh_bits ^= uint32_t(t & 0xFFFF'FFFFull);
      }break;
      case EffConstAdd:{ // ← hot path
        const int32_t dv = cadd(e.coeff());
        const size_t  wi = st.iw(uint32_t(t & 0xFFFFull));
        // w
        int64_t wv = int64_t(st.w[wi]) + dv;
        st.w[wi] = (wv<INT32_MIN)?INT32_MIN:(wv>INT32_MAX?INT32_MAX:int32_t(wv));
        // nu
        if(LIKELY(e.mono_code & Mono::PRICE_NONNEG)){
          int64_t nv = int64_t(st.nu[wi]) + dv;
          st.nu[wi] = (nv<0)?0:(nv>INT32_MAX?INT32_MAX:int32_t(nv));
        }
        // phi
        const size_t v0 = st.iphi(uint32_t(t>>16)), v1 = st.iphi(uint32_t(t>>24));
        int64_t p0 = int64_t(st.phi[v0]) + dv, p1 = int64_t(st.phi[v1]) - dv;
        st.phi[v0] = (p0<INT32_MIN)?INT32_MIN:(p0>INT32_MAX?INT32_MAX:int32_t(p0));
        st.phi[v1] = (p1<INT32_MIN)?INT32_MIN:(p1>INT32_MAX?INT32_MAX:int32_t(p1));
      }break;
      case EffIdSwap:{
        const size_t pi = st.ipath(uint32_t(t));
        const uint32_t nid = (uint32_t(d.cell_i)<<10) | uint32_t(d.cell_j);
        if(nid < st.path[pi]) st.path[pi] = nid;
      }break;
      default: break;
    }
  }

  inline void step(const Delta& d){
    ++m_total;
    const Key k = key(d);
#if HOTCACHE_ON
    const Entry& e = hc.get(bank, d.kind, k);
#else
    const Entry& e = bank.get(d.kind, k);
#endif

    // Gate order: budget → mono → witness
    bool ok_b = bud.use(e.eps_cost); if(UNLIKELY(!ok_b)) ++m_bud;
    bool ok_m = ((st.flags & e.mono_code) == e.mono_code); if(UNLIKELY(!ok_m)) ++m_mono;

    bool ok_w = true;
#if WITNESS_ON
    ok_w = (e.tag == tag_make(k,e));
    if(UNLIKELY(!ok_w)) ++m_mac;
#endif

    if(LIKELY(ok_b && ok_m && ok_w)){ apply(e,d); ++m_apply; }
    else { apply(safe_default(), d); ++m_fallback; }
  }

  // batch with prefetch of next row (LUT)
  inline void run_batch(const vector<Delta>& ds){
#if PREFETCH_ON
    for(size_t i=0;i<ds.size();++i){
      const Delta& d = ds[i];
      if(LIKELY(i+1<ds.size())){
        Key k2 = key(ds[i+1]);
        const Entry* p = bank.base(ds[i+1].kind) + (k2.v & KEY_MASK);
        __builtin_prefetch((const void*)p, 0, 1);
      }
      step(d);
    }
#else
    for(const auto& d: ds) step(d);
#endif
  }
};

// ---- Demo main (unchanged except KEY_BITS default)
int main(){
  const int NV=4096, NE=8192, NP=2048;
  Core c(NV,NE,NP);

  Delta d{0, BK_Robin, 124,42,17,88, 3,5,7, 12,34,1};
  Key k = c.key(d);

  Entry e{}; e.op_id=EffConstAdd; e.eps_cost=(BC_Robin<<2)|BL_Hard;
  e.mono_code = Mono::PRICE_NONNEG|Mono::ROBIN_CONTR; e.step=0x0021;
  e.tgt40[0]=1; e.tgt40[1]=2; e.tgt40[2]=3; e.tgt40[3]=4; e.tgt40[4]=5;
  c.install(BK_Robin, k, e);

  int Q=10000; vector<Delta> ds(Q, d);
  c.run_batch(ds);

  cout << c.st.w[0] << '\n';
  return 0;
}


