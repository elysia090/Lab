// glk_core_ultra_soa.rs — GrayLUT Core (Rust, SoA + 2way cache + pipelined prefetch)
// Build: rustc -C opt-level=3 -C target-cpu=native glk_core_ultra_soa.rs
#![allow(non_camel_case_types,dead_code)]
use std::cmp::max;

type EId=u8; const TG:EId=0; const AD:EId=1; const SW:EId=2; const HD:EId=7;
type Knd=u8; const R:Knd=0; const L:Knd=1; const P:Knd=2; const C:Knd=3; const U:Knd=4;

const KB:u32=20; const TB:u32=2; const KS:usize=1usize<<KB; const KM:u32=(1u32<<KB)-1;
const WIT:bool=false; const PF:bool=true; const FAST_UNSAFE:bool=false;

mod M{ pub const NN:u16=1<<0; pub const CD:u16=1<<1; pub const LM:u16=1<<2; pub const RC:u16=1<<3; pub const PS:u16=1<<4; }

#[derive(Copy,Clone,Default)] struct D{ tenant:u8,kind:Knd,load_u:u8,kappa:u8,delta_tau:u8,slack:u8,mask4:u8,coh4:u8,motif:u8,cell_i:u16,cell_j:u16,cong_bin:u8 }

#[derive(Copy,Clone,Default)] struct Hot{ op:EId,eps:u8,mono:u16,step:u16,tgt:[u8;5],pad:u8 }
#[derive(Copy,Clone,Default)] struct Cold{ tag:u64,wid:u32 }
#[inline] fn hdef()->Hot{ let mut h=Hot::default(); h.op=HD; h }
#[inline] fn rotl(x:u32,r:u32)->u32{ x.rotate_left(r) }
#[inline] fn q3(x:u8)->u8{ if x<=3{0}else if x<=7{1}else if x<=15{2}else if x<=31{3}else if x<=63{4}else if x<=95{5}else if x<=159{6}else{7} }
#[inline] fn q(x:u8)->u32{ (q3(x) ^ (q3(x)>>1)) as u32 } // g3(q3(x))
#[inline] fn tgt40(t:&[u8;5])->u64{ (t[0] as u64)|((t[1] as u64)<<8)|((t[2] as u64)<<16)|((t[3] as u64)<<24)|((t[4] as u64)<<32) }
#[inline] fn cadd(c:u16)->i32{ (c as i32 & 31)-16 }

#[derive(Clone)] struct B{ cap:[u32;12], cnt:[u32;12] }
impl B{
  fn new()->Self{ B{ cap:[u32::MAX,u32::MAX,u32::MAX,u32::MAX,1_000_000,1_000_000,1_000_000,1_000_000,10_000_000,10_000_000,10_000_000,10_000_000], cnt:[0;12]} }
  #[inline] fn use_(&mut self,e:u8)->bool{ let i=((e&3) as usize)*4+(((e>>2)&3) as usize); self.cnt[i]+=1; self.cnt[i]<=self.cap[i] }
}

#[derive(Clone)] struct S{ phi:Vec<i32>,nu:Vec<i32>,w:Vec<i32>,path:Vec<u32>,coh:u32,fl:u16,mp:u32,mw:u32,mt:u32 }
#[inline] fn p2(n:usize)->bool{ n!=0&&(n&(n-1))==0 }
impl S{
  fn new(nv:usize,ne:usize,np:usize)->Self{
    let (mp,mw,mt)=(if p2(nv){(nv-1) as u32}else{0},if p2(ne){(ne-1) as u32}else{0},if p2(np){(np-1) as u32}else{0});
    S{phi:vec![0;nv],nu:vec![0;ne],w:vec![0;ne],path:vec![0;np],coh:0,fl:M::NN|M::CD|M::LM|M::RC|M::PS,mp,mw,mt}
  }
  #[inline] fn iw(&self,r:u32)->usize{ if self.mw!=0{(r&self.mw) as usize}else{(r as usize)%self.w.len()} }
  #[inline] fn ip(&self,r:u32)->usize{ if self.mp!=0{(r&self.mp) as usize}else{(r as usize)%self.phi.len()} }
  #[inline] fn it(&self,r:u32)->usize{ if self.mt!=0{(r&self.mt) as usize}else{(r as usize)%self.path.len()} }
}

#[derive(Clone)] struct K{ hot:[Vec<Hot>;5], cold:[Vec<Cold>;5] }
impl K{
  fn new()->Self{
    let h=hdef(); let c=Cold::default();
    K{ hot:[vec![h;KS],vec![h;KS],vec![h;KS],vec![h;KS],vec![h;KS]],
       cold:[vec![c;KS],vec![c;KS],vec![c;KS],vec![c;KS],vec![c;KS]] }
  }
  #[inline] fn gh(&self,k:Knd,i:u32)->Hot{ self.hot[k as usize][(i&KM) as usize] }
  #[inline] fn gc(&self,k:Knd,i:u32)->Cold{ self.cold[k as usize][(i&KM) as usize] }
  #[inline] fn sh(&mut self,k:Knd,i:u32,e:Hot){ self.hot[k as usize][(i&KM) as usize]=e; }
  #[inline] fn sc(&mut self,k:Knd,i:u32,c:Cold){ self.cold[k as usize][(i&KM) as usize]=c; }
  #[inline] fn base(&self,k:Knd)->*const Hot{ self.hot[k as usize].as_ptr() }
}

#[derive(Clone)] struct H2{ e:[Hot;2048], t:[u32;2048] } // 2-way
impl H2{
  fn new()->Self{ H2{ e:[hdef();2048], t:[0;2048] } }
  #[inline] fn idx(k:Knd,x:u32)->(usize,usize){ let h=(x.wrapping_mul(0x9E3779B1)^rotl((k as u32).wrapping_mul(0x85ebca6b),11))^((x>>16)|1); let i=(h&1023) as usize; (i, i^1024) }
  #[inline] fn get(&mut self,b:&K,k:Knd,x:u32)->Hot{
    let tg=rotl(x,7)^((k as u32).wrapping_mul(0x9E3779B1)); let (i0,i1)=Self::idx(k,x);
    let t0=self.t[i0]==tg; if t0 { return self.e[i0]; }
    if self.t[i1]==tg { return self.e[i1]; }
    let v=b.gh(k,x); let r= if (tg&1)==0 {i0} else {i1}; self.e[r]=v; self.t[r]=tg; v
  }
}

struct C{ s:S, bk:K, bd:B, k0:u64, k1:u64, hc:H2, tot:u64, ap:u64, fb:u64, mac:u64, bm:u64, mm:u64 }
impl C{
  fn new(nv:usize,ne:usize,np:usize)->Self{
    C{ s:S::new(nv,ne,np), bk:K::new(), bd:B::new(), k0:0x03030508d15ca11a, k1:0x0badcafedeadbeef,
       hc:H2::new(), tot:0, ap:0, fb:0, mac:0, bm:0, mm:0 }
  }
  #[inline(always)] fn key(&self,d:&D)->u32{
    let (mut p,c,m,s,dt,k,u)=(0u32,(d.cong_bin&3) as u32,(d.motif&15) as u32,q(d.slack),q(d.delta_tau),q(d.kappa),q(d.load_u));
    let mut low=(c<<p); p+=2; low|=(m<<p); p+=4; low|=(s<<p); p+=3; low|=(dt<<p); p+=3; low|=(k<<p); p+=3; low|=(u<<p);
    let t=if TB>0{((d.tenant as u32)&((1<<TB)-1))<<(KB-TB)}else{0}; (t|low)&KM
  }
  #[inline] fn make_tag(&self,k:u32,h:&Hot,c:&Cold)->u64{
    if !WIT {return 0;}
    let kb=[k as u8,(k>>8) as u8,(k>>16) as u8,(k>>24) as u8];
    let mut v=[0u8;15]; v[0]=h.op; v[1..6].copy_from_slice(&h.tgt); v[6]=h.step as u8; v[7]=(h.step>>8) as u8; v[8]=h.mono as u8; v[9]=(h.mono>>8) as u8; v[10]=h.eps;
    v[11]=c.wid as u8; v[12]=(c.wid>>8) as u8; v[13]=(c.wid>>16) as u8; v[14]=(c.wid>>24) as u8;
    let mut H: u64 = 1469598103934665603; for &b in kb.iter().chain(v.iter()){ H^=b as u64; H=H.wrapping_mul(1099511628211); }
    (H^self.k0).wrapping_add((H<<7)^self.k1)
  }
  #[inline] fn put(&mut self,k:Knd,i:u32,mut h:Hot,mut c:Cold){ if WIT{ c.tag=self.make_tag(i,&h,&c) } self.bk.sh(k,i,h); self.bk.sc(k,i,c); }

  #[inline(always)] fn exec(&mut self,d:&D,h:Hot,c:&Cold){
    self.tot+=1;
    let okb=self.bd.use_(h.eps); if !okb{self.bm+=1;}
    let okm=(self.s.fl & h.mono)==h.mono; if !okm{self.mm+=1;}
    let okw= if WIT{ let k=self.key(d); let g=c.tag==self.make_tag(k,&h,&c); if !g{self.mac+=1}; g } else { true };
    if okb && okm && okw {
      match h.op&7 {
        TG => { self.s.coh^=(tgt40(&h.tgt) as u32); }
        AD => { let t=tgt40(&h.tgt); let wi=self.s.iw(t as u32); let dv=cadd(h.step&0x03FF);
          self.s.w[wi]=self.s.w[wi].saturating_add(dv);
          if (h.mono & M::NN)!=0 { self.s.nu[wi]=max(0,self.s.nu[wi].saturating_add(dv)); }
          let v0=self.s.ip((t>>16) as u32); let v1=self.s.ip((t>>24) as u32);
          self.s.phi[v0]=self.s.phi[v0].saturating_add(dv);
          self.s.phi[v1]=self.s.phi[v1].saturating_sub(dv);
        }
        SW => { let t=tgt40(&h.tgt); let pi=self.s.it(t as u32); let nid=((d.cell_i as u32)<<10)|(d.cell_j as u32); if nid<self.s.path[pi]{ self.s.path[pi]=nid; } }
        _ => {}
      }
      self.ap+=1;
    } else { self.fb+=1; }
  }

  fn run(&mut self,ds:&[D]){
    if ds.is_empty(){return;}
    let mut h=self.hc.get(&self.bk,ds[0].kind,self.key(&ds[0])); // warm
    let mut c=self.bk.gc(ds[0].kind,self.key(&ds[0]));
    for i in 0..ds.len(){
      if PF && i+1<ds.len(){
        let k2=self.key(&ds[i+1]); // prefetch next LUT hot
        #[cfg(target_arch="x86_64")] unsafe{
          use std::arch::x86_64::_mm_prefetch; _mm_prefetch(self.bk.base(ds[i+1].kind).wrapping_add((k2&KM) as usize) as *const i8,3);
        }
      }
      // execute current
      self.exec(&ds[i],h,c);
      // fetch next + prefetch state rows
      if i+1<ds.len(){
        let d2=&ds[i+1]; let k2=self.key(d2);
        h=self.hc.get(&self.bk,d2.kind,k2); c=self.bk.gc(d2.kind,k2);
        if PF {
          let t=tgt40(&h.tgt) as u32; let wi=self.s.iw(t); let v0=self.s.ip(t>>16); let v1=self.s.ip(t>>24);
          #[cfg(target_arch="x86_64")] unsafe{
            use std::arch::x86_64::_mm_prefetch;
            let (pw,pn,pp0,pp1)=(self.s.w.as_ptr().add(wi), self.s.nu.as_ptr().add(wi), self.s.phi.as_ptr().add(v0), self.s.phi.as_ptr().add(v1));
            _mm_prefetch(pw as *const i8,3); _mm_prefetch(pn as *const i8,3); _mm_prefetch(pp0 as *const i8,3); _mm_prefetch(pp1 as *const i8,3);
          }
        }
      }
    }
  }
}

fn main(){
  let (nv,ne,np)=(4096,8192,2048); let mut c=C::new(nv,ne,np);
  let d=D{ tenant:0,kind:R,load_u:124,kappa:42,delta_tau:17,slack:88,mask4:3,coh4:5,motif:7,cell_i:12,cell_j:34,cong_bin:1 };
  let k=c.key(&d);
  let mut h=Hot::default(); h.op=AD; h.eps=(0<<2)|0; h.mono=M::NN|M::RC; h.step=0x0021; h.tgt=[1,2,3,4,5];
  let mut z=Cold::default(); if WIT{ z.wid=0x1234_5678; } // witness例
  c.put(R,k,h,z);
  c.run(&vec![d;10_000]);
  println!("{}",c.s.w[0]);
}
