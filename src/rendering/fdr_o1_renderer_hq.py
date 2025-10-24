
# FDR-O1-HQ v1.2  (Fixed-DAG Renderer, O(1)/px) — High-Quality Pack integrated
# Single-file, bin-less, deterministic. All per-pixel work and frame control are O(1).
# Additions vs FDR-O1: Toksvig-like specular AA, multi-scatter diffuse compensation,
# two-radius shadow blend (fixed eta), clearcoat micro-lobe, YCoCg clamp in TAA,
# log-mean exposure estimate from 256 samples, SO LUT 32x32.

import numpy as np
PI = np.float32(np.pi); CT_EPS = 1e-6

def sat(x): return np.clip(x, 0.0, 1.0).astype(np.float32)

# ---------- small fixed stencils ----------
def sobel_pair(img):
    H,W = img.shape
    pad = np.pad(img, ((1,1),(1,1)), mode='edge')
    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], np.float32)
    ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32)
    gx = np.zeros_like(img, np.float32); gy = np.zeros_like(img, np.float32)
    for j in range(3):
        for i in range(3):
            gx += kx[j,i] * pad[j:j+H, i:i+W]
            gy += ky[j,i] * pad[j:j+H, i:i+W]
    return gx, gy

def sobel_mag(img): gx,gy = sobel_pair(img); return np.abs(gx) + np.abs(gy)

def laplacian5(img):
    H,W = img.shape
    pad = np.pad(img, ((1,1),(1,1)), mode='edge')
    k = np.array([[0,1,0],[1,-4,1],[0,1,0]], np.float32)
    out = np.zeros_like(img, np.float32)
    for j in range(3):
        for i in range(3):
            out += k[j,i] * pad[j:j+H, i:i+W]
    return out

# ---------- deterministic 256-sample indices (LCG) ----------
def lcg_indices(H, W, count=256, seed=1337):
    a=1103515245; c=12345; m=2**31; s=seed; xs=[]; ys=[]
    for k in range(count):
        s=(a*s+c)%m; x=s%W
        s=(a*s+c)%m; y=s%H
        xs.append(int(x)); ys.append(int(y))
    return np.array(xs), np.array(ys)

# ---------- LUT builders (init-only) ----------
def build_fgd_lut(nv_res=32, a_res=32):
    NoV = np.linspace(1e-6, 1.0, nv_res, dtype=np.float32)
    alpha = np.linspace(1e-3, 1.0, a_res, dtype=np.float32)
    A = np.zeros((a_res, nv_res), dtype=np.float32)
    B = np.zeros((a_res, nv_res), dtype=np.float32)
    F0 = 0.04
    # simple analytic proxies to avoid sampling; stable and bin-less
    for ai,a in enumerate(alpha):
        A[ai,:] = 1.0 - (1.0 - F0) * (1.0 - NoV)**5       # Fresnel mean term
        B[ai,:] = 1.0 / (1.0 + 5.0*a)                     # geometry-roughness proxy
    return A,B

def build_so_lut(a_res=32, ao_res=32):
    a = np.linspace(1e-3, 1.0, a_res, dtype=np.float32)
    ao = np.linspace(0.0, 1.0, ao_res, dtype=np.float32)
    K = np.zeros((ao_res, a_res), dtype=np.float32)
    for yi, y in enumerate(ao):
        for xi, x in enumerate(a):
            k = 0.9 + 0.3*(x**0.5)*(1.0 - y) - 0.1*(1.0 - x)*(y)
            K[yi,xi] = np.clip(k, 0.6, 1.2)
    return K

_FGD_A, _FGD_B = build_fgd_lut(32,32)
_SO_LUT       = build_so_lut(32,32)

def lut_bilerp(arr, u, v):
    H,W = arr.shape[:2]
    x = np.clip(u*(W-1), 0, W-1-1e-6)
    y = np.clip(v*(H-1), 0, H-1-1e-6)
    ix = np.floor(x).astype(np.int32); tx = (x - ix).astype(np.float32)
    iy = np.floor(y).astype(np.int32); ty = (y - iy).astype(np.float32)
    f00 = arr[iy, ix];    f10 = arr[iy, ix+1]
    f01 = arr[iy+1, ix];  f11 = arr[iy+1, ix+1]
    l0 = (1-tx)*f00 + tx*f10
    l1 = (1-tx)*f01 + tx*f11
    return ((1-ty)*l0 + ty*l1).astype(np.float32)

# ---------- STATS256 ----------
def stats256(rgb, depth, hist_luma=None):
    L = (0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]).astype(np.float32)
    gxL, gyL = sobel_pair(L); gxZ, gyZ = sobel_pair(depth.astype(np.float32))
    gL = np.abs(gxL) + np.abs(gyL); gZ = np.abs(gxZ) + np.abs(gyZ)
    norm = np.sqrt(gxL*gxL + gyL*gyL + 1e-6)
    gT = np.abs(gxL*(gxL/norm) + gyL*(gyL/norm))
    # 3x3 kurtosis
    H,W = L.shape; pad = np.pad(L, ((1,1),(1,1)), mode='edge')
    kurt = np.zeros_like(L, np.float32)
    for y in range(H):
        for x in range(W):
            win = pad[y:y+3, x:x+3].reshape(-1)
            mu = np.mean(win); s2 = np.mean((win-mu)**2)+1e-6; s4 = np.mean((win-mu)**4)
            kurt[y,x] = s4/(s2*s2) - 3.0
    tv_map = np.zeros((H,W), np.float32) if hist_luma is None else np.abs(hist_luma - L).astype(np.float32)
    xs, ys = lcg_indices(H, W, 256, seed=1337)
    muL = np.mean(L[ys, xs]); gL_m = np.mean(gL[ys, xs]); gT_m = np.mean(gT[ys, xs]); gZ_m = np.mean(gZ[ys, xs]); kt_m = np.mean(kurt[ys, xs]); tv_m = np.mean(tv_map[ys, xs])
    return np.array([muL, gL_m, gT_m, gZ_m, kt_m, tv_m], dtype=np.float32)

# ---------- allocator: CTPA-S2 ----------
W_alloc = np.array([
    [ 1.2,  0.0,  0.0,  0.0,  0.0, -0.6],  # exposure
    [ 0.0,  1.0, -0.4,  0.0,  0.0,  0.1],  # shadow
    [ 0.0,  0.0,  0.7,  0.3,  0.0,  0.0],  # fusion
    [ 0.0,  0.3,  0.0,  0.7,  0.0,  0.0],  # TAA
    [ 0.2,  0.2,  0.0,  0.0,  0.2,  0.2],  # IBL
    [ 0.0,  0.1,  0.2,  0.0,  0.3,  0.2],  # SSS
], dtype=np.float32)
b_alloc = np.array([-0.5, 0.05, -0.1, 0.0, 0.0, 0.0], dtype=np.float32)
c_cost  = np.array([0.10, 0.20, 0.10, 0.10, 0.25, 0.25], dtype=np.float32)
B_budget= np.float32(0.90)

def allocate(S):
    q = W_alloc @ S.astype(np.float32) + b_alloc
    theta = sat(q)
    scale = np.float32(min(1.0, float(B_budget / (float(np.dot(c_cost, theta)) + 1e-6))))
    theta = theta * scale
    # discrete knobs
    exp_target = [0.42, 0.46, 0.50][min(2, int(theta[0]*3.0))]
    pcss_scale = np.float32(0.80 + 0.30*theta[1])
    shadow_enable = 1 if theta[1] > 0.05 else 0
    fusion_id = 'A' if theta[2] < 0.5 else 'B'
    taa_alpha = [0.06, 0.12, 0.20][min(2, int(theta[3]*3.0))]
    ibl_scale = np.float32(1.5 + 1.1*theta[4])
    pref_env  = np.float32(0.09 + 0.07*theta[4])
    sss_Fdr   = np.float32(0.50 + 0.18*theta[5])
    sss_alb   = np.float32(0.18 + 0.10*theta[5])
    spec_gain = np.float32(1.7)  # slightly higher default for HQ
    return {"exp_target":np.float32(exp_target), "pcss_scale":pcss_scale, "shadow_enable":int(shadow_enable),
            "fusion_id":fusion_id, "taa_alpha":np.float32(taa_alpha), "ibl_scale":ibl_scale, "pref_env":pref_env,
            "sss_Fdr":sss_Fdr, "sss_albedo":sss_alb, "spec_gain":spec_gain, "theta":theta, "scale":scale}

# ---------- BRDF / IBL ----------
def _schlick(F0, VH): return F0 + (1.0 - F0) * (1.0 - np.clip(VH,0.0,1.0))**5
def _smith_g1(x, k): return x / (x*(1.0 - k) + k + 1e-6)

def brdf_ggx(N,V,L, rough, F0_rgb):
    H = (V + L); H = H / (np.linalg.norm(H, axis=-1, keepdims=True) + 1e-6)
    NoV = np.clip(np.sum(N*V, axis=-1), 0.0, 1.0)
    NoL = np.clip(np.sum(N*L, axis=-1), 0.0, 1.0)
    HoN = np.clip(np.sum(H*N, axis=-1), 0.0, 1.0)
    VH  = np.clip(np.sum(V*H, axis=-1), 0.0, 1.0)
    a = np.clip(rough, 1e-3, 1.0); a2 = a*a
    denom = (HoN*HoN*(a2 - 1.0) + 1.0)
    D = (a2 / (PI * denom * denom + 1e-6)).astype(np.float32)
    k = (a * np.sqrt(2.0/PI)).astype(np.float32)
    G = (_smith_g1(NoV, k) * _smith_g1(NoL, k)).astype(np.float32)
    F = _schlick(F0_rgb, VH[...,None]).astype(np.float32)
    denom_gl = (4.0*np.maximum(NoV*NoL, 1e-6)).astype(np.float32)
    return (D[...,None] * G[...,None] / denom_gl[...,None]) * F

def brdf_clearcoat(N,V,L, alpha=0.08, F0=0.04, weight=0.06):
    H = (V + L); H = H / (np.linalg.norm(H, axis=-1, keepdims=True) + 1e-6)
    NoV = np.clip(np.sum(N*V, axis=-1), 0.0, 1.0)
    NoL = np.clip(np.sum(N*L, axis=-1), 0.0, 1.0)
    HoN = np.clip(np.sum(H*N, axis=-1), 0.0, 1.0)
    VH  = np.clip(np.sum(V*H, axis=-1), 0.0, 1.0)
    a = np.float32(max(0.02, min(1.0, alpha))); a2 = a*a
    denom = (HoN*HoN*(a2 - 1.0) + 1.0)
    D = (a2 / (np.pi * denom * denom + 1e-6)).astype(np.float32)
    k = (a * np.sqrt(2.0/np.pi)).astype(np.float32)
    G1 = lambda x: x / (x*(1.0 - k) + k + 1e-6)
    G = (G1(NoV) * G1(NoL)).astype(np.float32)
    F = (F0 + (1.0 - F0) * (1.0 - VH)**5).astype(np.float32)
    denom_gl = (4.0*np.maximum(NoV*NoL, 1e-6)).astype(np.float32)
    return (weight * (D * G / denom_gl)[...,None] * F[...,None]).astype(np.float32)

def sg16_setup():
    dirs = [
        (1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1),
        (1,1,1),(-1,1,1),(1,-1,1),(1,1,-1),(-1,-1,1),(-1,1,-1),(1,-1,-1),(-1,-1,-1),
        (1,1,0),(-1,1,0)
    ]
    mu = np.array([np.array(d)/np.linalg.norm(d) for d in dirs], dtype=np.float32)
    lam = np.full((16,), 6.0, dtype=np.float32)
    amp = 0.03*np.ones((16,), dtype=np.float32); amp[2] = 0.12  # +Y sky
    return mu, lam, amp
_SG_mu, _SG_lam, _SG_amp = sg16_setup()

def sg_eval(N, mu, lam):
    dotv = np.clip(np.sum(N[...,None,:]*mu[None,None,...], axis=3), -1.0, 1.0)
    return np.exp(lam[None,None,:]*(dotv-1.0)).astype(np.float32)

def ibl_diffuse(N, base, rho_eff, scale=1.0):
    val = sg_eval(N, _SG_mu, _SG_lam); L = (val * _SG_amp[None,None,:]).sum(axis=2, keepdims=True)
    return (base * rho_eff * (scale*np.repeat(L,3,axis=-1))).astype(np.float32)

def ibl_specular(N,V,rough, F0_rgb, AO, pref_env=0.06):
    PrefEnv = np.full_like(N, pref_env, dtype=np.float32)
    NoV = np.clip(np.sum(N*V, axis=-1), 0.0, 1.0)
    a   = np.clip(rough, 1e-3, 1.0)
    A = lut_bilerp(_FGD_A, NoV, a); B = lut_bilerp(_FGD_B, NoV, a)
    spec_env = PrefEnv * (F0_rgb * A[...,None] + (1.0 - F0_rgb) * B[...,None])
    k_so = lut_bilerp(_SO_LUT, a, AO)  # (rough, AO)
    SO = sat( 1.0 - k_so * (1.0 - AO) * (1.0 - NoV)**2 )
    return spec_env * SO[...,None]

# ---------- shadows ----------
PCSS = {"c00":0.20,"c10":-0.12,"c01":0.35,"c20":0.08,"c11":-0.18,"c02":0.10,"rmin":0.5,"rmax":2.5}

def pcss_radius(NoL, zr):
    c=PCSS; n=NoL.astype(np.float32); z=zr.astype(np.float32)
    r=(c["c00"]+c["c10"]*n+c["c01"]*z+c["c20"]*n*n+c["c11"]*n*z+c["c02"]*z*z).astype(np.float32)
    return np.clip(r, c["rmin"], c["rmax"])

def pcf9(depth, uvx, uvy, r, zref):
    H,W = depth.shape
    O = np.array([[-0.326,-0.406],[-0.840,-0.074],[-0.696, 0.457],[-0.203, 0.621],
                  [ 0.962,-0.195],[ 0.473,-0.480],[ 0.519, 0.767],[ 0.185,-0.893],[ 0.507, 0.064]], dtype=np.float32)
    vis = np.zeros_like(depth, dtype=np.float32)
    for k in range(9):
        dx = uvx + r*O[k,0]; dy = uvy + r*O[k,1]
        ix = np.clip(np.round(dx).astype(np.int32), 0, W-1)
        iy = np.clip(np.round(dy).astype(np.int32), 0, H-1)
        vis += (depth[iy, ix] > (zref - 1e-4)).astype(np.float32)
    return (vis / 9.0).astype(np.float32)

def shadow_robust(depth, uvx, uvy, r, zref, L_vec, tau_dir=0.018, gthr=0.008, kthr=0.003, a_g=0.9, tcap=0.85):
    H,W = depth.shape
    L2 = np.array(L_vec[:2], np.float32); L2 = L2/(np.linalg.norm(L2)+1e-6)
    s = 3.0
    dx = (uvx + s*L2[0]).round().astype(np.int32)
    dy = (uvy + s*L2[1]).round().astype(np.int32)
    dx = np.clip(dx, 0, W-1); dy = np.clip(dy, 0, H-1)
    dfx, dfy = sobel_pair(depth)
    gdir = (dfx*L2[0] + dfy*L2[1])
    curv = np.abs(laplacian5(depth))
    occ = ((depth[dy, dx] < (depth - tau_dir)) & ((gdir > gthr) | (curv > kthr))).astype(np.float32)
    T_raw = pcf9(depth, uvx, uvy, r, zref)
    T_min = 1.0 - np.clip(a_g * sobel_mag(depth), 0.0, tcap)
    return ((1.0 - occ) + occ * np.maximum(T_raw, T_min)).astype(np.float32)

# ---------- SSS / Fusion / Exposure / TAA ----------
def sss_k3(rgb, albedo, F_dr=0.5):
    W5 = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]], dtype=np.float32)
    W5 = W5 / np.sum(W5) * (1.0 - F_dr)
    H,W,_ = rgb.shape
    pad = np.pad(rgb * albedo[...,None], ((2,2),(2,2),(0,0)), mode='edge')
    out = np.zeros_like(rgb, np.float32)
    for j in range(5):
        for i in range(5):
            out += W5[j,i] * pad[j:j+H, i:i+W, :]
    return out

FUSION_A = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625], dtype=np.float32)
FUSION_B = np.array([0.05,   0.24, 0.42,  0.24, 0.05  ], dtype=np.float32)

def sep5(img, w):
    pad = ((0,0),(2,2),(0,0)); p = np.pad(img, pad, mode='edge')
    h = (w[0]*p[:,0:-4] + w[1]*p[:,1:-3] + w[2]*p[:,2:-2] + w[3]*p[:,3:-1] + w[4]*p[:,4:])
    pad2 = ((2,2),(0,0),(0,0)); p2 = np.pad(h, pad2, mode='edge')
    return (w[0]*p2[0:-4,:,:] + w[1]*p2[1:-3,:,:] + w[2]*p2[2:-2,:,:] + w[3]*p2[3:-1,:,:] + w[4]*p2[4:,:,:])

def fusion(rgb, depth, preset='A'):
    w = FUSION_A if preset=='A' else FUSION_B
    gz0,gz1 = sobel_pair(depth)
    gate = sat(0.5*np.abs(gz0) + 0.5*np.abs(gz1) + 0.25*np.abs(laplacian5(depth)))
    L1 = (1.0-gate)[...,None]*rgb + gate[...,None]*sep5(rgb, w)
    L2 = (1.0-gate)[...,None]*L1 + gate[...,None]*sep5(L1, w)
    return L2.astype(np.float32)

class TAAState:
    __slots__ = ['m','v']
    def __init__(self, H,W):
        self.m = np.zeros((H,W,3), dtype=np.float32)
        self.v = np.full((H,W,3), 1e-4, dtype=np.float32)

def rgb2ycgco(x):
    Y  = 0.25*x[...,0] + 0.5*x[...,1] + 0.25*x[...,2]
    Co = 0.5*x[...,0] - 0.5*x[...,2]
    Cg = -0.25*x[...,0] + 0.5*x[...,1] - 0.25*x[...,2]
    return Y, Cg, Co

def ycocg2rgb(Y, Cg, Co):
    R = Y + Co - Cg
    G = Y + Cg
    B = Y - Co - Cg
    return np.stack([R,G,B], axis=-1)

def taa_zerotap_ema_y(YC, Hh, st:TAAState, alpha=0.12, a=1/16, b=1/16, vmin=1e-6, vmax=0.25, k=2.5, c_hub=0.03):
    C = YC
    d = np.clip(C - st.m, -c_hub, c_hub)
    st.m += a * d
    st.v = (1.0 - b)*st.v + b * (d*d)
    st.v = np.clip(st.v, vmin, vmax)
    # clamp only Y, keep Co/Cg as from history
    Yh, Cgh, Coh = rgb2ycgco(Hh); Ym, Cgm, Com = rgb2ycgco(st.m)
    Yv = 0.25*st.v[...,0] + 0.5*st.v[...,1] + 0.25*st.v[...,2]
    Yc = np.minimum(np.maximum(Yh, Ym - k*np.sqrt(Yv + 1e-6)), Ym + k*np.sqrt(Yv + 1e-6))
    Hc = ycocg2rgb(Yc, Cgh, Coh)
    out = alpha*C + (1.0 - alpha)*Hc
    return sat(out)

def expose_aces(rgb, muL, target=0.46, smin=0.7, smax=2.6):
    s = np.clip(target / max(muL, 1e-6), smin, smax).astype(np.float32)
    a,b,c,d,e = 2.51,0.03,2.43,0.59,0.14
    img = (s*rgb).astype(np.float32)
    num = img*(a*img + b); den = img*(c*img + d) + e
    return sat(num/(den+1e-6))

# ---------- scene (5 spheres) ----------
def gbuffer_spheres(H=768, W=768):
    bg = np.array([0.32, 0.40, 0.43], dtype=np.float32)
    spheres = [
        (0.26*W, 0.34*H, 0.23*min(H,W), np.array([0.86,0.89,0.92], dtype=np.float32), 0.04, 0.36),
        (0.69*W, 0.36*H, 0.20*min(H,W), np.array([0.83,0.92,0.93], dtype=np.float32), 0.02, 0.32),
        (0.50*W, 0.63*H, 0.18*min(H,W), np.array([0.71,0.79,0.92], dtype=np.float32), 0.03, 0.44),
        (0.80*W, 0.70*H, 0.14*min(H,W), np.array([0.92,0.94,0.97], dtype=np.float32), 0.02, 0.30),
        (0.23*W, 0.71*H, 0.11*min(H,W), np.array([0.74,0.78,0.70], dtype=np.float32), 0.01, 0.50),
    ]
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing='ij')
    depth = np.full((H,W), 1.0, np.float32)
    N = np.dstack([np.zeros((H,W),np.float32), np.zeros((H,W),np.float32), np.ones((H,W),np.float32)])
    base = np.tile(bg[None,None,:], (H,W,1)).astype(np.float32)
    metal = np.zeros((H,W), dtype=np.float32)
    rough = np.ones((H,W), dtype=np.float32)
    for (cx, cy, r, col, m, a) in spheres:
        dx = (xx - cx); dy = (yy - cy)
        dist2 = dx*dx + dy*dy
        mask = dist2 <= (r*r)
        z_val = np.sqrt(np.maximum(r*r - dist2, 0.0)) / (r + 1e-6)
        cand_depth = 1.0 - z_val
        take = mask & (cand_depth < depth)
        depth = np.where(take, cand_depth, depth)
        nx = np.where(take, dx / (r + 1e-6), N[...,0])
        ny = np.where(take, dy / (r + 1e-6), N[...,1])
        nz = np.where(take, z_val,                N[...,2])
        N = np.stack([nx, ny, nz], axis=-1)
        base = np.where(take[...,None], col[None,None,:], base)
        metal = np.where(take, m, metal)
        rough = np.where(take, a, rough)
    norm = np.linalg.norm(N, axis=-1, keepdims=True)
    N = np.where(norm>1e-6, N/np.maximum(norm,1e-6), N)
    V = np.zeros_like(N); V[...,2] = 1.0
    return {"N":N,"V":V,"depth":depth,"base":base,"metal":metal,"rough":rough,"bg":bg}

# ---------- pipeline ----------
def render(H=768, W=768, use_shadows=True):
    G = gbuffer_spheres(H,W)
    N,V,depth,base,metal,rough,bg = G["N"],G["V"],G["depth"],G["base"],G["metal"],G["rough"],G["bg"]
    S = stats256(base, depth, None)
    K = allocate(S)

    # two directional lights
    lights = [(np.array([0.38,0.54,0.75],np.float32),1.2),(np.array([-0.22,0.42,0.88],np.float32),0.5)]
    L_dir = np.zeros_like(base, np.float32)
    F0_rgb = 0.04*(1.0 - metal)[...,None] + base*metal[...,None]

    for d,inten in lights:
        d = d/(np.linalg.norm(d)+1e-6)
        L = np.broadcast_to(d, N.shape)
        # Toksvig-like spec AA + clearcoat
        # compute rough_eff per-pixel (same for both lights, but fine)
        vN = normal_variance4(N)
        k_tok = 0.35
        rough_eff = np.sqrt(np.clip(rough*rough + k_tok*vN, rough*rough, rough*rough + 0.35*vN)).astype(np.float32)
        spec = brdf_ggx(N,V,L,rough_eff,F0_rgb) + brdf_clearcoat(N,V,L, alpha=0.08, F0=0.04, weight=0.06)
        NoL  = np.maximum(np.sum(N*L,axis=-1,keepdims=True), 0.0)
        # multi-scatter diffuse compensation
        F0m = np.mean(F0_rgb, axis=-1, keepdims=True)
        Ems = (1.0 - 0.4399*rough + 0.0927*rough*rough).astype(np.float32)
        rho_eff = (1.0 - F0m)*(1.0 - F0m) + Ems[...,None]*F0m
        diff  = base * (rho_eff * (NoL/PI))
        L_dir += inten * (K["spec_gain"] * spec*NoL + diff)

    if use_shadows and K["shadow_enable"]:
        yy, xx = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        NoL_ref = np.clip(np.sum(N*lights[0][0],axis=-1), 0.0, 1.0).astype(np.float32)
        zr  = sat((depth - depth.min()) / (depth.max()-depth.min()+CT_EPS))
        r   = pcss_radius(NoL_ref, zr) * K["pcss_scale"]
        T_s = shadow_robust(depth, xx.astype(np.float32), yy.astype(np.float32), r, depth+0.001, lights[0][0])
        T_l = shadow_robust(depth, xx.astype(np.float32), yy.astype(np.float32), 1.6*r, depth+0.001, lights[0][0])
        eta = 0.45
        T = (1.0-eta)*T_s + eta*T_l
        L_dir *= T[...,None]

    # IBL
    AO = np.ones((H,W), dtype=np.float32)
    F0m = np.mean(F0_rgb, axis=-1, keepdims=True)
    Ems = (1.0 - 0.4399*rough + 0.0927*rough*rough).astype(np.float32)
    rho_eff = (1.0 - F0m)*(1.0 - F0m) + Ems[...,None]*F0m
    L_diff = ibl_diffuse(N, base, rho_eff, scale=K["ibl_scale"])
    L_spec = ibl_specular(N,V,rough,F0_rgb,AO, pref_env=K["pref_env"])

    # SSS
    L_sss  = sss_k3(base, albedo=np.full((H,W), K["sss_albedo"], np.float32), F_dr=K["sss_Fdr"])

    # compose
    L_surf = L_dir + L_diff + L_spec + L_sss
    maxc = np.max(L_surf, axis=2, keepdims=True)
    L = L_surf / np.maximum(np.maximum(maxc, 1.0), CT_EPS)
    cov = (depth < 0.999).astype(np.float32)[...,None]
    L = (1.0-cov)*bg[None,None,:] + cov*L

    # fusion
    L = fusion(L, depth, preset=K["fusion_id"])

    # log-mean exposure estimate from 256 samples of current L
    Ys = (0.2126*L[...,0] + 0.7152*L[...,1] + 0.0722*L[...,2]).astype(np.float32)
    xs, ys = lcg_indices(H, W, 256, seed=1337)
    muL_log = float(np.exp(np.mean(np.log(1e-6 + Ys[ys, xs]))))

    # exposure + ACES
    L = expose_aces(L, muL_log, target=float(K["exp_target"]), smin=0.7, smax=2.6)

    # blue-noise-ish dither (hash), amplitude ~1/255
    yy_i, xx_i = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    def uhash(x):
        x = (x ^ (x>>17)) * 0xED5AD4BB & 0xFFFFFFFF
        x = (x ^ (x>>11)) * 0xAC4C1B51 & 0xFFFFFFFF
        x = (x ^ (x>>15)) * 0x31848BAB & 0xFFFFFFFF
        x = x ^ (x>>14); return x
    h = (uhash((xx_i + 131*yy_i + 71) & 0xFFFFFFFF) & 0xFFFF) / 65535.0
    L = np.clip(L + (h[...,None]-0.5)/255.0, 0.0, 1.0)

    # TAA 0-tap (EMA, Y clamp)
    taa = TAAState(H,W); taa.m[...] = L; taa.v[...] = 1e-4
    out = taa_zerotap_ema_y(L, L, taa, alpha=float(K["taa_alpha"]))
    return out, K

# ---------- math helpers ----------
def normal_variance4(N):
    nx, ny, nz = N[...,0], N[...,1], N[...,2]
    def v4(ch):
        H,W = ch.shape
        p = np.pad(ch, ((1,1),(1,1)), mode='edge')
        c = p[1:-1,1:-1]; l = p[1:-1,0:-2]; r = p[1:-1,2:]; u = p[0:-2,1:-1]; d = p[2:,1:-1]
        mu = (c+l+r+u+d)/5.0
        return ((c-mu)**2 + (l-mu)**2 + (r-mu)**2 + (u-mu)**2 + (d-mu)**2)/5.0
    return (v4(nx)+v4(ny)+v4(nz)).astype(np.float32)

# ---------- demo ----------
def demo(H=768,W=768):
    return render(H,W)

if __name__ == "__main__":
    import PIL.Image as P
    img, knobs = demo(768,768)
    P.fromarray((np.clip(img,0,1)*255+0.5).astype(np.uint8), "RGB").save("fdr_o1_demo_hq.png")
