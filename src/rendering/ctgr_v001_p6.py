
#!/usr/bin/env python3
# ============================================================================
# CT-GR v0.0.1-p3  (Constant-Time Graphics Renderer + Pareto Allocator)
# - Adds CTPA: O(1) per-frame optimizer choosing constants for AA/Shadow/AO/Fusion/MS
# - Two-pass fixed pipeline: prepass (stats) -> allocate -> final render
# - Deterministic, fixed tap counts, O(1)/px.
# ============================================================================

from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np, imageio.v2 as iio
import argparse, os

EPS = 1e-8

# ------------------------------- Config -------------------------------------
@dataclass(frozen=True)
class Config:
    size: int = 640
    sun_dir: tuple = (0.6, 0.4, 1.0)
    sun_intensity: float = 2.5
    # SSAO
    ao_floor: float = 0.90
    ao_strength: float = 1.20     # theta_3 (Case B)
    # Fusion
    fusion_lam1: float = 0.55     # theta_4 (Case A)
    fusion_lam2: float = 0.22
    fusion_thr: float = 0.10
    fusion_boost: float = 0.45
    # Tone/Exposure
    aces_gain: float = 1.05
    target_gray: float = 0.18
    exp_min: float = 0.80
    exp_max: float = 1.25
    dither: bool = True
    dither_amp: float = 0.75/255.0
    # Bent-normal weight
    bent_w: float = 0.15
    # Coverage AA width
    cov_kF: float = 1.25          # theta_1 (Case A)
    # Shadow PCF5
    pcf_r0: float = 0.60
    pcf_r1: float = 0.80
    pcf_t0: float = 0.00
    pcf_wmax: float = 0.50
    pcf_scale: float = 1.00       # theta_2 (Case A), scales final r_eff
    # OIT (weighted blended)
    oit_beta: float = 2.0
    # MS compensation blend (0..1)  theta_5 (Case C around ref)
    ms_blend: float = 1.00
    # Allocator toggles
    use_allocator: bool = True
    version: str = "0.0.1-p6"

# ----------------------------- Utilities ------------------------------------
def sat01(x): 
    return np.minimum(1.0, np.maximum(0.0, x)).astype(np.float32)

def roll_cross_rgb(img, c, n):
    s = c + 4.0*n
    L = np.roll(img,  1, axis=1)
    R = np.roll(img, -1, axis=1)
    U = np.roll(img,  1, axis=0)
    D = np.roll(img, -1, axis=0)
    return (c*img + n*(L+R+U+D)) / s

def roll_cross_scalar(img, c, n):
    s = c + 4.0*n
    L = np.roll(img,  1, axis=1)
    R = np.roll(img, -1, axis=1)
    U = np.roll(img,  1, axis=0)
    D = np.roll(img, -1, axis=0)
    return (c*img + n*(L+R+U+D)) / s

def grad_mag_scalar(img):
    gx = np.roll(img, -1, axis=1) - np.roll(img, 1, axis=1)
    gy = np.roll(img, -1, axis=0) - np.roll(img, 1, axis=0)
    return np.sqrt(0.25*(gx*gx + gy*gy)).astype(np.float32)

def luma(img): 
    return (0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]).astype(np.float32)

def blue_hash(i, j):
    x = (i.astype(np.uint32)*1664525 + j.astype(np.uint32)*1013904223) ^ np.uint32(0x68bc21eb)
    x ^= (x>>13); x *= np.uint32(1274126177); x ^= (x>>16)
    return (x & np.uint32(1023)).astype(np.float32) / 1023.0

def tone_map_aces(img, gain=1.05):
    x = np.maximum(0.0, img) * gain
    a=2.51; b=0.03; c=2.43; d=0.59; e=0.14
    y = (x*(a*x + b)) / (x*(c*x + d) + e)
    sr = np.where(y <= 0.0031308, 12.92*y, 1.055*np.power(np.maximum(y,0.0), 1/2.4) - 0.055)
    return sat01(sr)

# ----------------------------- Environment ----------------------------------
def env_color(dir3):
    z = dir3[...,2:3]
    s_dir = np.array([0.2, 0.2, 0.95], np.float32)
    cos_s = np.maximum(0.0, np.sum(dir3 * s_dir, axis=-1, keepdims=True))
    sun = (cos_s**64) * np.array([1.0, 0.9, 0.7], np.float32)[None,None,:] * 2.6
    sky = np.concatenate([0.45 + 0.25*z, 0.55 + 0.30*z, 0.65 + 0.35*z], -1)
    return (sky + sun).astype(np.float32)

def sh2_project(env_fn, S=64):
    u = np.linspace(0, 1, S, endpoint=False); v = np.linspace(0, 1, 2*S, endpoint=False)
    U,V = np.meshgrid(u,v, indexing='ij')
    theta = np.arccos(1 - 2*U); phi = 2*np.pi*V
    dirs = np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], -1).astype(np.float32)
    L = env_fn(dirs); dω = (4*np.pi)/(S*(2*S))
    x,y,z = dirs[...,0], dirs[...,1], dirs[...,2]
    Y = [ 0.282095 + 0*z, 0.488603*y, 0.488603*z, 0.488603*x,
          1.092548*x*y, 1.092548*y*z, 0.315392*(3*z*z - 1),
          1.092548*x*z, 0.546274*(x*x - y*y) ]
    c = [np.sum((Yk[...,None]*L).astype(np.float64), axis=(0,1))*dω for Yk in Y]
    return np.stack(c,0).astype(np.float32)

SH2_COEFF = sh2_project(env_color, 64)

def sh2_eval_irradiance(n):
    x,y,z = n[...,0], n[...,1], n[...,2]
    Y = np.stack([
        0.282095 + 0*x,
        0.488603*y,
        0.488603*z,
        0.488603*x,
        1.092548*x*y,
        1.092548*y*z,
        0.315392*(3*z*z - 1),
        1.092548*x*z,
        0.546274*(x*x - y*y)
    ], -1).astype(np.float32)
    return sat01(np.tensordot(Y.astype(np.float64), SH2_COEFF.astype(np.float64), axes=([-1],[0])).astype(np.float32))

def prefiltered_env(r_dir, rough):
    # simple roughness mix to average env; kept constant-time
    avg = np.array([0.55,0.60,0.65], np.float32)
    base = env_color(r_dir)
    t = np.clip(rough*rough, 0.0, 1.0)[...,None]
    return (1.0 - t)*base + t*avg

# ----------------------------- BRDF -----------------------------------------
def F_schlick(F0, u): return F0 + (1.0 - F0)*(1.0 - u)[...,None]**5
def D_ggx(NoH, alpha):
    a2 = alpha*alpha; d = (NoH*NoH*(a2 - 1.0) + 1.0)
    return (a2 / (np.pi * d*d + EPS))[...,None]
def G1_smith(t, alpha): 
    return (2.0*t / (t + np.sqrt(alpha*alpha + (1.0 - alpha*alpha)*t*t + EPS) + EPS))[...,None]
def G_smith(NoV, NoL, alpha): return G1_smith(NoV, alpha) * G1_smith(NoL, alpha)

def ms_energy_comp(F0, alpha, blend):
    # blend ∈ [0,1], 0: off, 1: full compensation
    Favg = np.mean(F0, axis=-1, keepdims=True)  # (H,W,1)
    Ems_full = 1.0 - 0.28*alpha[...,None] + 0.35*(alpha[...,None]**2)
    Ems_full = sat01(Ems_full)
    Ems = blend*Ems_full + (1.0-blend)*0.0
    return Favg, Ems

def brdf_terms(n, v, l, rho_d, F0, alpha, metallic, ms_blend):
    h = (v + l); h = h/np.clip(np.linalg.norm(h, axis=-1, keepdims=True), 1e-6, None)
    NoV = np.clip(np.sum(n*v, -1), 0.0, 1.0); NoL = np.clip(np.sum(n*l, -1), 0.0, 1.0)
    NoH = np.clip(np.sum(n*h, -1), 0.0, 1.0); VoH = np.clip(np.sum(v*h, -1), 0.0, 1.0)
    D = D_ggx(NoH, alpha); G = G_smith(NoV, NoL, alpha); F = F_schlick(F0, VoH)
    spec = (D * G * F) / np.maximum(4.0*NoV[...,None]*NoL[...,None] + EPS, EPS)

    # Multiple scattering compensation in diffuse albedo
    Favg, Ems = ms_energy_comp(F0, alpha, ms_blend)
    rho_eff = rho_d * (1.0 - Favg) + Ems * Favg

    Fv = F_schlick(F0, NoV)                     # (H,W,3)
    kd = (1.0 - Fv) * (1.0 - metallic[...,None])
    diff = (kd*rho_eff/np.pi)[...,None,:] * NoL[...,None,None]
    return diff, spec

# ----------------------------- Scene ----------------------------------------
def build_spheres_scene(H=640, W=640):
    centers = np.array([[0.30,0.35,0.55],[0.68,0.40,0.45],[0.50,0.68,0.60],[0.24,0.74,0.40],[0.78,0.74,0.70]], np.float32)
    radii   = np.array([0.18,0.16,0.14,0.10,0.12], np.float32)
    rho_d_s = np.array([[0.85,0.36,0.36],[0.36,0.85,0.42],[0.38,0.50,0.88],[0.85,0.80,0.45],[0.80,0.80,0.85]], np.float32)
    rough_s = np.array([0.20,0.35,0.55,0.75,0.40], np.float32)
    metal_s = np.array([0.90,0.20,0.05,0.00,0.60], np.float32)

    y,x = np.mgrid[0:H,0:W].astype(np.float32)
    u = x/W; v = y/H

    S = centers.shape[0]
    Z_s = np.full((S,H,W), np.inf, np.float32)
    Nx = np.zeros((S,H,W), np.float32); Ny = np.zeros_like(Nx); Nz = np.zeros_like(Nx)

    for s in range(S):
        cx,cy,cz = centers[s]; r = radii[s]
        dx = u - cx; dy = v - cy; d2 = dx*dx + dy*dy
        inside = d2 <= (r*r)
        zsurf = cz - np.sqrt(np.maximum(1e-8, r*r - d2))
        Z_s[s][inside] = zsurf[inside]
        nx = dx; ny = dy; nz = zsurf - cz
        inv = 1.0/np.sqrt(np.maximum(1e-8, nx*nx + ny*ny + nz*nz))
        Nx[s] = nx*inv; Ny[s] = ny*inv; Nz[s] = nz*inv

    Z_min = np.min(Z_s, axis=0)
    hit = np.isfinite(Z_min)
    sid = np.argmin(Z_s, axis=0)
    ii = np.arange(H)[:,None]; jj = np.arange(W)[None,:]

    N = np.zeros((H,W,3), np.float32); P = np.zeros((H,W,3), np.float32)
    rho_d = np.zeros((H,W,3), np.float32); rough = np.zeros((H,W), np.float32); metal = np.zeros((H,W), np.float32)

    N[:,:,0] = Nx[sid, ii, jj]; N[:,:,1] = Ny[sid, ii, jj]; N[:,:,2] = Nz[sid, ii, jj]
    Z = Z_min.copy()
    P[:,:,0] = u; P[:,:,1] = v; P[:,:,2] = Z
    for c in range(3): rho_d[:,:,c] = rho_d_s[sid, c]
    rough = rough_s[sid]; metal = metal_s[sid]

    bg = ~hit
    N[bg] = np.array([0,0,1], np.float32)
    Z[bg] = 1.2; P[bg,2] = Z[bg]
    rho_d[bg] = np.array([0.22,0.24,0.26], np.float32); rough[bg] = 0.65; metal[bg] = 0.0

    V = np.dstack([np.zeros_like(Z), np.zeros_like(Z), np.ones_like(Z)])
    return SimpleNamespace(P=P,N=N,Z=Z,V=V, rho_d=rho_d, rough=rough, metal=metal,
                           centers=centers, radii=radii, sid=sid, hit=hit, u=u, v=v)

# ----------------------------- Lighting -------------------------------------
def direct_dir_sun(gb, Ldir, intensity, cfg: Config):
    Ld = Ldir/np.linalg.norm(Ldir)
    L = np.broadcast_to(Ld, gb.P.shape)
    F0 = 0.04*(1.0 - gb.metal)[...,None] + gb.rho_d*gb.metal[...,None]

    # Toksvig roughness via roll gradients with clamp
    Nx,Ny,Nz = gb.N[:,:,0], gb.N[:,:,1], gb.N[:,:,2]
    gx = np.sqrt(0.25*((np.roll(Nx,-1,1)-np.roll(Nx,1,1))**2 + (np.roll(Nx,-1,0)-np.roll(Nx,1,0))**2))
    gy = np.sqrt(0.25*((np.roll(Ny,-1,1)-np.roll(Ny,1,1))**2 + (np.roll(Ny,-1,0)-np.roll(Ny,1,0))**2))
    gz = np.sqrt(0.25*((np.roll(Nz,-1,1)-np.roll(Nz,1,1))**2 + (np.roll(Nz,-1,0)-np.roll(Nz,1,0))**2))
    varN = 0.5*(gx+gy+gz).astype(np.float32)
    k_tok = np.where(varN < 0.05, 0.20 + (0.50-0.20)*(varN/0.05),
                     np.where(varN < 0.12, 0.50 + (0.80-0.50)*((varN-0.05)/0.07), 0.80)).astype(np.float32)
    alpha_raw = np.sqrt(np.clip(gb.rough**2 + k_tok*varN, 1e-4, 1.0))
    alpha = np.minimum(np.maximum(alpha_raw, gb.rough), np.sqrt(gb.rough*gb.rough + 0.35*varN))

    diff, spec = brdf_terms(gb.N, gb.V, L, gb.rho_d, F0, alpha, gb.metal, cfg.ms_blend)
    NoL = np.clip(np.sum(gb.N*L, -1, keepdims=True), 0, 1)
    Lo = (diff[...,0,:] + spec[...,0,:]) * intensity * NoL

    # Analytic sphere occlusion (hard)
    C = gb.centers; R = gb.radii; S = C.shape[0]
    Pc = gb.P[...,None,:] - C[None,None,:,:]
    b = np.sum(L[... ,None,:]*Pc, axis=-1)
    c = np.sum(Pc*Pc, axis=-1) - R[None,None,:]**2
    disc = b*b - c
    t = (-b - np.sqrt(np.maximum(disc, 0)))
    self_mask = (gb.sid[...,None] == np.arange(S)[None,None,:])
    occ = (disc > 0) & (t > 1e-4) & (~self_mask)
    T_hard = 1.0 - occ.any(axis=-1, keepdims=True).astype(np.float32)

    # Distance-dependent PCF5, scaled by cfg.pcf_scale
    t_pos = np.where(occ, t, np.inf)
    t_min = np.min(t_pos, axis=-1)
    t_min[~np.isfinite(t_min)] = 0.0
    eta = np.clip((t_min - cfg.pcf_t0) / max(cfg.pcf_wmax, 1e-6), 0.0, 1.0).astype(np.float32)

    baseA = np.array([[-0.5,-0.1],[0.6,-0.4],[-0.2,0.7],[0.8,0.5],[-0.7,0.3],
                      [0.1,0.9],[-0.9,-0.2],[0.3,-0.8],[-0.6,0.6]], np.float32)
    baseB = np.array([[0.4,0.2],[-0.6,-0.5],[0.7,-0.3],[-0.3,0.8],[0.2,-0.9],
                      [-0.8,0.4],[0.1,0.7],[-0.5,-0.7],[0.6,0.6]], np.float32)
    r_eff = (cfg.pcf_r0 + cfg.pcf_r1 * eta) * cfg.pcf_scale
    H,W = T_hard.shape[:2]
    base = baseA if ((H & 1)==0) else baseB
    def shift_clamp(arr, dx, dy):
        pad_x = abs(dx); pad_y = abs(dy)
        ap = np.pad(arr, ((pad_y,pad_y),(pad_x,pad_x),(0,0)), mode='edge')
        y0 = pad_y - dy; x0 = pad_x - dx
        return ap[y0:y0+H, x0:x0+W]

    # two fixed radii (small, large) then interpolate by eta
    r_mean = float(np.mean(r_eff))
    r_s = max(1, int(round(0.7 * r_mean)))
    r_l = max(1, int(round(1.6 * r_mean)))
    def blurN(arr, rpix):
        acc = np.zeros_like(arr)
        N = base.shape[0]
        for k in range(N):
            dx = int(round(base[k,0] * rpix))
            dy = int(round(base[k,1] * rpix))
            acc += shift_clamp(arr, dx, dy)
        return acc / float(N)
    T_s = blurN(T_hard, r_s)
    T_l = blurN(T_hard, r_l)
    T_pcf = (1.0 - eta[...,None])*T_s + eta[...,None]*T_l
    Ts = T_pcf
    Ts = sat01(Ts)
    return Lo * Ts, Ts[...,0]


# ----------------------------- FGD LUT (Specular IBL Split-Sum) -------------
def _importance_sample_ggx(xi, alpha):
    # xi: (...,2) uniform; return H (3)
    phi = 2*np.pi*xi[...,0]
    cos2 = (1 - xi[...,1]) / (1 + (alpha*alpha - 1)*xi[...,1] + 1e-8)
    cosTheta = np.sqrt(np.clip(cos2, 0.0, 1.0))
    sinTheta = np.sqrt(np.maximum(0.0, 1.0 - cosTheta*cosTheta))
    return np.stack([np.cos(phi)*sinTheta, np.sin(phi)*sinTheta, cosTheta], -1).astype(np.float32)

def _normalize(v): 
    return v/np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8, None)

def build_fgd_lut(Nov_res=32, rough_res=32, samples=128, seed=1337):
    # Returns (A,B) each [Nov_res, rough_res]
    rng = np.random.RandomState(seed)
    A = np.zeros((Nov_res, rough_res), np.float32)
    B = np.zeros_like(A)
    N = np.array([0,0,1], np.float32)
    for i in range(Nov_res):
        NoV = (i + 0.5)/Nov_res
        V = np.array([np.sqrt(max(0.0, 1-NoV*NoV)), 0.0, NoV], np.float32)
        for j in range(rough_res):
            rough = (j + 0.5)/rough_res
            alpha = max(1e-4, rough*rough)
            accA = 0.0; accB = 0.0
            for s in range(samples):
                xi = rng.rand(2).astype(np.float32)
                H = _importance_sample_ggx(xi, alpha)
                L = np.array([ -V[0] + 2*H[0]*np.dot(V,H),
                               -V[1] + 2*H[1]*np.dot(V,H),
                               -V[2] + 2*H[2]*np.dot(V,H)], np.float32)
                NoL = max(0.0, L[2]); NoH = max(0.0, H[2]); VoH = max(0.0, np.dot(V,H))
                if NoL > 0 and NoH > 0:
                    # Smith GGX (Schlick-GGX form for visibility term)
                    k = (alpha*alpha)/2.0
                    def G1_SchlickGGX(nv):
                        return nv / (nv*(1.0 - k) + k + 1e-8)
                    G = G1_SchlickGGX(NoV) * G1_SchlickGGX(NoL)
                    G_Vis = (G * VoH) / (NoH * NoV + 1e-8)
                    Fc = (1.0 - VoH)**5
                    accA += (1.0 - Fc) * G_Vis
                    accB += Fc * G_Vis
            A[i,j] = accA / samples
            B[i,j] = accB / samples
    return A, B

_FGD_A, _FGD_B = build_fgd_lut(32, 32, samples=96)

def sample_fgd(NoV, rough):
    # bilinear sample of A,B from LUT
    NoV = np.clip(NoV, 0.0, 1.0); rough = np.clip(rough, 0.0, 1.0)
    i = NoV * 31.0; j = rough * 31.0
    i0 = np.floor(i).astype(np.int32); j0 = np.floor(j).astype(np.int32)
    i1 = np.clip(i0+1, 0, 31); j1 = np.clip(j0+1, 0, 31)
    ti = (i - i0); tj = (j - j0)
    A00 = _FGD_A[i0, j0]; A01 = _FGD_A[i0, j1]; A10 = _FGD_A[i1, j0]; A11 = _FGD_A[i1, j1]
    B00 = _FGD_B[i0, j0]; B01 = _FGD_B[i0, j1]; B10 = _FGD_B[i1, j0]; B11 = _FGD_B[i1, j1]
    A = (1-ti)*( (1-tj)*A00 + tj*A01 ) + ti*( (1-tj)*A10 + tj*A11 )
    B = (1-ti)*( (1-tj)*B00 + tj*B01 ) + ti*( (1-tj)*B10 + tj*B11 )
    return A.astype(np.float32), B.astype(np.float32)

# ----------------------------- SO 8x8 table ---------------------------------
_SO_TAB = np.zeros((8,8), np.float32)
for ri in range(8):
    for ai in range(8):
        r = (ri+0.5)/8.0   # rough
        a = (ai+0.5)/8.0   # AO
        _SO_TAB[ri,ai] = np.clip(0.7 + 0.6*r - 0.3*a, 0.6, 1.2)

def sample_so_k(rough, ao):
    rough = np.clip(rough, 0.0, 1.0); ao = np.clip(ao, 0.0, 1.0)
    i = rough*7.0; j = ao*7.0
    i0 = np.floor(i).astype(np.int32); j0 = np.floor(j).astype(np.int32)
    i1 = np.clip(i0+1, 0, 7); j1 = np.clip(j0+1, 0, 7)
    ti = (i - i0); tj = (j - j0)
    v = (1-ti)*((1-tj)*_SO_TAB[i0,j0] + tj*_SO_TAB[i0,j1]) + ti*((1-tj)*_SO_TAB[i1,j0] + tj*_SO_TAB[i1,j1])
    return v.astype(np.float32)

# ----------------------------- IBL ------------------------------------------

def specular_occlusion(A, rough, NoV, k_so=1.0):
    # A in [0,1], NoV in [0,1], rough in [0,1]
    # SO = 1 - k*(1-A)*(1-NoV)^2*rough
    return np.clip(1.0 - k_so*(1.0 - A)*(1.0 - NoV)**2 * np.clip(rough,0.0,1.0), 0.0, 1.0).astype(np.float32)

def ibl_lighting(gb, cfg: Config):
    F0 = 0.04*(1.0 - gb.metal)[...,None] + gb.rho_d*gb.metal[...,None]
    E = sh2_eval_irradiance(gb.N); L_diff = (gb.rho_d/np.pi) * E
    R = 2*(np.sum(gb.N*gb.V, axis=-1, keepdims=True))*gb.N - gb.V
    R = R/np.clip(np.linalg.norm(R, axis=-1, keepdims=True), 1e-6, None)
    env_pref = prefiltered_env(R, gb.rough)
    NoV = np.clip(np.sum(gb.N*gb.V, -1), 0.0, 1.0)
    A,B = sample_fgd(NoV, gb.rough)
    L_spec = env_pref * (F0*A[...,None] + (1.0 - F0)*B[...,None])
    return L_diff, L_spec

# ----------------------------- SSAO -----------------------------------------
def shift_clamp_gray(arr, dx, dy):
    H,W = arr.shape
    pad_x = abs(dx); pad_y = abs(dy)
    ap = np.pad(arr, ((pad_y,pad_y),(pad_x,pad_x)), mode='edge')
    y0 = pad_y - dy; x0 = pad_x - dx
    return ap[y0:y0+H, x0:x0+W]

def ssao_8x4(gb, floor_val, strength):
    # link strength to floor: higher floor -> gentler strength
    k = np.clip(1.0 - 0.8*((floor_val - 0.85)/0.15), 0.5, 1.2)
    strength = strength * k
    Z = gb.Z
    # 12 directions (parity-rotated) × 5 radii = 60 samples (constant)
    dirs_A = np.array([[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1],
                       [2,1],[-1,2],[-2,-1],[1,-2]], np.float32)
    dirs_A = (dirs_A.T/np.maximum(1.0,np.linalg.norm(dirs_A,axis=1))).T
    dirs_B = np.array([[1,1],[1,0],[0,1],[-1,0],[-1,1],[-1,-1],[0,-1],[1,-1],
                       [2,-1],[1,2],[-2,1],[-1,-2]], np.float32)
    dirs_B = (dirs_B.T/np.maximum(1.0,np.linalg.norm(dirs_B,axis=1))).T

    r_steps = np.array([2,3,5,7,11], np.int32)
    H,W = Z.shape
    acc = np.zeros_like(Z, dtype=np.float32)
    for y in range(H):
        parity = (y & 1)
        dirs = dirs_A if parity==0 else dirs_B
        row = np.zeros((W,), np.float32)
        for d in dirs:
            for r in r_steps:
                dx = int(np.rint(d[0]*r)); dy = int(np.rint(d[1]*r))
                Zs = shift_clamp_gray(Z, dx, dy)[y]
                row += np.maximum(0.0, (Zs - Z[y]) - 0.0015)
        acc[y] = row
    acc /= float(8*len(r_steps))
    ao = sat01(1.0 - strength*acc)
    ao2 = roll_cross_scalar(ao, c=4.0, n=1.0)
    return np.maximum(ao2, floor_val).astype(np.float32)

# ----------------------------- Coverage AA ----------------------------------
def sdf_edge_coverage(gb, kF=1.25):
    sid = gb.sid; C = gb.centers; R = gb.radii
    cx = C[sid,0]; cy = C[sid,1]; r = R[sid]
    dx = gb.u - cx; dy = gb.v - cy
    s = np.sqrt(np.maximum(0.0, dx*dx + dy*dy)) - r
    F = (1.0/gb.Z.shape[1] + 1.0/gb.Z.shape[0]) * kF
    cov = sat01(0.5 - s / (F+EPS))
    return sat01(cov*(2.0 - cov))

def tri_coverage(UV, H, W, kF=1.25):
    y,x = np.mgrid[0:H,0:W].astype(np.float32)
    p = np.stack([x/W, y/H], -1)
    v0,v1,v2 = UV[0], UV[1], UV[2]
    def edge_func(a, b, x):
        ab = b - a
        return (x[...,0]-a[0])*(ab[1]) - (x[...,1]-a[1])*(ab[0]), np.linalg.norm(ab) + 1e-8
    e0, l0 = edge_func(v0, v1, p)
    e1, l1 = edge_func(v1, v2, p)
    e2, l2 = edge_func(v2, v0, p)
    d0 = e0 / l0; d1 = e1 / l1; d2 = e2 / l2
    F = (1.0/W + 1.0/H) * kF
    cov = sat01(0.5 + d0/F) * sat01(0.5 + d1/F) * sat01(0.5 + d2/F)
    return sat01(cov*(2.0 - cov))

# ----------------------------- Background -----------------------------------
def background_shading(H,W, cfg: Config):
    N = np.dstack([np.zeros((H,W),np.float32), np.zeros((H,W),np.float32), np.ones((H,W),np.float32)])
    V = N.copy()
    rho = np.array([0.22,0.24,0.26], np.float32)[None,None,:]*np.ones((H,W,1), np.float32)
    rough = 0.65*np.ones((H,W), np.float32); metal = np.zeros((H,W), np.float32)
    gb = SimpleNamespace(N=N,V=V, rho_d=rho, rough=rough, metal=metal, P=np.zeros((H,W,3),np.float32))
    Ld = np.array([0.6,0.4,1.0], np.float32); Ld = Ld/np.linalg.norm(Ld)
    L = np.broadcast_to(Ld, gb.P.shape)
    F0 = 0.04*(1.0 - gb.metal)[...,None] + gb.rho_d*gb.metal[...,None]
    alpha_eff = gb.rough**2
    diff, spec = brdf_terms(gb.N, gb.V, L, gb.rho_d, F0, alpha_eff, gb.metal, cfg.ms_blend)
    NoL = np.clip(np.sum(gb.N*L, -1, keepdims=True), 0, 1)
    Lo = (diff[...,0,:] + spec[...,0,:]) * 2.2 * NoL
    E = sh2_eval_irradiance(gb.N); L_diff = (gb.rho_d/np.pi) * E
    R = 2*(np.sum(gb.N*gb.V, axis=-1, keepdims=True))*gb.N - gb.V
    env_pref = prefiltered_env(R, gb.rough)
    NoV = np.clip(np.sum(gb.N*gb.V, -1), 0.0, 1.0)
    A,B = sample_fgd(NoV, gb.rough)
    L_spec = env_pref * (F0*A[...,None] + (1.0 - F0)*B[...,None])
    return L_spec + L_diff + Lo

# ----------------------------- Fusion / Exposure ----------------------------
def robin_harmonic(img, Z, lam=0.55, beta_hi=0.34, tz=0.12, tk=0.05, sz=16.0, sk=16.0):
    N4 = roll_cross_rgb(img, c=0.0, n=0.25)
    gZ  = grad_mag_scalar(Z) / (np.abs(Z)+EPS)
    lap = grad_mag_scalar(np.gradient(Z)[0]) + grad_mag_scalar(np.gradient(Z)[1])
    bz = 0.5*(1.0 + np.tanh((gZ - tz)*sz))
    bk = 0.5*(1.0 + np.tanh((lap - tk)*sk))
    beta = beta_hi * np.maximum(bz, bk)[...,None]
    fused = (1.0-beta)*img + beta*N4
    c = 1.0/(1.0+4.0*lam); n = lam/(1.0+4.0*lam)
    return roll_cross_rgb(fused, c=c, n=n)

def fusion_shadow_aware(I, Z, gT, lam1, lam2, thr, boost):
    I1 = robin_harmonic(I, Z, lam=lam1)
    L  = luma(I1); g = grad_mag_scalar(L)
    M  = np.clip((g - thr) / max(1e-6, (0.20-thr)), 0.0, 1.0)
    M2 = np.clip(M + boost*gT, 0.0, 1.0)[...,None]
    I2 = robin_harmonic(I1, Z, lam=lam2)
    return I1*(1.0 - M2) + I2*(M2)

def exposure_normalize(img_lin, target=0.18, smin=0.8, smax=1.25):
    m = float(np.mean(luma(img_lin)))
    s = np.clip(target / max(m, 1e-6), smin, smax)
    return img_lin * s, s

# ----------------------------- OIT (Weighted Blended) -----------------------
def oit_weighted_blended(layers, beta, H, W, B):
    C_acc = np.zeros((H,W,3), np.float32)
    A_acc = np.zeros((H,W,1), np.float32)
    for L in layers:
        a = L["alpha"] * L["mask"][...,None]
        w = np.exp(-beta * L["depth"], dtype=np.float32)
        C_acc += (w * a) * L["color"][None,None,:]
        A_acc += (w * a)
    C_out = C_acc / (A_acc + 1e-6)
    A_out = sat01(A_acc)
    return (1.0 - A_out)*B + A_out*C_out

# ----------------------------- Renderer -------------------------------------
class Renderer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def build(self, H, W): return build_spheres_scene(H, W)

    def render_core(self, H, W, cfg_local: Config):
        gb = self.build(H,W)
        L_dir, Tsoft = direct_dir_sun(gb, np.array(cfg_local.sun_dir, np.float32), cfg_local.sun_intensity, cfg_local)
        L_diff, L_spec = ibl_lighting(gb, cfg_local)

        A = ssao_8x4(gb, cfg_local.ao_floor, cfg_local.ao_strength)
        # Specular occlusion for environment reflections
        NoV = np.clip(np.sum(gb.N*gb.V, axis=-1), 0.0, 1.0)
        k_so = sample_so_k(gb.rough, A)
        SO = specular_occlusion(A, gb.rough, NoV, k_so=k_so)[...,None]
        L_spec = L_spec * SO
        L_diff_biased = (1.0 - cfg_local.bent_w)*L_diff + cfg_local.bent_w*(A[...,None]*L_diff)

        L_pre = L_dir + L_spec + L_diff_biased
        gT = grad_mag_scalar(Tsoft); gT = np.clip((gT - 0.02) / 0.10, 0.0, 1.0)
        fused = fusion_shadow_aware(L_pre, gb.Z, gT, cfg_local.fusion_lam1, cfg_local.fusion_lam2, cfg_local.fusion_thr, cfg_local.fusion_boost)

        cov = sdf_edge_coverage(gb, cfg_local.cov_kF)[...,None]
        bg = background_shading(H,W, cfg_local)
        comp = cov*fused + (1.0 - cov)*bg

        # Example translucent triangles (kept for parity with p2; can disable by setting zero alphas)
        tris = [
            {"uv": np.array([[0.25,0.25],[0.45,0.15],[0.40,0.35]], np.float32), "depth": 0.52, "color": np.array([0.9,0.2,0.2], np.float32), "alpha": 0.20},
            {"uv": np.array([[0.60,0.25],[0.80,0.20],[0.75,0.40]], np.float32), "depth": 0.55, "color": np.array([0.2,0.9,0.2], np.float32), "alpha": 0.18},
            {"uv": np.array([[0.40,0.65],[0.65,0.55],[0.55,0.80]], np.float32), "depth": 0.58, "color": np.array([0.2,0.4,0.9], np.float32), "alpha": 0.15},
        ]
        layers = []
        for T in tris:
            mask = tri_coverage(T["uv"], H, W, cfg_local.cov_kF)
            layers.append({"color": T["color"], "alpha": T["alpha"], "depth": T["depth"], "mask": mask})
        comp = oit_weighted_blended(layers, cfg_local.oit_beta, H, W, comp)

        comp, exposure = exposure_normalize(comp, cfg_local.target_gray, cfg_local.exp_min, cfg_local.exp_max)
        return SimpleNamespace(img=comp, ao=A, Tsoft=Tsoft, Z=gb.Z, exposure=exposure)

    # --------------------- Allocator (CTPA) ---------------------
    def allocate(self, stats):
        # Modules (j=1..5): kF, pcf_scale, ao_strength, fusion_lam1, ms_blend
        # Bounds:
        bmin = np.array([1.00, 0.80, 1.00, 0.40, 0.50], np.float32)
        bmax = np.array([1.60, 1.40, 1.60, 0.70, 1.00], np.float32)
        theta_ref = np.array([1.25, 1.00, 1.20, 0.55, 1.00], np.float32)  # nominal
        
        # Costs per module (arbitrary units, constant):
        c = np.array([1.0, 1.0, 1.0, 1.0, 0.5], np.float32)
        B = 5.0  # budget equals sum of nominal costs
        
        # Surrogate parameters (a,b,k) per case
        # Case A (smooth): idx 0,1,3
        aA = np.array([0.30, 0.35, 0.40], np.float32)
        bA = np.array([0.08, 0.10, 0.12], np.float32)
        kA = np.array([2.0, 1.6, 2.2], np.float32)
        # Case B (strength): idx 2
        aB = 0.15; bB = 0.10; epsB = 1e-3
        # Case C (clamp-around-ref): idx 4
        aC = 1.20
        
        # Quality axes weights from stats
        gL, gT, gZ, kurt = stats["gL"], stats["gT"], stats["gZ"], stats["kurt"]
        w = np.array([1.0, 1.0, 1.0, 1.0], np.float32)
        gamma = np.array([1.2, 1.0, 1.0, 0.6], np.float32)
        q = np.array([max(gL,1e-6)**gamma[0], max(gT,1e-6)**gamma[1], max(gZ,1e-6)**gamma[2], max(kurt,1e-6)**gamma[3]], np.float32)
        lam = (w*q) / np.maximum(np.sum(w*q), 1e-6)
        
        # A_ij (axes × modules)
        A = np.array([
            # kF, pcf_scale, ao_strength, fusion_lam1, ms_blend
            [0.60, 0.10, 0.05, 0.25, 0.00],  # i=1 edge aliasing
            [0.05, 0.55, 0.05, 0.35, 0.00],  # i=2 shadow ringing
            [0.00, 0.10, 0.70, 0.20, 0.00],  # i=3 geo occlusion
            [0.05, 0.05, 0.05, 0.05, 0.80],  # i=4 highlight fidelity
        ], np.float32)
        
        L = (lam[:,None]*A).sum(axis=0)  # module loads (length 5)
        
        # mu initializer (simple)
        # Build per-case arrays aligned with theta indices
        # Case A indices [0,1,3]
        LA = L[[0,1,3]]; aA1=aA; bA1=bA; kA1=kA
        xi = (LA * aA1 * kA1) / np.maximum(LA * bA1, 1e-6)
        xi_c = xi * c[[0,1,3]]
        mu0_A = float(np.exp(np.mean(np.log(np.maximum(xi_c, 1e-6)))))
        # Case B index [2]
        yi = L[2] / max(c[2], 1e-6)
        mu0_B = yi
        mu0 = max(1e-3, min(10.0, mu0_A / max(mu0_B,1e-6)))
        
        # Define theta(mu) and dtheta/dmu
        def theta_mu(mu):
            th = np.zeros(5, np.float32)
            # Case A: indices 0,1,3
            for idx, jj in enumerate([0,1,3]):
                num = L[jj]*aA1[idx]*kA1[idx]
                den = L[jj]*bA1[idx] + mu*c[jj]
                val = (1.0/kA1[idx]) * np.log(np.maximum(num/np.maximum(den,1e-6), 1e-6))
                th[jj] = val
            # Case B: j=2
            denom = aB + (mu*c[2])/max(L[2],1e-6)
            th[2] = np.sqrt(max(bB/np.maximum(denom,1e-6), 1e-6)) - epsB
            # Case C: j=4
            th[4] = theta_ref[4] - (mu*c[4])/(2.0 * max(L[4]*aC,1e-6))
            # Clamp
            th = np.minimum(np.maximum(th, bmin), bmax)
            return th
        
        def dtheta_dmu(mu):
            d = np.zeros(5, np.float32)
            # Case A: derivative of (1/k) ln(num/(L b + mu c)) => -(1/k) * c/(L b + mu c)
            for idx, jj in enumerate([0,1,3]):
                den = L[jj]*bA1[idx] + mu*c[jj]
                d[jj] = - (1.0/kA1[idx]) * (c[jj]/np.maximum(den,1e-6))
            # Case B:
            denom = aB + (mu*c[2])/max(L[2],1e-6)
            d[2] = -0.5 * np.sqrt(bB) * ( (c[2]/max(L[2],1e-6)) ) / (np.maximum(denom,1e-6)**1.5)
            # Case C:
            d[4] = - c[4] / (2.0 * max(L[4]*aC,1e-6))
            return d
        
        # Newton one step
        th0 = theta_mu(mu0)
        F0 = float(np.dot(c, th0) - B)
        dth = dtheta_dmu(mu0)
        Fp = float(np.dot(c, dth))
        mu1 = mu0 - F0 / np.sign(Fp) / max(abs(Fp), 1e-6)  # guard sign
        mu = float(np.clip(mu1, 1e-4, 10.0))
        th = theta_mu(mu)
        
        # Optional simplex rescale if cost off due to clamping
        cost = float(np.dot(c, th))
        if abs(cost - B) > 1e-3:
            base = bmin
            num = B - float(np.dot(c, base))
            den = float(np.dot(c, th - base))
            rho = float(num / max(den, 1e-6))
            th = base + rho*(th - base)
            th = np.minimum(np.maximum(th, bmin), bmax)
        
        # Map to config
        out = dict(cov_kF=float(th[0]), pcf_scale=float(th[1]), ao_strength=float(th[2]), fusion_lam1=float(th[3]), ms_blend=float(th[4]),
                   lam=lam.tolist(), loads=L.tolist(), theta=th.tolist(), mu=mu, cost=float(np.dot(c, th)))
        return out

    def stats_from_prepass(self, pre):
        Y = luma(pre.img)
        m = float(np.mean(Y))
        gL = float(np.mean(grad_mag_scalar(Y)))
        gT = float(np.mean(grad_mag_scalar(pre.Tsoft)))
        gZ = float(np.mean(grad_mag_scalar(pre.Z)))
        kurt = float(np.mean((Y - m)**4))
        return dict(gL=gL, gT=gT, gZ=gZ, kurt=kurt, mean_luma=m)

    def render(self, H, W):
        # prepass with current cfg
        pre = self.render_core(H, W, self.cfg)
        if not self.cfg.use_allocator:
            return pre

        # allocate constants
        stats = self.stats_from_prepass(pre)
        alloc = self.allocate(stats)
        # apply
        cfg2 = Config(**{**self.cfg.__dict__,
                         "cov_kF": alloc["cov_kF"],
                         "pcf_scale": alloc["pcf_scale"],
                         "ao_strength": alloc["ao_strength"],
                         "fusion_lam1": alloc["fusion_lam1"],
                         "ms_blend": alloc["ms_blend"],
                         "use_allocator": False})  # avoid recursion
        post = self.render_core(H, W, cfg2)
        post.alloc = alloc
        post.stats = stats
        return post

    def save(self, prefix):
        H=W=self.cfg.size
        out = self.render(H,W)
        img_lin = out.img
        img_srgb = tone_map_aces(img_lin, gain=self.cfg.aces_gain)
        if self.cfg.dither:
            i = np.arange(H)[:,None]; j = np.arange(W)[None,:]
            d = (blue_hash(i, j) - 0.5)*self.cfg.dither_amp
            img_srgb = sat01(img_srgb + d[...,None])
        iio.imwrite(prefix + "_srgb.png",  (sat01(img_srgb)*255).astype(np.uint8))
        iio.imwrite(prefix + "_linear.png", (sat01(img_lin)*255).astype(np.uint8))
        iio.imwrite(prefix + "_ao.png",    (sat01(out.ao)*255).astype(np.uint8))
        iio.imwrite(prefix + "_shadow.png",(sat01(out.Tsoft)*255).astype(np.uint8))
        # dump allocation
        with open(prefix + "_alloc.json","w") as f:
            import json; json.dump(dict(alloc=out.alloc, stats=out.stats, version=self.cfg.version), f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=640, help="square resolution (pixels)")
    ap.add_argument("--prefix", type=str, default="v001p3/spheres", help="output prefix (path)")
    ap.add_argument("--no-dither", action="store_true", help="disable output dithering")
    ap.add_argument("--no-alloc", action="store_true", help="disable allocator prepass (use current cfg constants)")
    args = ap.parse_args()

    cfg = Config(size=args.size, dither=(not args.no_dither), use_allocator=(not args.no_alloc))
    os.makedirs(os.path.dirname(args.prefix) if os.path.dirname(args.prefix) else ".", exist_ok=True)
    Renderer(cfg).save(args.prefix)

if __name__ == "__main__":
    main()
