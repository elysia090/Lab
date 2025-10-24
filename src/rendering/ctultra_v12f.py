#!/usr/bin/env python3
# CT-ULTRA v1.2f — Constant-time renderer (NumPy + ImageIO only)
# Fixed taps, deterministic. Fast profile.

import os, sys, json, argparse
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import imageio.v2 as iio

EPS = 1e-6

def saturate(x):
    return np.minimum(1.0, np.maximum(0.0, x)).astype(np.float32)

def luma(rgb):
    return (0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]).astype(np.float32)

def _slice_from_maxpad(pad, dx, dy, max_dx, max_dy):
    H = pad.shape[0] - 2*max_dy
    W = pad.shape[1] - 2*max_dx
    y0 = max_dy - dy
    x0 = max_dx - dx
    if pad.ndim == 3:
        return pad[y0:y0+H, x0:x0+W, :]
    else:
        return pad[y0:y0+H, x0:x0+W]

def shift_clamp_rgb(arr, dx, dy):
    H,W,_ = arr.shape
    px,py = abs(dx), abs(dy)
    ap = np.pad(arr, ((py,py),(px,px),(0,0)), mode='edge')
    y0 = py - dy; x0 = px - dx
    return ap[y0:y0+H, x0:x0+W]

def shift_clamp_gray(arr, dx, dy):
    H,W = arr.shape
    px,py = abs(dx), abs(dy)
    ap = np.pad(arr, ((py,py),(px,px)), mode='edge')
    y0 = py - dy; x0 = px - dx
    return ap[y0:y0+H, x0:x0+W]

def roll_cross_rgb(img, c, n):
    s = c + 4.0*n
    L = shift_clamp_rgb(img, -1,  0)
    R = shift_clamp_rgb(img,  1,  0)
    U = shift_clamp_rgb(img,  0, -1)
    D = shift_clamp_rgb(img,  0,  1)
    return (c*img + n*(L+R+U+D)) / s

def roll_cross_scalar(img, c, n):
    s = c + 4.0*n
    L = shift_clamp_gray(img, -1,  0)
    R = shift_clamp_gray(img,  1,  0)
    U = shift_clamp_gray(img,  0, -1)
    D = shift_clamp_gray(img,  0,  1)
    return (c*img + n*(L+R+U+D)) / s

def grad_mag_scalar(img):
    gx = shift_clamp_gray(img,  1, 0) - shift_clamp_gray(img, -1, 0)
    gy = shift_clamp_gray(img,  0, 1) - shift_clamp_gray(img,  0,-1)
    return np.sqrt(0.25*(gx*gx + gy*gy)).astype(np.float32)

def tone_map_aces(linear_rgb, gain=1.05):
    x = np.maximum(0.0, linear_rgb) * gain
    a=2.51; b=0.03; c=2.43; d=0.59; e=0.14
    y = (x*(a*x+b)) / (x*(c*x+d)+e + EPS)
    sr = np.where(y <= 0.0031308, 12.92*y, 1.055*np.power(np.maximum(y,0.0), 1/2.4) - 0.055)
    return saturate(sr)

@dataclass(frozen=True)
class RenderConfig:
    version: str = "ctultra_v1.2f"
    size: int = 640
    sun_dir: tuple = (0.6, 0.4, 1.0)
    sun_intensity: float = 2.5
    ao_floor: float = 0.90
    ao_strength: float = 1.20
    fusion_lambda1: float = 0.55
    coverage_k: float = 1.25
    pcf_r0: float = 0.55
    pcf_r1: float = 0.70
    pcf_eta_t0: float = 0.00
    pcf_eta_wmax: float = 0.50
    pcf_scale: float = 0.95
    oit_beta: float = 2.0
    ms_blend: float = 1.00
    aces_gain: float = 1.05
    target_gray: float = 0.18
    exposure_min: float = 0.80
    exposure_max: float = 1.25
    exposure_mode: str = "p50"  # "mean" or "p50"
    dither: bool = True
    dither_amp: float = 0.75/255.0
    use_allocator: bool = True

def _env_color(dir3):
    z = dir3[...,2:3]
    sdir = np.array([0.2,0.2,0.95], np.float32)
    cos_s = np.maximum(0.0, np.sum(dir3*sdir, axis=-1, keepdims=True))
    sun = (cos_s**64) * np.array([1.0,0.9,0.7], np.float32)[None,None,:] * 2.6
    sky = np.concatenate([0.45 + 0.25*z, 0.55 + 0.30*z, 0.65 + 0.35*z], -1)
    return (sky + sun).astype(np.float32)

def _sh2_project(env_fn, S=48):
    u = np.linspace(0,1,S, endpoint=False); v = np.linspace(0,1,2*S, endpoint=False)
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

_SH2 = _sh2_project(_env_color, 48)

def _sh2_eval(n):
    x = n[...,0]; y = n[...,1]; z = n[...,2]
    c = _SH2
    E = np.zeros(n.shape, np.float32)
    E += 0.282095 * c[0]
    E += (0.488603*y)[...,None] * c[1]
    E += (0.488603*z)[...,None] * c[2]
    E += (0.488603*x)[...,None] * c[3]
    E += (1.092548*x*y)[...,None] * c[4]
    E += (1.092548*y*z)[...,None] * c[5]
    E += (0.315392*(3*z*z - 1))[...,None] * c[6]
    E += (1.092548*x*z)[...,None] * c[7]
    E += (0.546274*(x*x - y*y))[...,None] * c[8]
    return saturate(E)

def _importance_sample_ggx(xi, alpha):
    phi = 2*np.pi*xi[...,0]
    cos2 = (1 - xi[...,1]) / (1 + (alpha*alpha - 1)*xi[...,1] + EPS)
    cosTheta = np.sqrt(np.clip(cos2, 0.0, 1.0))
    sinTheta = np.sqrt(np.maximum(0.0, 1.0 - cosTheta*cosTheta))
    return np.stack([np.cos(phi)*sinTheta, np.sin(phi)*sinTheta, cosTheta], -1).astype(np.float32)

def build_fgd_lut(nov_res=32, rough_res=32, samples=32, seed=1337):
    rng = np.random.RandomState(seed)
    A = np.zeros((nov_res, rough_res), np.float32)
    B = np.zeros_like(A)
    for i in range(nov_res):
        NoV = (i + 0.5)/nov_res
        V = np.array([np.sqrt(max(0.0, 1-NoV*NoV)), 0.0, NoV], np.float32)
        for j in range(rough_res):
            rough = (j + 0.5)/rough_res
            alpha = max(1e-4, rough*rough)
            accA = 0.0; accB = 0.0
            for _ in range(samples):
                xi = rng.rand(2).astype(np.float32)
                H = _importance_sample_ggx(xi, alpha)
                L = np.array([ -V[0] + 2*H[0]*np.dot(V,H),
                               -V[1] + 2*H[1]*np.dot(V,H),
                               -V[2] + 2*H[2]*np.dot(V,H)], np.float32)
                NoL = max(0.0, L[2]); NoH = max(0.0, H[2]); VoH = max(0.0, np.dot(V,H))
                if NoL > 0 and NoH > 0:
                    k = (alpha*alpha)/2.0
                    def G1(nv): return nv / (nv*(1.0 - k) + k + EPS)
                    G = G1(NoV) * G1(NoL)
                    G_Vis = (G * VoH) / (NoH * NoV + EPS)
                    Fc = (1.0 - VoH)**5
                    accA += (1.0 - Fc) * G_Vis
                    accB += Fc * G_Vis
            A[i,j] = accA / samples
            B[i,j] = accB / samples
    return A, B

_FGD_A, _FGD_B = build_fgd_lut(32, 32, samples=32)

def sample_fgd(nov, rough):
    nov = np.clip(nov, 0.0, 1.0); rough = np.clip(rough, 0.0, 1.0)
    i = nov*31.0; j = rough*31.0
    i0 = np.floor(i).astype(np.int32); j0 = np.floor(j).astype(np.int32)
    i1 = np.clip(i0+1, 0, 31); j1 = np.clip(j0+1, 0, 31)
    ti = i - i0; tj = j - j0
    A00 = _FGD_A[i0,j0]; A01 = _FGD_A[i0,j1]; A10 = _FGD_A[i1,j0]; A11 = _FGD_A[i1,j1]
    B00 = _FGD_B[i0,j0]; B01 = _FGD_B[i0,j1]; B10 = _FGD_B[i1,j0]; B11 = _FGD_B[i1,j1]
    A = (1-ti)*((1-tj)*A00 + tj*A01) + ti*((1-tj)*A10 + tj*A11)
    B = (1-ti)*((1-tj)*B00 + tj*B01) + ti*((1-tj)*B10 + tj*B11)
    return A.astype(np.float32), B.astype(np.float32)

_SO_TAB = np.zeros((16,16), np.float32)
for ri in range(16):
    for ai in range(16):
        r = (ri+0.5)/16.0
        a = (ai+0.5)/16.0
        _SO_TAB[ri,ai] = np.clip(0.7 + 0.6*r - 0.3*a, 0.6, 1.2)

def sample_so_k(rough, ao):
    rough = np.clip(rough, 0.0, 1.0); ao = np.clip(ao, 0.0, 1.0)
    i = rough*15.0; j = ao*15.0
    i0 = np.floor(i).astype(np.int32); j0 = np.floor(j).astype(np.int32)
    i1 = np.clip(i0+1, 0, 15); j1 = np.clip(j0+1, 0, 15)
    ti = (i - i0); tj = (j - j0)
    v = (1-ti)*((1-tj)*_SO_TAB[i0,j0] + tj*_SO_TAB[i0,j1]) + ti*((1-tj)*_SO_TAB[i1,j0] + tj*_SO_TAB[i1,j1])
    return v.astype(np.float32)

def specular_occlusion(AO, rough, NoV, k_so):
    return np.clip(1.0 - k_so*(1.0 - AO)*(1.0 - NoV)**2 * np.clip(rough,0.0,1.0), 0.0, 1.0).astype(np.float32)

def build_spheres(size):
    H=W=size
    centers = np.array([[0.30,0.35,0.55],[0.68,0.40,0.45],[0.50,0.68,0.60],[0.24,0.74,0.40],[0.78,0.74,0.70]], np.float32)
    radii   = np.array([0.18,0.16,0.14,0.10,0.12], np.float32)
    base_s  = np.array([[0.85,0.36,0.36],[0.36,0.85,0.42],[0.38,0.50,0.88],[0.85,0.80,0.45],[0.80,0.80,0.85]], np.float32)
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

    Z = np.min(Z_s, axis=0)
    hit = np.isfinite(Z)
    sid = np.argmin(Z_s, axis=0)
    ii = np.arange(H)[:,None]; jj = np.arange(W)[None,:]

    N = np.zeros((H,W,3), np.float32); P = np.zeros((H,W,3), np.float32)
    base = np.zeros((H,W,3), np.float32); rough = np.zeros((H,W), np.float32); metal = np.zeros((H,W), np.float32)

    N[:,:,0] = Nx[sid, ii, jj]; N[:,:,1] = Ny[sid, ii, jj]; N[:,:,2] = Nz[sid, ii, jj]
    P[:,:,0] = u; P[:,:,1] = v; P[:,:,2] = Z

    for c in range(3): base[:,:,c] = base_s[sid, c]
    rough = rough_s[sid]; metal = metal_s[sid]

    bg = ~hit
    N[bg] = np.array([0,0,1], np.float32)
    Z[bg] = 1.2; P[bg,2] = Z[bg]
    base[bg] = np.array([0.22,0.24,0.26], np.float32)
    rough[bg] = 0.65; metal[bg] = 0.0

    V = np.dstack([np.zeros_like(Z), np.zeros_like(Z), np.ones_like(Z)])
    T = np.zeros_like(N); B = np.zeros_like(N)
    return SimpleNamespace(P=P,N=N,T=T,B=B,V=V, Z=Z, base=base, rough=rough, metal=metal, sid=sid, u=u, v=v, centers=centers, radii=radii)

def F_schlick(F0, u):
    return F0 + (1.0 - F0)*(1.0 - u)[...,None]**5

def D_ggx(NoH, alpha):
    a2 = alpha*alpha; d = (NoH*NoH*(a2 - 1.0) + 1.0)
    return (a2 / (np.pi * d*d + EPS))[...,None]

def G1_smith(t, alpha):
    return (2.0*t / (t + np.sqrt(alpha*alpha + (1.0 - alpha*alpha)*t*t + EPS) + EPS))[...,None]

def G_smith(NoV, NoL, alpha):
    return G1_smith(NoV, alpha) * G1_smith(NoL, alpha)

def ms_energy_comp(F0, alpha, blend):
    Favg = np.mean(F0, axis=-1, keepdims=True)
    Ems = 1.0 - 0.4399*alpha[...,None] + 0.0927*(alpha[...,None]**2)
    Ems = saturate(Ems)
    Ems = blend*Ems + (1.0-blend)*0.0
    return Favg, Ems

def brdf_terms(n, v, l, base, F0, alpha, metallic, ms_blend):
    h = (v + l); h = h/np.clip(np.linalg.norm(h, axis=-1, keepdims=True), 1e-6, None)
    NoV = np.clip(np.sum(n*v, -1), 0.0, 1.0); NoL = np.clip(np.sum(n*l, -1), 0.0, 1.0)
    NoH = np.clip(np.sum(n*h, -1), 0.0, 1.0); VoH = np.clip(np.sum(v*h, -1), 0.0, 1.0)
    D = D_ggx(NoH, alpha); G = G_smith(NoV, NoL, alpha); F = F_schlick(F0, VoH)
    spec = (D * G * F) / np.maximum(4.0*NoV[...,None]*NoL[...,None] + EPS, EPS)
    Favg, Ems = ms_energy_comp(F0, alpha, ms_blend)
    rho_eff = base * (1.0 - Favg) + Ems * Favg
    Fv = F_schlick(F0, NoV)
    kd = (1.0 - Fv) * (1.0 - metallic[...,None])
    diff = (kd * rho_eff / np.pi)
    return diff, spec

def _pcf_batch(arr, offsets):
    if len(offsets) == 0:
        return arr
    max_dx = max(abs(dx) for dx,dy in offsets)
    max_dy = max(abs(dy) for dx,dy in offsets)
    if arr.ndim == 3:
        pad = np.pad(arr, ((max_dy,max_dy),(max_dx,max_dx),(0,0)), mode='edge')
    else:
        pad = np.pad(arr, ((max_dy,max_dy),(max_dx,max_dx)), mode='edge')
    acc = np.zeros_like(arr, dtype=np.float32)
    for dx, dy in offsets:
        acc += _slice_from_maxpad(pad, dx, dy, max_dx, max_dy)
    return acc / float(len(offsets))

def pass_direct_and_shadow(gb, cfg, coverage_k):
    Ld = np.array(cfg.sun_dir, np.float32); Ld = Ld/np.linalg.norm(Ld)
    L = np.broadcast_to(Ld, gb.P.shape)
    F0 = 0.04*(1.0 - gb.metal)[...,None] + gb.base*gb.metal[...,None]

    Nx,Ny,Nz = gb.N[:,:,0], gb.N[:,:,1], gb.N[:,:,2]
    gx = np.sqrt(0.25*((shift_clamp_gray(Nx,1,0)-shift_clamp_gray(Nx,-1,0))**2 + (shift_clamp_gray(Nx,0,1)-shift_clamp_gray(Nx,0,-1))**2))
    gy = np.sqrt(0.25*((shift_clamp_gray(Ny,1,0)-shift_clamp_gray(Ny,-1,0))**2 + (shift_clamp_gray(Ny,0,1)-shift_clamp_gray(Ny,0,-1))**2))
    gz = np.sqrt(0.25*((shift_clamp_gray(Nz,1,0)-shift_clamp_gray(Nz,-1,0))**2 + (shift_clamp_gray(Nz,0,1)-shift_clamp_gray(Nz,0,-1))**2))
    varN = 0.5*(gx+gy+gz).astype(np.float32)
    k_tok = np.where(varN < 0.05, 0.20 + (0.50-0.20)*(varN/0.05),
                     np.where(varN < 0.12, 0.50 + (0.80-0.50)*((varN-0.05)/0.07), 0.80)).astype(np.float32)
    alpha_raw = np.sqrt(np.clip(gb.rough**2 + k_tok*varN, 1e-4, 1.0))
    alpha = np.minimum(np.maximum(alpha_raw, gb.rough), np.sqrt(gb.rough*gb.rough + 0.35*varN))

    diff, spec = brdf_terms(gb.N, gb.V, L, gb.base, F0, alpha, gb.metal, cfg.ms_blend)
    NoL = np.clip(np.sum(gb.N*L, -1, keepdims=True), 0, 1)
    Lo = (diff + spec) * cfg.sun_intensity * NoL

    C = gb.centers; R = gb.radii; S = C.shape[0]
    Pc = gb.P[...,None,:] - C[None,None,:,:]
    b = np.sum(L[... ,None,:]*Pc, axis=-1)
    c = np.sum(Pc*Pc, axis=-1) - R[None,None,:]**2
    disc = b*b - c
    t = (-b - np.sqrt(np.maximum(disc, 0)))
    self_mask = (gb.sid[...,None] == np.arange(S)[None,None,:])
    occ = (disc > 0) & (t > 1e-4) & (~self_mask)
    T_hard = 1.0 - occ.any(axis=-1, keepdims=True).astype(np.float32)

    t_pos = np.where(occ, t, np.inf)
    t_min = np.min(t_pos, axis=-1)
    t_min[~np.isfinite(t_min)] = 0.0
    eta = np.clip((t_min - cfg.pcf_eta_t0) / max(cfg.pcf_eta_wmax, 1e-6), 0.0, 1.0).astype(np.float32)

    H,W = T_hard.shape[:2]
    baseA = np.array([[-0.5,-0.1],[0.6,-0.4],[-0.2,0.7],[0.8,0.5],[-0.7,0.3],[0.1,0.9],[-0.9,-0.2],[0.3,-0.8],[-0.6,0.6]], np.float32)
    baseB = np.array([[ 0.4, 0.2],[-0.6,-0.5],[ 0.7,-0.3],[-0.3, 0.8],[ 0.2,-0.9],[-0.8, 0.4],[ 0.1, 0.7],[-0.5,-0.7],[ 0.6, 0.6]], np.float32)
    base = baseA if ((H & 1)==0) else baseB

    r_eff = (cfg.pcf_r0 + cfg.pcf_r1 * eta) * cfg.pcf_scale
    r_mean = float(np.mean(r_eff))
    r_s = max(1, int(round(0.7 * r_mean)))
    r_l = max(1, int(round(1.6 * r_mean)))

    def mk_offsets(rpix):
        return [(int(round(base[k,0]*rpix)), int(round(base[k,1]*rpix))) for k in range(base.shape[0])]

    T_s = _pcf_batch(T_hard, mk_offsets(r_s))
    T_l = _pcf_batch(T_hard, mk_offsets(r_l))
    T_soft = (1.0 - eta[...,None])*T_s + eta[...,None]*T_l
    T_soft = saturate(T_soft)
    return Lo * T_soft, T_soft[...,0]

def pass_ibl(gb, ao_map):
    F0 = 0.04*(1.0 - gb.metal)[...,None] + gb.base*gb.metal[...,None]
    Eirr = _sh2_eval(gb.N)
    Ldiff = (gb.base/np.pi) * Eirr
    R = 2*(np.sum(gb.N*gb.V, axis=-1, keepdims=True))*gb.N - gb.V
    R = R/np.clip(np.linalg.norm(R, axis=-1, keepdims=True), 1e-6, None)
    avg = np.array([0.55,0.60,0.65], np.float32)
    env_pref = (1.0 - (gb.rough*gb.rough)[...,None]) * _env_color(R) + (gb.rough*gb.rough)[...,None]*avg
    NoV = np.clip(np.sum(gb.N*gb.V, -1), 0.0, 1.0)
    Afgd, Bfgd = sample_fgd(NoV, gb.rough)
    Lspec = env_pref * (F0*Afgd[...,None] + (1.0 - F0)*Bfgd[...,None])
    k_so = sample_so_k(gb.rough, ao_map)
    SO = specular_occlusion(ao_map, gb.rough, NoV, k_so)[...,None]
    Lspec = Lspec * SO
    return Ldiff, Lspec

def _shift_batch_gray(img, offsets):
    if len(offsets) == 0:
        return np.empty((0,)+img.shape, img.dtype)
    max_dx = max(abs(dx) for dx,dy in offsets)
    max_dy = max(abs(dy) for dx,dy in offsets)
    pad = np.pad(img, ((max_dy,max_dy),(max_dx,max_dx)), mode='edge')
    H, W = img.shape
    out = np.empty((len(offsets), H, W), dtype=img.dtype)
    for i,(dx,dy) in enumerate(offsets):
        out[i] = _slice_from_maxpad(pad, dx, dy, max_dx, max_dy)
    return out

def pass_ssao(gb, floor_val, strength_base):
    Z = gb.Z
    H,W = Z.shape
    k = np.clip(1.0 - 0.8*((floor_val - 0.85)/0.15), 0.5, 1.2).astype(np.float32)
    strength = strength_base * k * 1.45

    dirs_A = np.array([[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1],[2,1],[-1,2]], np.float32)
    dirs_A = (dirs_A.T/np.maximum(1.0,np.linalg.norm(dirs_A,axis=1))).T
    dirs_B = np.array([[1,1],[1,0],[0,1],[-1,0],[-1,1],[-1,-1],[0,-1],[1,-1],[2,-1],[1,2]], np.float32)
    dirs_B = (dirs_B.T/np.maximum(1.0,np.linalg.norm(dirs_B,axis=1))).T
    r_steps = np.array([2,4,7,10], np.int32)

    def accum(dirs):
        offs = []
        for d in dirs:
            for r in r_steps:
                offs.append((int(np.rint(d[0]*r)), int(np.rint(d[1]*r))))
        Zs = _shift_batch_gray(Z, offs)
        dz = Zs - Z[None, ...] - 0.0015
        dz = np.maximum(dz, 0.0)
        acc = dz.mean(axis=0, dtype=np.float32)
        return acc

    accA = accum(dirs_A)
    accB = accum(dirs_B)
    row_mask = (np.arange(H) & 1).astype(np.float32)[:,None]
    acc = (1.0 - row_mask) * accA + row_mask * accB

    ao = saturate(1.0 - strength*acc)
    ao2 = roll_cross_scalar(ao, c=4.0, n=1.0)
    return np.maximum(ao2, floor_val).astype(np.float32)

def coverage_sdf_sphere(gb, kF):
    sid = gb.sid; C = gb.centers; R = gb.radii
    cx = C[sid,0]; cy = C[sid,1]; r = R[sid]
    dx = gb.u - cx; dy = gb.v - cy
    s = np.sqrt(np.maximum(0.0, dx*dx + dy*dy)) - r
    F = (1.0/gb.Z.shape[1] + 1.0/gb.Z.shape[0]) * kF
    cov = saturate(0.5 - s / (F+EPS))
    return saturate(cov*(2.0 - cov))

def tri_coverage(UV, H, W, kF):
    y,x = np.mgrid[0:H,0:W].astype(np.float32)
    p = np.stack([x/W, y/H], -1)
    v0,v1,v2 = UV[0], UV[1], UV[2]
    def edge_func(a, b, x):
        ab = b - a
        return (x[...,0]-a[0])*(ab[1]) - (x[...,1]-a[1])*(ab[0]), np.linalg.norm(ab) + 1e-8
    e0, l0 = edge_func(v0, v1, p); e1, l1 = edge_func(v1, v2, p); e2, l2 = edge_func(v2, v0, p)
    d0 = e0 / l0; d1 = e1 / l1; d2 = e2 / l2
    F = (1.0/W + 1.0/H) * kF
    cov = saturate(0.5 + d0/F) * saturate(0.5 + d1/F) * saturate(0.5 + d2/F)
    return saturate(cov*(2.0 - cov))

def oit_weighted_blended(layers, beta, H, W, base_rgb):
    C_acc = np.zeros((H,W,3), np.float32)
    A_acc = np.zeros((H,W,1), np.float32)
    for L in layers:
        a = L["alpha"] * L["mask"][...,None]
        w = np.exp(-beta * L["depth"]).astype(np.float32)
        C_acc += (w * a) * L["color"][None,None,:]
        A_acc += (w * a)
    C_out = C_acc / (A_acc + 1e-6)
    A_out = saturate(A_acc)
    return (1.0 - A_out)*base_rgb + A_out*C_out

def robin_harmonic(rgb, Z, lam=0.55, beta_hi=0.34, tz=0.10, tk=0.03, sz=14.0, sk=12.0):
    n4 = roll_cross_rgb(rgb, c=0.0, n=0.25)
    gZx = shift_clamp_gray(Z, 1, 0) - shift_clamp_gray(Z, -1, 0)
    gZy = shift_clamp_gray(Z, 0, 1) - shift_clamp_gray(Z, 0, -1)
    gZ  = np.sqrt(0.25*(gZx*gZx + gZy*gZy)) / (np.abs(Z)+EPS)
    lap = (shift_clamp_gray(Z,1,0) + shift_clamp_gray(Z,-1,0) +
           shift_clamp_gray(Z,0,1) + shift_clamp_gray(Z,0,-1) - 4.0*Z)
    lap = np.abs(lap)
    bz = 0.5*(1.0 + np.tanh((gZ - tz)*sz))
    bk = 0.5*(1.0 + np.tanh((lap - tk)*sk))
    beta = beta_hi * np.maximum(bz, bk)[...,None]
    fused = (1.0-beta)*rgb + beta*n4
    c = 1.0/(1.0+4.0*lam); n = lam/(1.0+4.0*lam)
    return roll_cross_rgb(fused, c=c, n=n)

def exposure_normalize(rgb_linear, target=0.18, smin=0.8, smax=1.25, mode="p50"):
    Y = luma(rgb_linear)
    if mode == "p50":
        hist, edges = np.histogram(np.clip(Y,0,1), bins=256, range=(0.0,1.0))
        c = np.cumsum(hist)
        p = np.searchsorted(c, (c[-1]//2))
        mid = 0.5*(edges[p] + edges[p+1]) if p+1 < len(edges) else float(np.mean(Y))
        m = float(mid if mid>1e-6 else np.mean(Y))
    else:
        m = float(np.mean(Y))
    s = np.clip(target / max(m, 1e-6), smin, smax)
    return rgb_linear * s, s

def pass_stats(gb, pre_rgb, t_soft):
    Y = luma(pre_rgb)
    m = float(np.mean(Y))
    grad_L = float(np.mean(grad_mag_scalar(Y)))
    grad_T = float(np.mean(grad_mag_scalar(t_soft)))
    grad_Z = float(np.mean(grad_mag_scalar(gb.Z)))
    kurt = float(np.mean((Y - m)**4))
    tvar = 0.0
    return dict(mean_luma=m, gL=grad_L, gT=grad_T, gZ=grad_Z, kurt=kurt, tv=tvar)

def allocate_constants(stats, cfg):
    bmin = np.array([1.00, 0.80, 1.00, 0.40, 0.50], np.float32)
    bmax = np.array([1.60, 1.40, 1.60, 0.70, 1.00], np.float32)
    theta_ref = np.array([cfg.coverage_k, cfg.pcf_scale, cfg.ao_strength, cfg.fusion_lambda1, cfg.ms_blend], np.float32)
    c = np.array([1.0, 1.0, 1.0, 1.0, 0.5], np.float32); B = float(np.dot(c, theta_ref))

    gL,gT,gZ,kurt,tv = stats["gL"], stats["gT"], stats["gZ"], stats["kurt"], stats["tv"]
    w = np.array([1,1,1,1,1,1], np.float32)
    gamma = np.array([1.2,1.0,1.0,0.6,0.8,0.4], np.float32)
    q = np.array([max(gL,1e-6)**gamma[0], max(gT,1e-6)**gamma[1], max(gZ,1e-6)**gamma[2],
                  max(kurt,1e-6)**gamma[3], max(tv,1e-6)**gamma[4], 1.0], np.float32)
    lam = (w*q) / np.maximum(np.sum(w*q), 1e-6)

    A = np.array([
        [0.60, 0.10, 0.05, 0.25, 0.00],
        [0.05, 0.55, 0.05, 0.35, 0.00],
        [0.00, 0.10, 0.70, 0.20, 0.00],
        [0.05, 0.05, 0.05, 0.05, 0.80],
        [0.00, 0.05, 0.05, 0.10, 0.05],
        [0.05, 0.05, 0.05, 0.05, 0.05],
    ], np.float32)
    L = (lam[:,None]*A).sum(axis=0)

    aA = np.array([0.30, 0.35, 0.40], np.float32)
    bA = np.array([0.08, 0.10, 0.12], np.float32)
    kA = np.array([2.0, 1.6, 2.2], np.float32)
    aB = 0.15; bB = 0.10; epsB = 1e-3
    aC = 1.20

    LA = L[[0,1,3]]
    xi = (LA * aA * kA) / np.maximum(LA * bA, 1e-6)
    xi_c = xi * c[[0,1,3]]
    mu0_A = float(np.exp(np.mean(np.log(np.maximum(xi_c, 1e-6)))))
    yi = L[2] / max(c[2], 1e-6); mu0_B = yi
    mu0 = max(1e-3, min(10.0, mu0_A / max(mu0_B,1e-6)))

    def theta_mu(mu):
        th = np.zeros(5, np.float32)
        for idx, jj in enumerate([0,1,3]):
            num = L[jj]*aA[idx]*kA[idx]
            den = L[jj]*bA[idx] + mu*c[jj]
            val = (1.0/kA[idx]) * np.log(np.maximum(num/np.maximum(den,1e-6), 1e-6))
            th[jj] = val
        denom = aB + (mu*c[2])/max(L[2],1e-6)
        th[2] = np.sqrt(max(bB/np.maximum(denom,1e-6), 1e-6)) - epsB
        th[4] = theta_ref[4] - (mu*c[4])/(2.0 * max(L[4]*aC,1e-6))
        th = np.minimum(np.maximum(th, bmin), bmax)
        return th

    def dtheta_dmu(mu):
        d = np.zeros(5, np.float32)
        for idx, jj in enumerate([0,1,3]):
            den = L[jj]*bA[idx] + mu*c[jj]
            d[jj] = - (1.0/kA[idx]) * (c[jj]/np.maximum(den,1e-6))
        denom = aB + (mu*c[2])/max(L[2],1e-6)
        d[2] = -0.5 * np.sqrt(bB) * ( (c[2]/max(L[2],1e-6)) ) / (np.maximum(denom,1e-6)**1.5)
        d[4] = - c[4] / (2.0 * max(L[4]*aC,1e-6))
        return d

    th0 = theta_mu(mu0)
    F0 = float(np.dot(c, th0) - B)
    dth = dtheta_dmu(mu0); Fp = float(np.dot(c, dth))
    mu1 = mu0 - F0 / (Fp if abs(Fp)>1e-6 else np.sign(Fp)+1e-6)
    mu = float(np.clip(mu1, 1e-4, 10.0))
    th = theta_mu(mu)

    cost = float(np.dot(c, th))
    if abs(cost - B) > 1e-3:
        base = bmin
        num = B - float(np.dot(c, base))
        den = float(np.dot(c, th - base))
        rho = float(num / max(den, 1e-6))
        th = base + rho*(th - base)
        th = np.minimum(np.maximum(th, bmin), bmax)

    pcf_profile = "S" if th[1] < 1.1 else "L"
    ssao_profile = 40
    oit_profile = "WB"
    taa_alpha_id = "a1"

    return dict(theta=th.tolist(), mu=mu, cost=float(np.dot(c, th)),
                lam=lam.tolist(), loads=L.tolist(),
                profiles=dict(pcf=pcf_profile, ssao=ssao_profile, oit=oit_profile, taa=taa_alpha_id))

def blue_tile_hash(T=64):
    i = np.arange(T, dtype=np.uint32)[:,None]
    j = np.arange(T, dtype=np.uint32)[None,:]
    x = (i*1664525 + j*1013904223) ^ np.uint32(0x68bc21eb)
    x ^= (x>>13); x *= np.uint32(1274126177); x ^= (x>>16)
    return ((x & np.uint32(1023)).astype(np.float32) / 1023.0)

def add_dither_inplace(srgb, amp, tile=64):
    H,W,_ = srgb.shape
    t = blue_tile_hash(tile)
    reps_y = (H + tile - 1) // tile
    reps_x = (W + tile - 1) // tile
    d = np.tile(t, (reps_y, reps_x))[:H,:W]
    srgb += (d - 0.5)[...,None]*amp
    np.clip(srgb, 0.0, 1.0, out=srgb)

class RenderEngine:
    def __init__(self, cfg: RenderConfig):
        self.cfg = cfg

    def render(self, size, prefix):
        H=W=size
        gb = build_spheres(size)

        pre_Ldir, pre_T = pass_direct_and_shadow(gb, self.cfg, self.cfg.coverage_k)
        pre_Ldiff, pre_Lspec = pass_ibl(gb, np.ones_like(gb.Z, np.float32))
        pre_rgb = pre_Ldir + pre_Ldiff + pre_Lspec

        stats = pass_stats(gb, pre_rgb, pre_T)
        if self.cfg.use_allocator:
            alloc = allocate_constants(stats, self.cfg)
            th = alloc["theta"]
            cfg2 = RenderConfig(**{**self.cfg.__dict__,
                                   "coverage_k": float(th[0]),
                                   "pcf_scale": float(th[1]),
                                   "ao_strength": float(th[2]),
                                   "fusion_lambda1": float(th[3]),
                                   "ms_blend": float(th[4]),
                                   "use_allocator": False})
        else:
            alloc = {"theta":[self.cfg.coverage_k, self.cfg.pcf_scale, self.cfg.ao_strength, self.cfg.fusion_lambda1, self.cfg.ms_blend],
                     "profiles": {"pcf":"S","ssao":40,"oit":"WB","taa":"a1"}, "lam":[], "loads":[], "mu":0.0, "cost":0.0}
            cfg2 = self.cfg

        L_dir, T_soft = pass_direct_and_shadow(gb, cfg2, cfg2.coverage_k)
        AO = pass_ssao(gb, cfg2.ao_floor, cfg2.ao_strength)
        L_diff, L_spec = pass_ibl(gb, AO)
        L_diff = 0.85*L_diff + 0.15*(AO[...,None]*L_diff)

        comp = L_dir + L_spec + L_diff

        cov = coverage_sdf_sphere(gb, cfg2.coverage_k)[...,None]
        bgN = np.dstack([np.zeros((H,W),np.float32), np.zeros((H,W),np.float32), np.ones((H,W),np.float32)])
        bgV = bgN.copy()
        bg_base = np.array([0.22,0.24,0.26], np.float32)[None,None,:]*np.ones((H,W,1), np.float32)
        bg_rough = 0.65*np.ones((H,W), np.float32); bg_metal = np.zeros((H,W), np.float32)
        bg = SimpleNamespace(N=bgN,V=bgV, base=bg_base, rough=bg_rough, metal=bg_metal)
        bg_diff, bg_spec = pass_ibl(bg, np.ones((H,W), np.float32))
        L_bg = bg_diff + bg_spec

        comp = cov*comp + (1.0 - cov)*L_bg

        tris = [
            {"uv": np.array([[0.25,0.25],[0.45,0.15],[0.40,0.35]], np.float32), "depth": 0.52, "color": np.array([0.9,0.2,0.2], np.float32), "alpha": 0.20},
            {"uv": np.array([[0.60,0.25],[0.80,0.20],[0.75,0.40]], np.float32), "depth": 0.55, "color": np.array([0.2,0.9,0.2], np.float32), "alpha": 0.18},
            {"uv": np.array([[0.40,0.65],[0.65,0.55],[0.55,0.80]], np.float32), "depth": 0.58, "color": np.array([0.2,0.4,0.9], np.float32), "alpha": 0.15},
        ]
        layers = []
        for T in tris:
            mask = tri_coverage(T["uv"], H, W, cfg2.coverage_k)
            layers.append({"color": T["color"], "alpha": T["alpha"], "depth": T["depth"], "mask": mask})
        comp = oit_weighted_blended(layers, cfg2.oit_beta, H, W, comp)

        fused = robin_harmonic(comp, gb.Z, lam=cfg2.fusion_lambda1)
        linear_rgb, exposure = exposure_normalize(fused, cfg2.target_gray, cfg2.exposure_min, cfg2.exposure_max, mode=cfg2.exposure_mode)
        srgb = tone_map_aces(linear_rgb, cfg2.aces_gain)

        if cfg2.dither:
            add_dither_inplace(srgb, cfg2.dither_amp, tile=64)

        os.makedirs(os.path.dirname(prefix) if os.path.dirname(prefix) else ".", exist_ok=True)
        iio.imwrite(prefix + "_srgb.png",  (saturate(srgb)*255).astype(np.uint8))
        iio.imwrite(prefix + "_linear.png", (saturate(linear_rgb)*255).astype(np.uint8))
        iio.imwrite(prefix + "_ao.png",    (saturate(AO)*255).astype(np.uint8))
        iio.imwrite(prefix + "_shadow.png",(saturate(T_soft)*255).astype(np.uint8))
        with open(prefix + "_alloc.json","w") as f:
            json.dump(dict(version=self.cfg.version,
                           exposure_mode=cfg2.exposure_mode,
                           exposure_scale=float(exposure)), f, indent=2)

        return dict(
            srgb=prefix+"_srgb.png",
            linear=prefix+"_linear.png",
            ao=prefix+"_ao.png",
            shadow=prefix+"_shadow.png",
            alloc=prefix+"_alloc.json",
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--prefix", type=str, default="/mnt/data/ctultra_v1/fast_spheres")
    ap.add_argument("--no-alloc", action="store_true")
    ap.add_argument("--no-dither", action="store_true")
    ap.add_argument("--exp", type=str, default="p50", choices=["mean","p50"])
    args = ap.parse_args()

    cfg = RenderConfig(size=args.size,
                       use_allocator=(not args.no_alloc),
                       dither=(not args.no_dither),
                       exposure_mode=args.exp)
    engine = RenderEngine(cfg)
    outputs = engine.render(cfg.size, args.prefix)
    print(json.dumps(outputs, indent=2))

if __name__ == "__main__":
    os.makedirs("/mnt/data/ctultra_v1", exist_ok=True)
    sys.argv = ["ctultra_v12f.py", "--size", "640", "--prefix", "/mnt/data/ctultra_v1/fast_spheres", "--exp", "p50"]
    main()
