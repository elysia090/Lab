
import numpy as np, math, imageio.v2 as iio

EPS = 1e-6
def saturate(x): return np.clip(x, 0.0, 1.0).astype(np.float32)
def luma(rgb):   return (0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]).astype(np.float32)

def ihash2(x, y):
    v = (x * 0x27d4eb2d) ^ (y * 0x165667b1) ^ 0x9e3779b9
    v ^= (v >> 15); v *= 0x85ebca6b; v ^= (v >> 13); v *= 0xc2b2ae35; v ^= (v >> 16)
    return v & 0xffffffff

# FGD LUT (vectorized 32x32 analytic)
def build_fgd_lut(n=32):
    NoV = np.linspace(1e-4, 1.0, n, dtype=np.float32)
    alpha = np.linspace(1e-3, 1.0, n, dtype=np.float32)
    nv, a = np.meshgrid(NoV, alpha, indexing="ij")
    A = 1.0/(1.0+nv) * (1.0 - 0.28*a) + 0.28*a
    B = ((a+1.0)**2)/8.0
    return A.astype(np.float32), B.astype(np.float32)
FGD_A, FGD_B = build_fgd_lut()
def fgd_sample(NoV, alpha):
    i = np.minimum((NoV * 31.0).astype(np.int32), 31)
    j = np.minimum((alpha*31.0).astype(np.int32), 31)
    return FGD_A[i,j], FGD_B[i,j]

def aces_tonemap(x):
    a=2.51; b=0.03; c=2.43; d=0.59; e=0.14
    return saturate((x*(a*x+b))/(x*(c*x+d)+e))

def logmean_exposure(luma_img, target_gray=0.20):
    H,W = luma_img.shape
    N  = 256
    seed = 0x1234abcd
    idx = []
    for k in range(N):
        seed = (1664525*seed + 1013904223) & 0xffffffff
        y = (seed >> 16) & 0xffff
        seed = (1664525*seed + 1013904223) & 0xffffffff
        x = (seed >> 16) & 0xffff
        idx.append((y % H, x % W))
    samp = np.array([luma_img[iy,ix] for (iy,ix) in idx], np.float32)
    E = np.exp(np.mean(np.log(np.maximum(samp, 1e-6), dtype=np.float32)))
    return float(target_gray / (E + 1e-6))

def rasterize_spheres(H,W, spheres):
    depth = np.full((H,W), -1e-9, np.float32)
    albedo = np.zeros((H,W,3), np.float32)
    rough  = np.zeros((H,W,1), np.float32)
    metal  = np.zeros((H,W,1), np.float32)
    normal = np.zeros((H,W,3), np.float32)
    mask   = np.zeros((H,W,1), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for s in spheres:
        cx, cy, R = s["cx"], s["cy"], s["r"]
        dx = (xx - cx); dy = (yy - cy)
        rr = dx*dx + dy*dy
        inside = rr <= (R*R)
        z = np.sqrt(np.maximum(0.0, R*R - rr))
        n = np.zeros((H,W,3), np.float32)
        denom = np.maximum(np.sqrt(rr + z*z), 1e-6)
        n[...,0] = dx/denom; n[...,1] = dy/denom; n[...,2] = z/denom
        closer = (z > depth) & inside
        yi, xi = np.where(closer)
        depth[yi,xi] = z[yi,xi]
        normal[yi,xi,:] = n[yi,xi,:]
        albedo[yi,xi,:] = s["albedo"]
        rough[yi,xi,0]  = s["rough"]
        metal[yi,xi,0]  = s["metal"]
        mask[yi,xi,0]   = 1.0
    return depth, normal, albedo, rough, metal, mask

def shift9_gray(img):
    H,W = img.shape
    p = np.pad(img, ((1,1),(1,1)), mode='edge')
    out = np.empty((H,W,9), np.float32)
    out[...,0] = p[0:H,   0:W  ]
    out[...,1] = p[0:H,   1:W+1]
    out[...,2] = p[0:H,   2:W+2]
    out[...,3] = p[1:H+1, 0:W  ]
    out[...,4] = p[1:H+1, 1:W+1]
    out[...,5] = p[1:H+1, 2:W+2]
    out[...,6] = p[2:H+2, 0:W  ]
    out[...,7] = p[2:H+2, 1:W+1]
    out[...,8] = p[2:H+2, 2:W+2]
    return out

def bilinear_sample_gray(img, xs, ys):
    H,W = img.shape
    x0 = np.clip(np.floor(xs).astype(np.int32), 0, W-1)
    y0 = np.clip(np.floor(ys).astype(np.int32), 0, H-1)
    x1 = np.clip(x0 + 1, 0, W-1); y1 = np.clip(y0 + 1, 0, H-1)
    fx = (xs - x0).astype(np.float32); fy = (ys - y0).astype(np.float32)
    I00 = img[y0, x0]; I10 = img[y0, x1]; I01 = img[y1, x0]; I11 = img[y1, x1]
    return (I00*(1-fx)*(1-fy) + I10*fx*(1-fy) + I01*(1-fx)*fy + I11*fx*fy).astype(np.float32)

# Precompute K sample points (unit disk, golden-angle)
def make_disk_samples(K):
    phi = (np.sqrt(5.0)-1.0)/2.0
    k = np.arange(K, dtype=np.float32)
    ang = 2.0*np.pi*((k*phi) % 1.0)
    r   = np.sqrt((k+0.5)/K)
    return (r*np.cos(ang)).astype(np.float32), (r*np.sin(ang)).astype(np.float32)  # (K,), (K,)
def rotate_offsets(dx, dy, ct, st, scale=1.0):
    # dx,dy: (K,), ct,st: [H,W], return [H,W,K]
    X = (ct[...,None]*dx[None,None,:] - st[...,None]*dy[None,None,:]) * scale
    Y = (st[...,None]*dx[None,None,:] + ct[...,None]*dy[None,None,:]) * scale
    return X.astype(np.float32), Y.astype(np.float32)

# BRDF
def ggx_D(NoH, a):
    a2 = a*a
    d = (NoH*NoH)*(a2-1.0)+1.0
    return a2/(math.pi*d*d + 1e-6)
def smith_G1(NoX, a):
    k = (a+1.0); k = (k*k)/8.0
    return NoX/(NoX*(1.0-k)+k + 1e-6)
def fresnel_schlick(u, F0):
    return F0 + (1.0-F0)*((1.0-u)**5)
def direct_lighting(N, V, L, albedo, rough, metal, ao, shadow_T):
    NoL = np.maximum(np.sum(N*L, axis=-1, keepdims=True), 0.0)
    NoV = np.maximum(np.sum(N*V, axis=-1, keepdims=True), 0.0)
    H = (V+L); H /= np.maximum(np.linalg.norm(H, axis=-1, keepdims=True), 1e-6)
    NoH = np.maximum(np.sum(N*H, axis=-1, keepdims=True), 0.0)
    VoH = np.maximum(np.sum(V*H, axis=-1, keepdims=True), 0.0)
    a = np.maximum(rough*rough, 1e-3)
    D = ggx_D(NoH, a); G = smith_G1(NoV, a)*smith_G1(NoL, a)
    F0 = 0.04*(1.0-metal) + albedo*metal
    F = fresnel_schlick(VoH, F0)
    spec = (D*G*F)/(4.0*NoL*NoV + 1e-6)
    kd = (1.0 - F) * (1.0 - metal)
    diff = kd * albedo / math.pi
    Li = np.array([1.0,1.0,1.0], np.float32)
    return (diff + spec)*Li*NoL*ao*shadow_T
def ibl_lighting(N, V, albedo, rough, metal, ao):
    NoV = np.maximum(np.sum(N*V, axis=-1, keepdims=True), 0.0)
    a = np.maximum(rough*rough, 1e-3)
    A,B = fgd_sample(NoV[...,0], a[...,0]); A=A[...,None]; B=B[...,None]
    F0 = 0.04*(1.0-metal) + albedo*metal
    F_avg = A*F0 + B
    kd = (1.0 - F_avg) * (1.0 - metal)
    E_diff = np.array([0.55,0.65,0.75], np.float32)
    E_spec = np.array([0.85,0.90,1.00], np.float32)
    return kd*albedo/math.pi*E_diff*ao + F_avg*E_spec*0.5

def normal_variance_3x3(normal):
    H,W,_ = normal.shape
    p = np.pad(normal, ((1,1),(1,1),(0,0)), mode='edge')
    m = np.zeros_like(normal)
    for dy in range(3):
        for dx in range(3):
            m += p[dy:dy+H, dx:dx+W, :]
    m /= 9.0
    var = np.mean((normal - m)**2, axis=-1, keepdims=True)
    return var.astype(np.float32)
def apply_toksvig(rough, normal, strength=0.9):
    var = normal_variance_3x3(normal)
    a2  = np.maximum(rough*rough, 1e-4)
    a2p = a2 + strength * var
    return np.sqrt(a2p).astype(np.float32)

def contact_ao(depth, radius=0.9, K=8, gain=1.1, base_dx=None, base_dy=None, ct=None, st=None):
    rx, ry = rotate_offsets(base_dx, base_dy, ct, st, scale=radius)
    H,W = depth.shape
    ix, iy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    X = ix[...,None] + rx; Y = iy[...,None] + ry
    zc = depth[...,None]
    zn = bilinear_sample_gray(depth, X, Y)
    acc = np.mean(np.maximum(zn - zc, 0.0), axis=-1, keepdims=True)
    return saturate(1.0 - gain*acc)

def pcf_shadow(depth, normal, light_dir, K=10, phase_bias=0.0, t_min=0.08, base_dx=None, base_dy=None):
    H,W = depth.shape
    ix, iy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    base = ((ihash2(ix.astype(np.int32), iy.astype(np.int32)) & 0xffff) / 65536.0).astype(np.float32)
    phase = (base + phase_bias) % 1.0
    ang = 2.0*np.pi*phase; ct = np.cos(ang).astype(np.float32); st = np.sin(ang).astype(np.float32)
    rx1, ry1 = rotate_offsets(base_dx, base_dy, ct, st, scale=1.6)
    rx2, ry2 = rotate_offsets(base_dx, base_dy, ct*0.73- st*0.68, st*0.73+ ct*0.68, scale=3.3)  # decorrelated rotation
    X = ix[...,None]; Y = iy[...,None]; zc = depth[...,None]
    z1 = bilinear_sample_gray(depth, X+rx1, Y+ry1)
    z2 = bilinear_sample_gray(depth, X+rx2, Y+ry2)
    d9 = shift9_gray(depth)
    dzdx = (d9[...,5]-d9[...,3])*0.5; dzdy = (d9[...,7]-d9[...,1])*0.5
    slope = (np.abs(dzdx)+np.abs(dzdy))[...,None]
    bias = 0.0008 + 0.03*slope
    T1 = 1.0 - np.mean(((zc-bias) > z1).astype(np.float32), axis=-1, keepdims=True)
    T2 = 1.0 - np.mean(((zc-bias) > z2).astype(np.float32), axis=-1, keepdims=True)
    return np.maximum(0.6*T1 + 0.4*T2, t_min)

def ssao(depth, normal, radius=3.0, power=1.05, K=10, phase_bias=0.0, base_dx=None, base_dy=None):
    H,W = depth.shape
    ix, iy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    base = (((ihash2(ix.astype(np.int32), iy.astype(np.int32))>>8)&0xffff)/65536.0).astype(np.float32)
    phase = (base + phase_bias) % 1.0
    ang = 2.0*np.pi*phase; ct = np.cos(ang).astype(np.float32); st = np.sin(ang).astype(np.float32)
    rx, ry = rotate_offsets(base_dx, base_dy, ct, st, scale=radius)
    X = ix[...,None]; Y = iy[...,None]; zc = depth[...,None]
    zn = bilinear_sample_gray(depth, X+rx, Y+ry)
    occ = np.maximum(zn - zc, 0.0)
    ao  = 1.0 - np.mean(saturate(occ*4.0), axis=-1, keepdims=True)
    Nz  = saturate(normal[...,2:3])
    ao  = saturate(ao*(0.5 + 0.5*Nz))
    ao *= contact_ao(depth, radius=0.9, K=K, gain=1.1, base_dx=base_dx, base_dy=base_dy, ct=ct, st=st)
    return ao**power

def temporal_ema(history, current, alpha=0.12, clamp=1.4):
    Ly = luma(history); Lc = luma(current)
    band = clamp * (np.abs(Ly - Lc) + 1e-4)
    cmin = history - band[...,None]; cmax = history + band[...,None]
    cur  = np.minimum(np.maximum(current, cmin), cmax)
    return (1.0-alpha)*history + alpha*cur

def render_hq(W=768, H=768, K=10, N=10):
    spheres = [
        dict(cx=W*0.28, cy=H*0.35, r=W*0.20, albedo=np.array([0.85,0.86,0.88], np.float32), rough=0.20, metal=0.05),
        dict(cx=W*0.70, cy=H*0.35, r=W*0.19, albedo=np.array([0.82,0.90,0.92], np.float32), rough=0.25, metal=0.00),
        dict(cx=W*0.25, cy=H*0.70, r=W*0.12, albedo=np.array([0.70,0.75,0.68], np.float32), rough=0.45, metal=0.00),
        dict(cx=W*0.50, cy=H*0.65, r=W*0.18, albedo=np.array([0.72,0.78,0.92], np.float32), rough=0.35, metal=0.00),
        dict(cx=W*0.80, cy=H*0.70, r=W*0.13, albedo=np.array([0.92,0.92,0.95], np.float32), rough=0.18, metal=0.00),
    ]
    depth, normal, albedo, rough, metal, mask = rasterize_spheres(H,W,spheres)
    rough_tok = apply_toksvig(rough, normal, strength=0.9)

    V = np.zeros_like(normal); V[...,2] = 1.0
    Ldir = np.array([0.5, 0.6, 0.62], np.float32); Ldir /= np.linalg.norm(Ldir)
    L = np.broadcast_to(Ldir, normal.shape)

    # precompute K-disk
    base_dx, base_dy = make_disk_samples(K)

    # first frame
    ao = ssao(depth, normal, radius=3.0, power=1.05, K=K, phase_bias=0.0, base_dx=base_dx, base_dy=base_dy)
    T  = pcf_shadow(depth, normal, Ldir, K=K, phase_bias=0.0, t_min=0.08, base_dx=base_dx, base_dy=base_dy)
    col = direct_lighting(normal, V, L, albedo, rough_tok, metal, ao, T)
    col += ibl_lighting(normal, V, albedo, rough_tok, metal, ao)
    bg = np.array([0.30,0.38,0.42], np.float32)
    rgb = col*mask + bg*(1.0-mask)
    Ls = luma(col*mask + 1e-6)
    gain = float(np.clip(logmean_exposure(Ls, target_gray=0.20), 0.75, 1.5))
    hist = aces_tonemap(rgb * gain)

    for t in range(1, N):
        ph = (t*0.41) % 1.0
        ao = ssao(depth, normal, radius=3.0, power=1.05, K=K, phase_bias=ph, base_dx=base_dx, base_dy=base_dy)
        T  = pcf_shadow(depth, normal, Ldir, K=K, phase_bias=ph, t_min=0.08, base_dx=base_dx, base_dy=base_dy)
        col = direct_lighting(normal, V, L, albedo, rough_tok, metal, ao, T)
        col += ibl_lighting(normal, V, albedo, rough_tok, metal, ao)
        rgb = aces_tonemap((col*mask + bg*(1.0-mask)) * gain)
        hist = temporal_ema(hist, rgb, alpha=0.12, clamp=1.4)

    return (saturate(hist)*255.0 + 0.5).astype(np.uint8)
