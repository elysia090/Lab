# -*- coding: utf-8 -*-
"""
Sera-pro-core (refactored)
Single-file, O(1) online learner with spherical MoE + adaptive prior gating,
optionally fused end-to-end with SeraCore features (PRF, GateClamp, Overlays).

Dependencies: numpy, math
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, Any, List

import numpy as _np
import math as _math

_EPS = 1e-12

# ======================================================================
# Utilities
# ======================================================================

def _clamp(x: float, a: float, b: float) -> float:
    return a if x < a else (b if x > b else x)

def _softmax(x: _np.ndarray, axis: int = -1) -> _np.ndarray:
    x = x - _np.max(x, axis=axis, keepdims=True)
    e = _np.exp(x)
    return e / (_np.sum(e, axis=axis, keepdims=True) + 1e-12)

def _entropy(p: _np.ndarray) -> float:
    p = _np.clip(p, 1e-12, 1.0)
    return float(-_np.sum(p * _np.log(p)))

def _margin_from_probs(p: _np.ndarray) -> float:
    idx = _np.argsort(-p)
    a = float(p[idx[0]])
    b = float(p[idx[1]]) if p.size >= 2 else 0.0
    return a - b

# ======================================================================
# Real-valued spherical harmonics on S^2
# ======================================================================

def _K(l: int, m: int) -> float:
    from math import factorial, sqrt, pi
    return sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m))

def _associated_legendre_all(lmax: int, x: float):
    # returns P[l][m] = P_l^m(x), for 0<=l<=lmax, 0<=m<=l
    P = [None] * (lmax + 1)
    P[0] = _np.array([1.0])
    if lmax == 0:
        return P
    x2 = max(0.0, 1.0 - x * x)
    pmm = 1.0
    for m in range(1, lmax + 1):
        pmm *= -(2 * m - 1) * _math.sqrt(x2)
        arr = _np.zeros(m + 1)
        arr[m] = pmm
        P[m] = arr
    # m=0 ladder
    if P[1] is None:
        P[1] = _np.zeros(2)
    P[1][0] = x
    for l in range(2, lmax + 1):
        a = (2 * l - 1) / l
        b = (l - 1) / l
        P[l][0] = a * x * P[l - 1][0] - b * P[l - 2][0]
    # m>0 ladder
    for m in range(1, lmax + 1):
        for l in range(m + 2, lmax + 1):
            P[l][m] = ((2 * l - 1) * x * P[l - 1][m] - (l + m - 1) * P[l - 2][m]) / (l - m)
    return P

def sh_real_vec(L: int, theta: float, phi: float) -> _np.ndarray:
    x = _math.cos(theta)
    P = _associated_legendre_all(L, x)
    out = _np.zeros((L + 1) * (L + 1), dtype=_np.float64)
    idx = 0
    for l in range(0, L + 1):
        for m in range(-l, 0):
            am = -m
            out[idx] = _math.sqrt(2.0) * _K(l, am) * P[l][am] * _math.sin(am * phi)
            idx += 1
        out[idx] = _K(l, 0) * P[l][0]
        idx += 1
        for m in range(1, l + 1):
            out[idx] = _math.sqrt(2.0) * _K(l, m) * P[l][m] * _math.cos(m * phi)
            idx += 1
    return out

def _dir_to_theta_phi(n: _np.ndarray) -> Tuple[float, float]:
    n = _np.asarray(n, dtype=_np.float64)
    n = n / (_np.linalg.norm(n) + 1e-12)
    x, y, z = float(n[0]), float(n[1]), float(n[2])
    z = _clamp(z, -1.0, 1.0)
    theta = _math.acos(z)
    phi = _math.atan2(y, x)
    if phi < 0:
        phi += 2 * _math.pi
    return theta, phi

def expert_to_unitvec(idx: int) -> Tuple[_np.ndarray, Dict[str, int]]:
    # quasi-uniform spiral (golden angle)
    k = (idx + 0.5)
    z = 1.0 - 2.0 * ((k % 9973) / 9973.0)
    r = _math.sqrt(max(0.0, 1.0 - z * z))
    phi = (_math.pi * (1.0 + 5.0 ** 0.5)) * k
    x = r * _math.cos(phi)
    y = r * _math.sin(phi)
    n = _np.array([x, y, z], dtype=_np.float64)
    n = n / (_np.linalg.norm(n) + 1e-12)
    return n, {"idx": int(idx)}

# ======================================================================
# Prior field on S^2 (real SH coefficients)
# ======================================================================

class SHField:
    def __init__(self, L: int = 8, t_heat: float = 0.01, beta_mem: float = 0.9,
                 n_theta: int = 32, n_phi: int = 64, eta: float = 0.2):
        self.L = int(L)
        self.t_heat = float(t_heat)
        self.beta_mem = float(beta_mem)
        self.n_theta = int(n_theta)
        self.n_phi = int(n_phi)
        self.eta = float(eta)
        D = (self.L + 1) * (self.L + 1)
        self.c_st = _np.zeros(D)
        self.c_lt = _np.zeros(D)
        self._heat = _np.array([_math.exp(-l * (l + 1) * self.t_heat) for l in range(self.L + 1)])
        self._grid_cache = None  # lazy cache

    def _apply_heat(self, vec: _np.ndarray) -> _np.ndarray:
        out = vec.copy()
        for l in range(self.L + 1):
            out[l * l:(l + 1) * (l + 1)] *= self._heat[l]
        return out

    def update_from_topk(self, dirs: Iterable[_np.ndarray], probs: Iterable[float], to_memory: bool = True) -> None:
        D = (self.L + 1) * (self.L + 1)
        u = _np.zeros(D)
        s = 0.0
        for p, n in zip(probs, dirs):
            th, ph = _dir_to_theta_phi(n)
            u += float(p) * sh_real_vec(self.L, th, ph)
            s += float(p)
        if s <= 0:
            return
        u /= s
        u = self._apply_heat(u)
        self.c_st = (1.0 - self.eta) * self.c_st + self.eta * u
        if to_memory:
            self.c_lt = self.beta_mem * self.c_lt + (1.0 - self.beta_mem) * self.c_st

    def _ensure_grid_cache(self) -> _np.ndarray:
        if self._grid_cache is None:
            th = _np.linspace(0.0, _math.pi, self.n_theta)
            ph = _np.linspace(0.0, 2 * _math.pi, self.n_phi, endpoint=False)
            Ys = _np.zeros((self.n_theta, self.n_phi, (self.L + 1) * (self.L + 1)))
            for i, t in enumerate(th):
                for j, p in enumerate(ph):
                    Ys[i, j, :] = sh_real_vec(self.L, t, p)
            self._grid_cache = Ys
        return self._grid_cache

    def render(self, percentile_norm: bool = True) -> _np.ndarray:
        Ys = self._ensure_grid_cache()
        c = 0.5 * self.c_lt + 0.5 * self.c_st
        img = _np.tensordot(Ys, c, axes=([2], [0]))
        if percentile_norm:
            a = _np.percentile(img, 1.0)
            b = _np.percentile(img, 99.0)
            img = (img - a) / (b - a + 1e-12)
        return img

# ======================================================================
# Configs
# ======================================================================

@dataclass
class LearningRates:
    eta_B: float = 1e-3
    eta_tau: float = 1e-3
    eta_lambda: float = 1e-3
    eta_lin: float = 1e-3

@dataclass
class Thresholds:
    tau_H: float = 3.0
    tau_margin: float = 0.08
    tau_margin_hi: float = 0.15
    tau_sharp: float = 0.10
    tau_H_min: float = 1.2
    tau_H_max: float = 3.5
    tau_margin_min: float = 0.05
    tau_margin_max: float = 0.25
    tau_sharp_min: float = 0.02
    tau_sharp_max: float = 0.20

@dataclass
class PriorSettings:
    L: int = 8
    t_heat: float = 0.01
    beta_mem: float = 0.9
    alpha_assim: float = 0.06
    assim_temp: float = 1.5
    q_temp_min: float = 0.6
    q_temp_max: float = 1.2
    sharp_target: float = 0.10

@dataclass
class ModelDims:
    d_in: int
    d_out: int
    K: int

@dataclass
class GateControllerCfg:
    gate_target: Optional[float] = None
    window: int = 30
    step: float = 0.05

@dataclass
class ClipBounds:
    B_max: float = 5.0
    tau_logit_min: float = 0.3
    lambda_max: float = 2.0

# ======================================================================
# Minimal Sera-core building blocks (for glue)
# ======================================================================

def _safe_sigmoid(x: float) -> float:
    if x >= 0:
        z = _math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = _math.exp(x)
        return z / (1.0 + z)

class PRFAttn:
    def __init__(self, d_q: int, R, beta_floor: float = 1e-3, clip_q: float = 5.0, ema: float = 0.98, seed: int = 0):
        self.d_q = int(d_q)
        self.beta_floor = float(beta_floor)
        self.clip_q = float(clip_q)
        self.ema = float(ema)
        self.R = _np.asarray(R, dtype=float)
        self.mean = _np.zeros(d_q)
        self.var = _np.ones(d_q)
        self.eps = 1e-6
        self.rng = _np.random.default_rng(seed)

    def _phi(self, q: _np.ndarray) -> _np.ndarray:
        q = _np.asarray(q, dtype=float).reshape(-1)
        self.mean = self.ema * self.mean + (1 - self.ema) * q
        self.var = self.ema * self.var + (1 - self.ema) * (q - self.mean) ** 2
        std = _np.sqrt(self.var + self.eps)
        z = (q - self.mean) / std
        z = _np.clip(z, -self.clip_q, self.clip_q)
        return z

    def ingest(self, q, y):
        return None

    def _den_num(self, phi_w: _np.ndarray, overlays: Optional[Dict[str, Any]]) -> Tuple[float, float]:
        den = float(phi_w @ phi_w) + self.beta_floor
        num = float(phi_w @ self.R)
        if overlays:
            if "A" in overlays:
                for a, u in overlays["A"]:
                    kf = self._phi(a)
                    num += float(phi_w @ kf) * float(u)
            if "B" in overlays:
                for a, beta in overlays["B"]:
                    beta = max(0.0, float(beta))
                    kf = self._phi(a)
                    den += float((phi_w @ kf) * beta)
        return num, den

    def query(self, q: _np.ndarray, overlays: Optional[Dict[str, Any]] = None) -> float:
        phi = self._phi(q)
        num, den = self._den_num(phi, overlays)
        return float(num / max(den, self.beta_floor))

class GateClamp:
    def __init__(self, tau_H: float, tau_margin: float, tau_sharp: float, horizon: int = 8, smooth: float = 0.2):
        from collections import deque
        self.tau_H = float(tau_H)
        self.tau_margin = float(tau_margin)
        self.tau_sharp = float(tau_sharp)
        self.horizon = int(horizon)
        self.smooth = float(smooth)
        self.H_hist = deque(maxlen=max(4, horizon))
        self.M_hist = deque(maxlen=max(4, horizon))
        self.S_hist = deque(maxlen=max(4, horizon))
        self.state = 0.0

    def forward(self, H: float, margin: float, sharp: float) -> float:
        self.H_hist.append(H)
        self.M_hist.append(margin)
        self.S_hist.append(sharp)
        pH = _np.mean(self.H_hist)
        pM = _np.mean(self.M_hist)
        pS = _np.mean(self.S_hist)
        raw = 1.0 if ((pH >= self.tau_H) and (pM <= self.tau_margin) and (pS >= self.tau_sharp)) else 0.0
        self.state = (1.0 - self.smooth) * self.state + self.smooth * raw
        return float(self.state)

class SOSMemory:
    def __init__(self, a1: float = 0.0, a2: float = 0.0, b0: float = 1.0, b1: float = 0.0, b2: float = 0.0, sections: int = 1):
        self.a1 = float(a1); self.a2 = float(a2)
        self.b0 = float(b0); self.b1 = float(b1); self.b2 = float(b2)
        self.sections = int(max(1, sections))
        self.x1 = _np.zeros(self.sections); self.x2 = _np.zeros(self.sections)
        self.y1 = _np.zeros(self.sections); self.y2 = _np.zeros(self.sections)

    def step(self, x: float) -> float:
        v = float(x)
        for i in range(self.sections):
            y = self.b0 * v + self.b1 * self.x1[i] + self.b2 * self.x2[i] - self.a1 * self.y1[i] - self.a2 * self.y2[i]
            self.x2[i], self.x1[i] = self.x1[i], v
            self.y2[i], self.y1[i] = self.y1[i], y
            v = y
        return float(v)

class OverlayManager:
    def __init__(self):
        self.A: List[Tuple[_np.ndarray, float]] = []
        self.B: List[Tuple[_np.ndarray, float]] = []
        self.enabled: bool = True
    def set_anchors(self, A=None, B=None):
        self.A = list(A) if A else []
        self.B = list(B) if B else []
    def enable(self, flag: bool = True):
        self.enabled = bool(flag)
    def build_overlays(self) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        out = {}
        if self.A: out["A"] = list(self.A)
        if self.B: out["B"] = list(self.B)
        return out if out else None

# ======================================================================
# SeraProCore (refactored)
# ======================================================================

class SeraProCore:
    def __init__(self, d_in: int, d_out: int, K: int,
                 L: int = 8,
                 lr: LearningRates = LearningRates(),
                 thr: Thresholds = Thresholds(),
                 prior: PriorSettings = PriorSettings(),
                 gatectl: GateControllerCfg = GateControllerCfg(),
                 clips: ClipBounds = ClipBounds(),
                 seed: int = 0):
        # dims
        self.d_in = int(d_in); self.d_out = int(d_out)
        self.K = int(K); self.L = int(L)
        self.D_sh = (self.L + 1) * (self.L + 1)

        # hyper
        self.lr = lr
        self.thr = thr
        self.prior = prior
        self.gatectl = gatectl
        self.clips = clips

        # rng
        self.rng = _np.random.default_rng(seed)

        # SH prior + geometry
        self.qms = SHField(L=self.L, t_heat=self.prior.t_heat, beta_mem=self.prior.beta_mem)
        self.dirs = _np.stack([expert_to_unitvec(i)[0] for i in range(self.K)], axis=0)
        self.Y_all = _np.stack([sh_real_vec(self.L, *_dir_to_theta_phi(n)) for n in self.dirs], axis=0)

        # parameters
        self.B = self.rng.standard_normal((self.D_sh, self.d_in)) / _math.sqrt(self.D_sh)
        self.gB_row = _np.zeros(self.D_sh); self.gB_col = _np.zeros(self.d_in)

        self.linW = _np.zeros((self.d_in, self.d_out))
        self.g_lin = _np.zeros(self.d_in)

        self.tau_logit = 1.0
        self.lambda_prior = 0.25

        # gate controller state
        self._gate_hist: List[int] = []
        self._Hs_hist: "deque" = __import__("collections").deque(maxlen=60)
        self.tau_H_hi = 2.6  # adaptive occlusion threshold

        # prior temperature
        self.q_temp = 1.0

        # glue (lazy)
        self._glue_enabled = False
        self._glue_ready = False

    # ---------------- Internal helpers ----------------

    def _logits_from_B(self, x: _np.ndarray) -> _np.ndarray:
        z = self.B @ x.reshape(-1)
        l = (self.Y_all @ z) / max(self.tau_logit, 1e-6)
        return l

    def _prior_q(self) -> _np.ndarray:
        img = self.qms.render(percentile_norm=True)
        n_theta, n_phi = img.shape
        vals = []
        for n in self.dirs:
            th, ph = _dir_to_theta_phi(n)
            i_th = int(round(th / _math.pi * (n_theta - 1)))
            j_ph = int(round(ph / (2 * _math.pi) * (n_phi - 1)))
            vals.append(max(img[i_th, j_ph], 1e-6))
        v = _np.asarray(vals, dtype=_np.float64)
        q = v / (v.sum() + 1e-12)
        # temperature
        t = self.q_temp
        if abs(t - 1.0) > 1e-6:
            q = q ** (1.0 / max(t, 1e-6))
            q = q / (q.sum() + 1e-12)
        return q

    @staticmethod
    def _sharpness(q: _np.ndarray) -> float:
        q_sorted = _np.sort(q)
        return float(q_sorted[-1] - _np.median(q_sorted))

    def _assimilate_from_logits(self, l: _np.ndarray) -> None:
        z = l / max(self.prior.assim_temp, 1e-6)
        z = z - _np.max(z)
        p = _np.exp(z); p = p / (p.sum() + 1e-12)
        idx = _np.argsort(-p)[:min(16, self.K)]
        p_top = p[idx]; p_top = p_top / (p_top.sum() + 1e-12)
        u = _np.zeros(self.D_sh)
        for i, pi in zip(idx.tolist(), p_top.tolist()):
            u += float(pi) * self.Y_all[i]
        # heat
        for ldeg in range(self.L + 1):
            u[ldeg * ldeg:(ldeg + 1) * (ldeg + 1)] *= self.qms._heat[ldeg]
        # EMA
        self.qms.c_st = (1.0 - self.prior.alpha_assim) * self.qms.c_st + self.prior.alpha_assim * u
        self.qms.c_lt = self.qms.beta_mem * self.qms.c_lt + (1.0 - self.qms.beta_mem) * self.qms.c_st

    def _update_B(self, x: _np.ndarray, s: _np.ndarray, p_t: _np.ndarray) -> None:
        topM = min(8, self.K)
        idx = _np.argsort(-s)[:topM]
        delta = s[idx] - p_t[idx]                # (topM,)
        u_B = self.Y_all[idx].T @ delta          # (D_sh,)
        # AdaGrad-like per-row/col
        self.gB_row = _np.minimum(self.gB_row + (u_B ** 2), 1e4)
        self.gB_col = _np.minimum(self.gB_col + (x.reshape(-1) ** 2), 1e4)
        denom = _np.sqrt(_np.outer(self.gB_row, self.gB_col)) + 1e-6
        gradB = _np.outer(u_B, x.reshape(-1)) / max(self.tau_logit, 1e-6)
        self.B -= self.lr.eta_B * (gradB / denom)
        # clip
        nrm = _np.linalg.norm(self.B)
        if nrm > self.clips.B_max:
            self.B *= (self.clips.B_max / (nrm + 1e-12))

    def _update_tau_logit(self, s: _np.ndarray, p_t: _np.ndarray, l: _np.ndarray) -> None:
        delta = s - p_t
        grad = -(1.0 / max(self.tau_logit, 1e-6)) * float(_np.dot(delta, l))
        if not hasattr(self, "g_tau"):
            self.g_tau = 0.0
        self.g_tau = min(self.g_tau + grad * grad, 1e6)
        self.tau_logit = max(self.clips.tau_logit_min, self.tau_logit - self.lr.eta_tau * grad / (_math.sqrt(self.g_tau) + 1e-6))

    def _update_lambda(self, z: _np.ndarray, p_t: _np.ndarray, logq: _np.ndarray, gate_on: bool) -> None:
        if not gate_on:
            return
        s_post = _softmax(z)
        idx = _np.argsort(-s_post)[:min(8, self.K)]
        grad = float(_np.dot(s_post[idx] - p_t[idx], logq[idx]))
        if not hasattr(self, "g_lambda"):
            self.g_lambda = 0.0
        self.g_lambda = min(self.g_lambda + grad * grad, 1e6)
        self.lambda_prior = _clamp(self.lambda_prior - self.lr.eta_lambda * grad / (_math.sqrt(self.g_lambda) + 1e-6), 0.0, self.clips.lambda_max)

    def _update_linW(self, x: _np.ndarray, y_att: _np.ndarray, y_lin: _np.ndarray) -> _np.ndarray:
        e = (y_att - y_lin)
        self.g_lin = _np.minimum(self.g_lin + (x.reshape(-1) ** 2) * float(_np.mean(e * e)), 1e6)
        self.linW += self.lr.eta_lin * _np.outer(x.reshape(-1), e) / (_np.sqrt(self.g_lin)[:, None] + 1e-6)
        return self.linW

    def _gate_control_step(self) -> None:
        if (self.gatectl.gate_target is None) or (len(self._gate_hist) < self.gatectl.window):
            return
        rate = sum(self._gate_hist[-self.gatectl.window:]) / float(self.gatectl.window)
        if rate < self.gatectl.gate_target - 0.02:
            self.thr.tau_H = max(self.thr.tau_H_min, self.thr.tau_H - self.gatectl.step)
            self.thr.tau_margin = min(self.thr.tau_margin_max, self.thr.tau_margin + 0.01)
            self.thr.tau_sharp = max(self.thr.tau_sharp_min, self.thr.tau_sharp - 0.01)
        elif rate > self.gatectl.gate_target + 0.02:
            self.thr.tau_H = min(self.thr.tau_H_max, self.thr.tau_H + self.gatectl.step)
            self.thr.tau_margin = max(self.thr.tau_margin_min, self.thr.tau_margin - 0.01)
            self.thr.tau_sharp = min(self.thr.tau_sharp_max, self.thr.tau_sharp + 0.01)

    # ---------------- Glue management ----------------

    def enable_seracore_glue(self, flag: bool = True) -> None:
        self._glue_enabled = bool(flag)
        if not flag:
            return
        if not self._glue_ready:
            self._prf = PRFAttn(d_q=self.D_sh, R=_np.ones(self.D_sh))
            self._gate2 = GateClamp(self.thr.tau_H, self.thr.tau_margin, self.thr.tau_sharp, horizon=8, smooth=0.2)
            self._sos = SOSMemory(a1=-0.1, a2=0.0, b0=0.2, b1=0.2, b2=0.0, sections=2)
            self._ovm = OverlayManager(); self._ovm.enable(True)
            self._glue_ready = True

    def set_overlays(self, A=None, B=None, enabled: bool = True) -> None:
        if getattr(self, "_glue_enabled", False) and hasattr(self, "_ovm"):
            self._ovm.set_anchors(A=A, B=B); self._ovm.enable(enabled)

    def snapshot_glue(self) -> Dict[str, Any]:
        if not getattr(self, "_glue_enabled", False):
            return {}
        return {
            "prf": {"mean": self._prf.mean.copy(), "var": self._prf.var.copy()},
            "gate2": {"state": getattr(self._gate2, "state", 0.0),
                      "tau_H": self._gate2.tau_H, "tau_margin": self._gate2.tau_margin, "tau_sharp": self._gate2.tau_sharp},
        }

    def restore_glue(self, st: Dict[str, Any]) -> None:
        if not st:
            return
        self.enable_seracore_glue(True)
        if "prf" in st:
            self._prf.mean[:] = st["prf"]["mean"]; self._prf.var[:] = st["prf"]["var"]
        if "gate2" in st:
            self._gate2.state = float(st["gate2"].get("state", 0.0))
            self._gate2.tau_H = float(st["gate2"]["tau_H"]); self._gate2.tau_margin = float(st["gate2"]["tau_margin"]); self._gate2.tau_sharp = float(st["gate2"]["tau_sharp"])

    # ---------------- Forward (with optional glue) ----------------

    def forward(self, x: _np.ndarray, use_prior: bool = True) -> Dict[str, Any]:
        x = _np.asarray(x, dtype=_np.float64).reshape(-1)

        # gate-rate controller
        self._gate_control_step()

        # surrogate head
        y_att = x @ self.linW

        # logits & base probs
        l = self._logits_from_B(x)
        s = _softmax(l)
        Hs = _entropy(s)
        margin = _margin_from_probs(s)

        # prior q + stats
        q = self._prior_q()
        sharp = self._sharpness(q)
        logq = _np.log(_np.clip(q, 1e-9, 1.0))

        # adaptive hard-occlusion
        self._Hs_hist.append(Hs)
        if (self.gatectl.gate_target is not None) and (len(self._Hs_hist) >= 10):
            arr = _np.array(list(self._Hs_hist), dtype=_np.float64)
            qtl = _np.quantile(arr, max(0.0, min(1.0, 1.0 - self.gatectl.gate_target)))
            self.tau_H_hi = float(qtl)

        hard_occ = (Hs >= self.tau_H_hi)
        gate_on = bool(use_prior and (((Hs >= self.thr.tau_H) and (margin <= self.thr.tau_margin) and (sharp >= self.thr.tau_sharp)) or hard_occ))

        # glue modulation for lambda (optional)
        if self._glue_enabled:
            gate_lvl = self._gate2.forward(Hs, margin, sharp)
            zproj = self.B @ x
            ov = self._ovm.build_overlays() if hasattr(self, "_ovm") else None
            prf_score = self._prf.query(zproj, overlays=ov)
            sig = _safe_sigmoid(prf_score)
            lambda_eff = self.lambda_prior * (0.75 + 0.25 * sig) * (0.5 + 0.5 * gate_lvl)
        else:
            gate_lvl = 0.0
            prf_score = 0.0
            lambda_eff = self.lambda_prior

        # posterior
        z = l + (lambda_eff * logq) if gate_on else l
        s_post = _softmax(z)

        # assimilation
        self._assimilate_from_logits(l)

        # teacher target (stable O(1))
        teacher_defined = False
        if margin >= self.thr.tau_margin_hi:
            p_t = s; teacher_defined = True
        elif gate_on and (sharp >= self.thr.tau_sharp):
            p_t = q; teacher_defined = True

        if teacher_defined:
            self._update_B(x, s, p_t)
            self._update_tau_logit(s, p_t, l)
            self._update_lambda(z, p_t, logq, gate_on)

        # SH field update from posterior
        k = min(max(4, 8 // 2), self.K)
        idx = _np.argsort(-s_post)[:k]
        p_top = s_post[idx]; p_top = p_top / (p_top.sum() + 1e-12)
        dirs = [self.dirs[i] for i in idx.tolist()]
        self.qms.update_from_topk(dirs, p_top.tolist(), to_memory=True)

        # CCR-like light residual
        y_lin = x @ self.linW
        y = y_att + 0.25 * _np.clip(y_att - y_lin, -1.5, 1.5)

        # update linear head
        self._update_linW(x, y_att, y_lin)

        # q_temp feedback
        if sharp < self.prior.sharp_target:
            self.q_temp = max(self.prior.q_temp_min, self.q_temp - 0.02)
        elif sharp > self.prior.sharp_target + 0.05:
            self.q_temp = min(self.prior.q_temp_max, self.q_temp + 0.02)

        # push gate flag
        self._gate_hist.append(1 if gate_on else 0)

        return {
            "y": y, "y_att": y_att, "y_lin": y_lin,
            "logits": l, "probs": s, "probs_post": s_post,
            "prior_q": q, "gate_on": gate_on,
            "entropy": Hs, "margin": margin, "sharp": sharp,
            "tau_logit": self.tau_logit, "lambda_prior": self.lambda_prior,
            "lambda_eff": float(lambda_eff), "prf_score": float(prf_score), "gate_level2": float(gate_lvl),
            "tau_H": self.thr.tau_H, "tau_margin": self.thr.tau_margin, "tau_sharp": self.thr.tau_sharp,
            "tau_H_hi": self.tau_H_hi, "q_temp": self.q_temp
        }

# ======================================================================
# Back-compat facade (optional)
# ======================================================================

@dataclass
class SeraConfig:
    d_in: int = 24
    d_out: int = 8
    K: int = 12
    L: int = 3
    gate_target: float = 0.25
    seed: int = 0

class SeraCore:
    """Small wrapper to keep legacy Sera-core style API."""
    def __init__(self, cfg: SeraConfig):
        self.cfg = cfg
        lr = LearningRates(eta_B=4e-3, eta_tau=1e-3, eta_lambda=2e-3, eta_lin=1e-3)
        thr = Thresholds(tau_H=2.6, tau_margin=0.10, tau_margin_hi=0.18, tau_sharp=0.10)
        prior = PriorSettings(L=cfg.L, t_heat=0.01, beta_mem=0.9, alpha_assim=0.10)
        gatectl = GateControllerCfg(gate_target=cfg.gate_target, window=20, step=0.06)
        self.kernel = SeraProCore(d_in=cfg.d_in, d_out=cfg.d_out, K=cfg.K, L=cfg.L,
                                  lr=lr, thr=thr, prior=prior, gatectl=gatectl, seed=cfg.seed)
        self.kernel.enable_seracore_glue(True)
        self.ov = OverlayManager(); self.ov.enable(True)

    def step_ingest(self, x: _np.ndarray) -> Dict[str, Any]:
        z = self.kernel.B @ _np.asarray(x, dtype=float).reshape(-1)
        # In original design: PRF ingests stats; that is implicit here via query()
        return {"z": z}

    def fuse_predict(self, x: Optional[_np.ndarray] = None, overlays: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if x is None:
            x = _np.zeros(self.cfg.d_in, dtype=float)
        if overlays is not None:
            # override internal overlays
            self.kernel.set_overlays(enabled=True, A=overlays.get("A"), B=overlays.get("B"))
        return self.kernel.forward(x, use_prior=True)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "B": self.kernel.B.copy(),
            "linW": self.kernel.linW.copy(),
            "c_st": self.kernel.qms.c_st.copy(),
            "c_lt": self.kernel.qms.c_lt.copy(),
            "tau_logit": self.kernel.tau_logit,
            "lambda_prior": self.kernel.lambda_prior,
            "glue": self.kernel.snapshot_glue(),
        }

    def restore(self, state: Dict[str, Any]) -> bool:
        self.kernel.B[:] = state["B"]; self.kernel.linW[:] = state["linW"]
        self.kernel.qms.c_st[:] = state["c_st"]; self.kernel.qms.c_lt[:] = state["c_lt"]
        self.kernel.tau_logit = float(state["tau_logit"]); self.kernel.lambda_prior = float(state["lambda_prior"])
        self.kernel.restore_glue(state.get("glue", {}))
        return True

__all__ = [
    "SeraProCore", "SeraConfig", "SeraCore",
    "LearningRates", "Thresholds", "PriorSettings", "GateControllerCfg", "ClipBounds",
    "PRFAttn", "GateClamp", "SOSMemory", "OverlayManager"
]
