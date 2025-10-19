
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sera-core.py
Plain ASCII, English, calm naming.
v2.1 core: constant-time streaming attention (PRF), optional whitening/clipping/floor,
CCR corrector, SOS memory, deterministic clamp gate, snapshot/restore.
All hot-path loops are over configuration-time constants (r, d_v, L, r_v).
"""

from __future__ import annotations
import math
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np

# ------------------------
# Utilities
# ------------------------

def safe_sigmoid(u: float) -> float:
    if u >= 0.0:
        z = math.exp(-u)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(u)
        return z / (1.0 + z)

def kahan_add_inplace(arr: np.ndarray, comp: np.ndarray, delta: np.ndarray) -> None:
    """
    Kahan/Neumaier compensated addition, in-place on arrays.
    Shapes must match.
    """
    t = delta - comp
    y = arr + t
    comp[...] = (y - arr) - t
    arr[...] = y

# ------------------------
# PRF attention with options
# ------------------------

class PRFAttn:
    """
    Positive random-feature streaming attention with optional query-only whitening,
    clipping, and denominator floor.
    State size: O(r * d_v) for R, plus O(r) for s and moments.
    """
    def __init__(
        self,
        d: int,
        d_v: int = 1,
        r: int = 256,
        tau: float = 1.0,
        gamma: float = 0.985,
        clip_c: Optional[float] = 3.0,
        eps: float = 1e-6,
        beta_floor: float = 1e-3,
        seed: int = 17,
        use_kahan: bool = True,
        r_v: int = 0
    ) -> None:
        self.d = int(d)
        self.d_v = int(d_v)
        self.r = int(r)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.clip_c = clip_c
        self.eps = float(eps)
        self.beta_floor = float(beta_floor)
        self.rng = np.random.default_rng(int(seed))
        self.W = self.rng.normal(0.0, 1.0, size=(self.r, self.d))
        self.R = np.zeros((self.r, self.d_v), dtype=float)
        self.s = np.zeros((self.r,), dtype=float)
        # compensated residuals
        self.use_kahan = bool(use_kahan)
        self.cR = np.zeros_like(self.R)
        self.cs = np.zeros_like(self.s)
        # moments for whitening (query-only whitening uses ingest-side moments)
        self.mu = np.zeros((self.r,), dtype=float)
        self.sig2 = np.ones((self.r,), dtype=float)
        # denominator additive predictable term (if used)
        self.lambda_star = 0.0
        # optional value-space head basis for Type-C overlays
        self.r_v = int(r_v)
        self.H = None
        if self.r_v > 0:
            self.H = self.rng.normal(0.0, 1.0, size=(self.r, self.r_v)) / math.sqrt(self.r)
        self.clip_hits = 0

    def _phi(self, x: np.ndarray) -> np.ndarray:
        proj = (self.W @ x) / math.sqrt(self.tau)
        if self.clip_c is not None:
            pc = np.clip(proj, -self.clip_c, self.clip_c)
            self.clip_hits += int(np.sum(pc != proj))
            proj = pc
        base = -(float(x @ x)) / (2.0 * self.tau)
        return np.exp(proj + base) / math.sqrt(self.r)

    def ingest(self, k: np.ndarray, v: np.ndarray) -> None:
        # decay state
        self.R *= self.gamma
        self.s *= self.gamma
        if self.use_kahan:
            self.cR *= self.gamma
            self.cs *= self.gamma
        # features
        phi = self._phi(k)  # shape (r,)
        # update R and s with compensation
        if self.use_kahan:
            kahan_add_inplace(self.R, self.cR, phi[:, None] * v[None, :])
            kahan_add_inplace(self.s, self.cs, phi)
        else:
            self.R += phi[:, None] * v[None, :]
            self.s += phi
        # update moments for whitening (EW)
        self.mu = self.gamma * self.mu + (1.0 - self.gamma) * phi
        d = phi - self.mu
        self.sig2 = self.gamma * self.sig2 + (1.0 - self.gamma) * (d * d)

    def _den_num(
        self,
        phi: np.ndarray,
        overlays: Optional[Dict[str, List]] = None,
        budgets: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, float]:
        D = 1.0 / np.sqrt(self.sig2 + self.eps)
        phi_w = D * phi
        den = float(phi_w @ self.s) + self.lambda_star
        if den < self.beta_floor:
            den = self.beta_floor
        num = phi_w @ self.R  # shape (d_v,)
        # overlays
        if overlays:
            if "A" in overlays:
                tot = 0.0
                for a, u in overlays["A"]:
                    kf = self._phi(a)
                    num += (phi_w @ kf) * np.asarray(u, dtype=float)
                    tot += float(np.linalg.norm(u, 1))
                if budgets and "A" in budgets:
                    assert tot <= budgets["A"] + 1e-9
            if "B" in overlays:
                tot = 0.0
                for a, beta in overlays["B"]:
                    beta = max(0.0, float(beta))
                    kf = self._phi(a)
                    den += float((phi_w @ kf) * beta)
                    tot += beta
                if budgets and "B" in budgets:
                    assert tot <= budgets["B"] + 1e-9
            if "C" in overlays:
                assert self.H is not None and self.r_v > 0, "Type-C overlays require r_v > 0 and H."
                z = phi_w @ self.H  # (r_v,)
                tot = 0.0
                for core in overlays["C"]:
                    U, V = core["U"], core["V"]  # U: r_v x k, V: d_v x k
                    t = z @ U                      # (k,)
                    delta = V @ t                  # (d_v,)
                    num += delta
                    tot += float(np.linalg.norm(U, 2) * np.linalg.norm(V, 2))
                if budgets and "C" in budgets:
                    assert tot <= budgets["C"] + 1e-9
        return num, den

    def query(
        self,
        q: np.ndarray,
        overlays: Optional[Dict[str, List]] = None,
        budgets: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        phi = self._phi(q)
        num, den = self._den_num(phi, overlays, budgets)
        y = num / max(den, self.beta_floor)
        return y, {"num": num, "den": float(den)}

    # snapshot/restore (deterministic, minimal state)
    def snapshot(self) -> Dict[str, np.ndarray]:
        return {
            "W": self.W.copy(),
            "R": self.R.copy(),
            "s": self.s.copy(),
            "mu": self.mu.copy(),
            "sig2": self.sig2.copy(),
            "cR": self.cR.copy(),
            "cs": self.cs.copy(),
        }

    def restore(self, snap: Dict[str, np.ndarray]) -> None:
        self.W = snap["W"].copy()
        self.R = snap["R"].copy()
        self.s = snap["s"].copy()
        self.mu = snap["mu"].copy()
        self.sig2 = snap["sig2"].copy()
        self.cR = snap["cR"].copy()
        self.cs = snap["cs"].copy()

# ------------------------
# SOS memory (DF-II Transposed)
# ------------------------

class SOSMemory:
    def __init__(self, L: int = 6, seed: int = 123) -> None:
        self.L = int(L)
        rng = np.random.default_rng(int(seed))
        self.b0 = np.zeros(self.L); self.b1 = np.zeros(self.L); self.b2 = np.zeros(self.L)
        self.a1 = np.zeros(self.L); self.a2 = np.zeros(self.L)
        self.z1 = np.zeros(self.L); self.z2 = np.zeros(self.L)
        for s in range(self.L):
            r = 0.85 * rng.uniform(0.5, 1.0); th = rng.uniform(0.0, math.pi)
            self.a1[s] = 2.0 * r * math.cos(th); self.a2[s] = -(r * r)
            rz = 0.3 * rng.uniform(0.5, 1.0); ph = rng.uniform(0.0, math.pi)
            self.b1[s] = 2.0 * rz * math.cos(ph); self.b2[s] = -(rz * rz)
            self.b0[s] = 1.0 - self.b1[s] - self.b2[s]

    def step(self, u: float) -> float:
        y = float(u)
        for s in range(self.L):
            tmp = y
            y = self.b0[s] * tmp + self.z1[s]
            self.z1[s] = self.b1[s] * tmp - self.a1[s] * y + self.z2[s]
            self.z2[s] = self.b2[s] * tmp - self.a2[s] * y
        return y

# ------------------------
# CCR corrector (cycle cover example)
# ------------------------

class CCRCorrector:
    """
    Minimal CCR backbone: cycle cover with finite overlaps.
    Provides a truncated homotopy corrector and an operator-norm energy bound.
    """
    def __init__(self, V: int = 3, rho: float = 0.2, m: int = 3) -> None:
        self.V = int(V); self.m = int(m)
        E = self.V
        B = np.zeros((E, self.V))
        for e in range(E):
            i = e; j = (e + 1) % self.V
            B[e, i] = +1.0; B[e, j] = -1.0
        self.B = B
        BBt = B @ B.T
        self.h = B.T @ np.linalg.pinv(BBt)  # V x E
        self.R = float(rho) * B
        self.A = self.R @ self.h            # E x E
        self.gamma = float(np.linalg.norm(self.A, 2))
        assert self.gamma < 1.0, "Choose rho so that ||R h|| < 1."
        # truncated Neumann series S = sum_{j=0..m} (-A)^j
        S = np.eye(self.A.shape[0])
        M = np.eye(self.A.shape[0])
        for _ in range(1, self.m + 1):
            M = - self.A @ M
            S = S + M
        self.S = S
        I = np.eye(self.A.shape[0])
        self.alpha_bound = float(np.linalg.norm(I - self.B @ self.h @ self.S, 2)**2)
        self.eps_tail = (self.gamma**(self.m + 1)) / (1.0 - self.gamma)

    def correct(self, y_loc: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        r = self.B @ y_loc
        c = - (self.h @ (self.S @ r))
        y_star = y_loc + c
        E0 = float(np.sum((self.B @ y_loc)**2))
        E1 = float(np.sum((self.B @ y_star)**2))
        ratio = (E1 / E0) if E0 > 1e-12 else 0.0
        return y_star, {
            "E_before": E0,
            "E_after": E1,
            "ratio": ratio,
            "alpha_bound": self.alpha_bound,
            "eps_tail": self.eps_tail
        }

# ------------------------
# Deterministic clamp gate
# ------------------------

class GateClamp:
    def __init__(
        self,
        lr: float = 0.03,
        l2: float = 1e-4,
        rj_th: float = 0.5,
        horizon: int = 10,
        cap_low: float = 0.3,
        relax: float = 0.97
    ) -> None:
        self.w = np.array([-0.5, +0.5, +0.2, 0.0], dtype=float)
        self.lr = float(lr); self.l2 = float(l2)
        self.rj_th = float(rj_th); self.horizon = int(horizon)
        self.cap_low = float(cap_low); self.relax = float(relax)
        self.cap = 1.0; self.counter = 0

    def forward(self, RJ: float, margin: float, den: float) -> Tuple[float, np.ndarray]:
        z = np.array([float(RJ), float(margin), 1.0 / max(float(den), 1e-12), 1.0], dtype=float)
        g = safe_sigmoid(float(self.w @ z))
        g = min(g, self.cap)
        return g, z

    def policy_update(self, RJ: float, margin: float) -> None:
        flag = (margin < 0.0) or (RJ > self.rj_th)
        if flag:
            self.counter = self.horizon
        else:
            self.counter = max(0, self.counter - 1)
        if self.counter > 0:
            self.cap = max(self.cap_low, self.relax * self.cap)
        else:
            self.cap = min(1.0, self.cap / self.relax)

    def sgd(self, z: np.ndarray, y_att: float, y_lin: float, target: float) -> Tuple[float, float, float]:
        g = safe_sigmoid(float(self.w @ z))
        g = min(g, self.cap)
        y = g * y_att + (1.0 - g) * y_lin
        e = y - target
        grad = e * (y_att - y_lin) * g * (1.0 - g) * z + self.l2 * self.w
        # simple gradient clip to avoid numeric warnings in stress tests
        nrm = float(np.linalg.norm(grad, 2))
        if nrm > 1.0:
            grad = grad / nrm
        self.w = self.w - self.lr * grad
        return y, g, float(e)

# ------------------------
# Sera core orchestrator
# ------------------------

@dataclass
class SeraConfig:
    d: int = 16
    d_v: int = 1
    r: int = 256
    tau: float = 1.3
    gamma: float = 0.985
    clip_c: Optional[float] = 3.0
    eps: float = 1e-6
    beta_floor: float = 1e-3
    seed: int = 20251019
    use_kahan: bool = True
    r_v: int = 32
    L: int = 6
    ccr_V: int = 3
    ccr_rho: float = 0.2
    ccr_m: int = 3

class SeraCore:
    def __init__(self, cfg: SeraConfig) -> None:
        self.cfg = cfg
        self.attn = PRFAttn(
            d=cfg.d, d_v=cfg.d_v, r=cfg.r, tau=cfg.tau, gamma=cfg.gamma,
            clip_c=cfg.clip_c, eps=cfg.eps, beta_floor=cfg.beta_floor,
            seed=cfg.seed ^ 1, use_kahan=cfg.use_kahan, r_v=cfg.r_v
        )
        self.mem = SOSMemory(L=cfg.L, seed=cfg.seed ^ 2)
        self.ccr = CCRCorrector(V=cfg.ccr_V, rho=cfg.ccr_rho, m=cfg.ccr_m)
        self.gate = GateClamp()
        self.prev_hash = hashlib.sha256(b"SeraCore-v2.1").digest().hex()

    def step_ingest(self, k: np.ndarray, v: np.ndarray) -> None:
        self.attn.ingest(k, v)

    def fuse_predict(
        self,
        q: np.ndarray,
        overlays: Optional[Dict[str, List]] = None,
        budgets: Optional[Dict[str, float]] = None,
        RJ: float = 0.0,
        margin: float = 0.0
    ) -> Dict[str, float]:
        y_vec, info = self.attn.query(q, overlays=overlays, budgets=budgets)
        y_att = float(np.sum(y_vec))  # d_v -> scalar for fusion demo
        # simple CCR correction around a local 3-patch vector
        y_loc = np.array([y_att, y_att, y_att], dtype=float)
        y_star, cinfo = self.ccr.correct(y_loc)
        y_ccr = float(np.mean(y_star))
        y_lin = float(self.mem.step(y_ccr))
        self.gate.policy_update(RJ=RJ, margin=margin)
        g, z = self.gate.forward(RJ=RJ, margin=margin, den=info["den"])
        y = g * y_ccr + (1.0 - g) * y_lin
        return {
            "y": y, "y_att": y_att, "y_ccr": y_ccr, "y_lin": y_lin,
            "g": g, "den": float(info["den"]), "ccr_ratio": cinfo["ratio"],
            "alpha_bound": cinfo["alpha_bound"]
        }

    def gate_learn(self, features: np.ndarray, y_att: float, y_lin: float, target: float) -> Tuple[float, float, float]:
        return self.gate.sgd(features, y_att, y_lin, target)

    def snapshot(self) -> Dict[str, dict]:
        return {
            "attn": self.attn.snapshot(),
            "mem": {
                "z1": self.mem.z1.copy(),
                "z2": self.mem.z2.copy(),
                "b0": self.mem.b0.copy(),
                "b1": self.mem.b1.copy(),
                "b2": self.mem.b2.copy(),
                "a1": self.mem.a1.copy(),
                "a2": self.mem.a2.copy(),
            },
            "prev_hash": self.prev_hash
        }

    def restore(self, snap: Dict[str, dict]) -> None:
        self.attn.restore(snap["attn"])
        m = snap["mem"]
        self.mem.z1 = m["z1"].copy(); self.mem.z2 = m["z2"].copy()
        self.mem.b0 = m["b0"].copy(); self.mem.b1 = m["b1"].copy(); self.mem.b2 = m["b2"].copy()
        self.mem.a1 = m["a1"].copy(); self.mem.a2 = m["a2"].copy()
        self.prev_hash = snap.get("prev_hash", self.prev_hash)

# ------------------------
# Lightweight smoke test
# ------------------------

def _smoke() -> Dict[str, float]:
    cfg = SeraConfig()
    core = SeraCore(cfg)
    rng = np.random.default_rng(123)
    for _ in range(32):
        k = rng.normal(0, 1, size=(cfg.d,))
        v = rng.normal(0, 1, size=(cfg.d_v,))
        core.step_ingest(k, v)
        q = rng.normal(0, 1, size=(cfg.d,))
        out = core.fuse_predict(q)
    snap = core.snapshot()
    core.restore(snap)
    # deterministic hash of a few floats
    h = hashlib.sha256(str((out["y"], out["g"], out["den"])).encode()).hexdigest()
    out["hash"] = h
    return out

if __name__ == "__main__":
    res = _smoke()
    print("SeraCore smoke:", res)
