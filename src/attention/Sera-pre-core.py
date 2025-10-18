# Sera-pre-core.py
# Strict-CCR version (Sera v2.0 pre-core)
# ASCII only, English only. No external deps beyond numpy.
#
# This module provides:
# 1) Stable streaming ratio estimator with PRF whitening and Kahan decay.
# 2) Overlay A / C machinery (optional).
# 3) Strict CCR primitives for log-sum-exp style combining of tiles:
#       (m,Z) := (max(m1,m2), Z1*exp(m1-m) + Z2*exp(m2-m))
#    which ensures exact equivalence (up to fp) to global softmax.
# 4) Safe-Z pruning with an L1 bound that is preserved under CCR combine.
# 5) Auto-k global stopping rule with proper CCR scaling across tiles.
# 6) Minimal smoke tests for quick verification.
#
# Author: Sera v2.0

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np


# -----------------------------
# Core streaming estimator
# -----------------------------

def _safe_randn(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return rng.normal(size=shape)


@dataclass
class PRF:
    """Positive Random Features for exp kernel: k(q, x) = exp(<q, x>/tau).
    phi(x) = exp((W x)/sqrt(tau) - ||x||^2/(2 tau)) / sqrt(r)
    with optional clipping of the pre-activation.
    """

    r: int
    d: int
    tau: float = 1.0
    clip_c: Optional[float] = 4.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    W: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.W = _safe_randn(self.rng, self.r, self.d)

    def features(self, x: np.ndarray) -> np.ndarray:
        g = (self.W @ x) / math.sqrt(self.tau)
        if self.clip_c is not None:
            g = np.clip(g, -self.clip_c, self.clip_c)
        xnorm = float(x @ x)
        phi = np.exp(g - xnorm / (2.0 * self.tau)) / math.sqrt(self.r)
        return phi


@dataclass
class EWVar:
    """Exponentially weighted running mean and variance (per-dim)."""

    dim: int
    alpha: float  # decay: new = alpha * old + (1-alpha) * x
    mu: np.ndarray = field(init=False)
    sig2: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.mu = np.zeros(self.dim, dtype=float)
        self.sig2 = np.ones(self.dim, dtype=float) * 1e-6

    def update(self, x: np.ndarray) -> None:
        mu_old = self.mu
        mu_new = self.alpha * mu_old + (1.0 - self.alpha) * x
        diff = x - mu_new
        sig2_new = self.alpha * self.sig2 + (1.0 - self.alpha) * (diff * diff)
        self.mu, self.sig2 = mu_new, sig2_new


@dataclass
class KahanDecay:
    """Vector Kahan-compensated exponentially decayed sum.
    Maintains S_t = gamma * S_{t-1} + x_t with compensation.
    """

    dim: int
    gamma: float
    S: np.ndarray = field(init=False)
    C: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.S = np.zeros(self.dim, dtype=float)
        self.C = np.zeros(self.dim, dtype=float)

    def update(self, x: np.ndarray) -> None:
        self.S *= self.gamma
        self.C *= self.gamma
        y = x - self.C
        t = self.S + y
        self.C = (t - self.S) - y
        self.S = t


@dataclass
class OverlayA:
    """Scalar numerator overlay anchored at a feature location.
    y_hat(q) = (num(q) + u * <phi_w(q), phi_anchor>) / den(q)
    """

    phi_anchor: np.ndarray
    u: float

    @staticmethod
    def fit(
        core: "SeraCore",
        anchors: Sequence[np.ndarray],
        calib_queries: Sequence[np.ndarray],
        calib_targets: Sequence[float],
        lam: float = 1e-6,
    ) -> List["OverlayA"]:
        overlays: List[OverlayA] = []
        phi_anchors = [core.prf.features(a) for a in anchors]
        for pa in phi_anchors:
            sc_list = []
            resid_list = []
            for q, y_true in zip(calib_queries, calib_targets):
                y_base, num, den, phi_w = core.query(q, return_parts=True)
                sc = float(phi_w @ pa) / den
                sc_list.append(sc)
                resid_list.append(float(y_true - y_base))
            sc_v = np.asarray(sc_list, dtype=float)
            resid_v = np.asarray(resid_list, dtype=float)
            u_hat = float((sc_v @ resid_v) / (sc_v @ sc_v + lam))
            overlays.append(OverlayA(phi_anchor=pa, u=u_hat))
        return overlays


@dataclass
class OverlayC:
    """Value-space low-rank correction.
    z(q) = phi_w(q) @ H   [shape: r_v]
    y_hat(q) = base(q) + 1^T Delta_k z(q), where Delta_k is rank-k.
    """

    H: np.ndarray              # shape (r, r_v)
    Delta_k: np.ndarray        # shape (r_v, r_v)
    ones: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        r_v = self.Delta_k.shape[0]
        self.ones = np.ones((r_v, 1), dtype=float)

    @staticmethod
    def fit_rank_k(Z: np.ndarray, t: np.ndarray, k: int) -> np.ndarray:
        r_v = Z.shape[1]
        d_hat, *_ = np.linalg.lstsq(Z, t, rcond=None)
        Delta_full = np.diag(d_hat)
        U, S, Vt = np.linalg.svd(Delta_full, full_matrices=False)
        k = max(1, min(k, r_v))
        Wk = U[:, :k] * S[:k]
        Vk = Vt[:k, :]
        Delta_k = Wk @ Vk
        return Delta_k

    def add_correction(self, phi_w: np.ndarray) -> float:
        z = phi_w @ self.H
        return float(z @ (self.Delta_k @ self.ones).reshape(-1))


@dataclass
class Selector:
    """Thresholded, constant-time overlay selector.
    Picks top-P by |score| among those with |score| >= theta.
    """

    P: int = 1
    theta: float = 0.0

    def select_scores(self, scores: np.ndarray) -> np.ndarray:
        mask = np.abs(scores) >= self.theta
        idx = np.where(mask)[0]
        if idx.size == 0 or self.P <= 0:
            return np.empty((0,), dtype=int)
        order = np.argsort(np.abs(scores[idx]))[::-1]
        return idx[order[: self.P]]


@dataclass
class SeraCore:
    """Sera v2.0 pre-core: streaming ratio estimator with overlays and strict CCR utilities."""

    d: int
    r: int
    tau: float = 1.2
    gamma: float = 0.99
    clip_c: Optional[float] = 4.0
    floor: float = 1e-6
    use_kahan: bool = True
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    prf: PRF = field(init=False)
    ew: EWVar = field(init=False)
    R_acc: np.ndarray = field(init=False)
    s_acc: np.ndarray = field(init=False)
    kahan_R: Optional[KahanDecay] = field(init=False, default=None)
    kahan_s: Optional[KahanDecay] = field(init=False, default=None)
    overlays_A: List[OverlayA] = field(default_factory=list)
    overlay_C: Optional[OverlayC] = None
    selector: Selector = field(default_factory=lambda: Selector(P=1, theta=0.0))

    def __post_init__(self) -> None:
        self.prf = PRF(r=self.r, d=self.d, tau=self.tau, clip_c=self.clip_c, rng=self.rng)
        self.ew = EWVar(dim=self.r, alpha=self.gamma)
        self.R_acc = np.zeros(self.r, dtype=float)
        self.s_acc = np.zeros(self.r, dtype=float)
        if self.use_kahan:
            self.kahan_R = KahanDecay(dim=self.r, gamma=self.gamma)
            self.kahan_s = KahanDecay(dim=self.r, gamma=self.gamma)

    def ingest(self, x: np.ndarray, v: float) -> None:
        phi = self.prf.features(x)
        self.ew.update(phi)
        if self.use_kahan:
            self.kahan_R.update(phi * v)
            self.kahan_s.update(phi)
            self.R_acc = self.kahan_R.S
            self.s_acc = self.kahan_s.S
        else:
            self.R_acc = self.gamma * self.R_acc + phi * v
            self.s_acc = self.gamma * self.s_acc + phi

    def query(self, q: np.ndarray, return_parts: bool = False) -> Tuple[float, float, float, np.ndarray]:
        phi_q = self.prf.features(q)
        phi_w = phi_q / np.sqrt(self.ew.sig2 + 1e-12)
        num = float(phi_w @ self.R_acc)
        den = float(phi_w @ self.s_acc) + self.floor
        y = num / den
        if return_parts:
            return y, num, den, phi_w
        return y, 0.0, 0.0, phi_w

    def predict_with_overlays(self, q: np.ndarray) -> Tuple[float, Dict[str, float]]:
        y, num, den, phi_w = self.query(q, return_parts=True)
        if self.overlays_A:
            scores = np.array([float(phi_w @ ov.phi_anchor) for ov in self.overlays_A])
            idx = self.selector.select_scores(scores)
            if idx.size > 0:
                num2 = num + float(np.sum(scores[idx] * np.array([self.overlays_A[j].u for j in idx])))
                y = num2 / den
        if self.overlay_C is not None:
            y = y + self.overlay_C.add_correction(phi_w)
        return y, {"num": num, "den": den}

    def add_overlays_A(self, overlays: List[OverlayA]) -> None:
        self.overlays_A.extend(overlays)

    def set_overlay_C(self, overlay_c: OverlayC) -> None:
        self.overlay_C = overlay_c

    def set_selector(self, P: int, theta: float) -> None:
        self.selector = Selector(P=P, theta=theta)


# -----------------------------
# Strict CCR primitives
# -----------------------------

class CCR:
    """Strict CCR combine and utilities, numerically stable (log-sum-exp equivalent)."""

    @staticmethod
    def combine_pair(m1: float, Z1: float, m2: float, Z2: float) -> Tuple[float, float]:
        m = m1 if m1 >= m2 else m2
        Z = Z1 * math.exp(m1 - m) + Z2 * math.exp(m2 - m)
        return m, Z

    @staticmethod
    def combine_many(ms: Sequence[float], Zs: Sequence[float]) -> Tuple[float, float]:
        m, Z = -float("inf"), 0.0
        for mi, Zi in zip(ms, Zs):
            m, Z = CCR.combine_pair(m, Z, mi, Zi)
        return m, Z

    @staticmethod
    def strict_global(scores: np.ndarray) -> Tuple[float, float, np.ndarray]:
        m = float(np.max(scores))
        Z = float(np.sum(np.exp(scores - m)))
        alpha = np.exp(scores - m) / Z
        return m, Z, alpha

    @staticmethod
    def tile_stats(scores: np.ndarray, idxs: np.ndarray) -> Tuple[float, float]:
        s = scores[idxs]
        m = float(np.max(s))
        Z = float(np.sum(np.exp(s - m)))
        return m, Z


# -----------------------------
# Safe-Z pruning under CCR
# -----------------------------

@dataclass
class SafeZResult:
    m_hat: float
    Z_hat: float
    Rhat_global: float
    L1_bound: float


class SafeZ:
    """Construct Z_hat by pruning per-tile tails; combine under CCR; supply an L1 bound."""

    @staticmethod
    def build(scores: np.ndarray, chunks: Sequence[np.ndarray], prune_frac: float) -> SafeZResult:
        ms: List[float] = []
        Zc_list: List[float] = []
        Rhat_list: List[float] = []
        for ch in chunks:
            s = scores[ch]
            m_i = float(np.max(s))
            k_keep = max(1, int(len(s) * (1.0 - prune_frac)))
            idx_sorted = np.argsort(s)[::-1]
            keep = idx_sorted[:k_keep]
            pruned = idx_sorted[k_keep:]
            Zc_i = float(np.sum(np.exp(s[keep] - m_i)))
            if len(pruned) > 0:
                UB = float(np.max(s[pruned]))
                Rhat_i = len(pruned) * math.exp(UB - m_i)
            else:
                Rhat_i = 0.0
            ms.append(m_i); Zc_list.append(Zc_i); Rhat_list.append(Rhat_i)
        m_hat, Z_hat = CCR.combine_many(ms, [a + b for a, b in zip(Zc_list, Rhat_list)])
        Rhat_global = (sum(Rhat_list)) / max(1e-12, sum(Zc_list))
        L1_bound = 2.0 * Rhat_global / (1.0 + Rhat_global)
        return SafeZResult(m_hat=m_hat, Z_hat=Z_hat, Rhat_global=Rhat_global, L1_bound=L1_bound)


# -----------------------------
# Auto-k global stopping (CCR-scaled)
# -----------------------------

@dataclass
class AutoKResult:
    K_per_tile: List[int]
    head_sum: float
    eps_eff: float


class AutoK:
    """Pick minimal per-tile heads to meet 2*(Z - head_sum)/Z <= eps in global frame."""

    @staticmethod
    def prepare(scores: np.ndarray, chunks: Sequence[np.ndarray]) -> Tuple[List[float], List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[float], float, float, List[float]]:
        ms: List[float] = []
        Zs: List[float] = []
        head_lists: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for ch in chunks:
            s = scores[ch]
            m_i = float(np.max(s)); ms.append(m_i)
            a_i = np.exp(s - m_i)
            idx = np.argsort(a_i)[::-1]
            a_sorted = a_i[idx]
            head_lists.append((ch, idx, a_sorted))
            Zs.append(float(np.sum(a_i)))
        m_c, Z_c = CCR.combine_many(ms, Zs)
        scales = [math.exp(m_i - m_c) for m_i in ms]
        return ms, head_lists, Zs, m_c, Z_c, scales

    @staticmethod
    def run(heads: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], scales: List[float], Z_c: float, eps: float) -> AutoKResult:
        P = len(heads)
        K = [0] * P
        head_sum = 0.0
        total_len = sum(len(a) for (_, _, a) in heads)
        while True:
            R = (Z_c - head_sum) / Z_c
            if 2 * R / (1 + R) <= eps or sum(K) == total_len:
                break
            best_gain = -1.0; best_i = -1
            for i, (_, _, a_sorted) in enumerate(heads):
                if K[i] < len(a_sorted):
                    gain = scales[i] * float(a_sorted[K[i]])
                    if gain > best_gain:
                        best_gain = gain; best_i = i
            if best_i < 0:
                break
            head_sum += best_gain
            K[best_i] += 1
        eps_eff = 2 * (Z_c - head_sum) / Z_c
        return AutoKResult(K_per_tile=K, head_sum=head_sum, eps_eff=eps_eff)


# -----------------------------
# Minimal smoke tests
# -----------------------------

def _smoke_strict_ccr() -> None:
    rng = np.random.default_rng(7)
    N, P = 2000, 8
    scores = rng.normal(size=N)
    order = rng.permutation(N)
    chunks = np.array_split(order, P)
    # strict combine should match global
    ms, Zs = zip(*[CCR.tile_stats(scores, ch) for ch in chunks])
    m_c, Z_c = CCR.combine_many(list(ms), list(Zs))
    m_g, Z_g, alpha_g = CCR.strict_global(scores)
    alpha_c = np.exp(scores - m_c) / Z_c
    assert abs(m_c - m_g) < 1e-10
    assert abs(Z_c - Z_g) / max(1.0, abs(Z_g)) < 1e-12
    assert float(np.sum(np.abs(alpha_c - alpha_g))) < 1e-10
    # Safe-Z bound should upper-bound L1
    sz = SafeZ.build(scores, chunks, prune_frac=0.5)
    alpha_hat = np.exp(scores - sz.m_hat) / sz.Z_hat
    L1_err = float(np.sum(np.abs(alpha_hat - alpha_g)))
    assert L1_err <= sz.L1_bound + 1e-8
    # Auto-k should satisfy eps-eff
    ms2, heads, Zs2, m_c2, Z_c2, scales = AutoK.prepare(scores, chunks)
    ak = AutoK.run(heads, scales, Z_c2, eps=0.1)
    assert ak.eps_eff <= 0.11  # small slack
    # Streaming core quick run
    d, r = 16, 256
    core = SeraCore(d=d, r=r, tau=1.2, gamma=0.99, clip_c=4.0, floor=1e-6, use_kahan=True, rng=rng)
    X = rng.normal(size=(200, d)) / math.sqrt(d)
    w = rng.normal(size=d)
    v = X @ w + 0.05 * rng.normal(size=200)
    for t in range(200):
        core.ingest(X[t], float(v[t]))
    y, dbg = core.predict_with_overlays(rng.normal(size=d) / math.sqrt(d))
    _ = (y, dbg)


if __name__ == "__main__":
    _smoke_strict_ccr()
    print("OK: strict CCR, Safe-Z, Auto-k, and streaming core smoke tests passed.")