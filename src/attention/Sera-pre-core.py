# Sera-pre-core.py
# Minimal, numerically-stable pre-core for Sera v2.0
# ASCII only, English only.
#
# Features
# - Positive Random Features (PRF) for exp kernel scoring
# - Streaming ratio estimator with whitening, floor, and Kahan-compensated decay
# - Overlay A (numerator scalar correction) with thresholded, constant-time selection
# - Optional Overlay C (value-space low-rank correction) with rank-k fit via SVD
# - CCR-style harmonization for multiple predictors (Laplacian pseudoinverse)
# - Anytime-margin policy helpers for generation (DecGen policy only; no sampler)
#
# Dependencies: numpy only.
#
# Author: Sera v2.0 pre-core

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _safe_randn(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return rng.normal(size=shape)


@dataclass
class PRF:
    """Positive Random Features for exp kernel: k(q, x) = exp(<q, x>/tau).
    phi(x) = exp((W x)/sqrt(tau) - ||x||^2/(2 tau)) / sqrt(r)
    with optional clipping of the pre-activation.

    Attributes
    ----------
    r : int
        Number of random features.
    d : int
        Input dimension.
    tau : float
        Temperature parameter of the kernel.
    clip_c : Optional[float]
        Clip bound applied elementwise to g = (W x) / sqrt(tau); None disables clipping.
    rng : np.random.Generator
        PRNG for reproducibility.
    W : np.ndarray
        Random weight matrix, shape (r, d).
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
    """Exponentially weighted running mean and variance (per-dimension)."""

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

    Maintains S_t = gamma * S_{t-1} + x_t with compensation to reduce rounding error.
    """

    dim: int
    gamma: float
    S: np.ndarray = field(init=False)
    C: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.S = np.zeros(self.dim, dtype=float)
        self.C = np.zeros(self.dim, dtype=float)

    def update(self, x: np.ndarray) -> None:
        # decay both the main sum and the compensation terms
        self.S *= self.gamma
        self.C *= self.gamma
        # Kahan compensated addition: y = x - C; t = S + y; C = (t - S) - y; S = t
        y = x - self.C
        t = self.S + y
        self.C = (t - self.S) - y
        self.S = t


@dataclass
class OverlayA:
    """Scalar numerator overlay anchored at a feature location.

    y_hat(q) = (num(q) + u * <phi_w(q), phi_anchor>) / den(q)
    where phi_w is whitened query features and u is fit on a local calibration set.
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
        """Fit one overlay per anchor via 1D ridge least squares.

        For each anchor a:
          sc_i = <phi_w(q_i), phi(a)> / den(q_i)
          resid_i = y_true_i - y_base(q_i)
          u = argmin_u sum_i (resid_i - u * sc_i)^2 + lam * u^2
            = (sc^T resid) / (sc^T sc + lam)

        Returns list of OverlayA.
        """
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

    Fit via SVD on calibration pairs (Z, t_full), where t_full is the target correction.
    """

    H: np.ndarray              # shape (r, r_v)
    Delta_k: np.ndarray        # shape (r_v, r_v)
    ones: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        r_v = self.Delta_k.shape[0]
        self.ones = np.ones((r_v, 1), dtype=float)

    @staticmethod
    def fit_rank_k(Z: np.ndarray, t: np.ndarray, k: int) -> np.ndarray:
        """Solve min_D ||Z D 1 - t||^2 and project to rank-k via SVD.
        Returns Delta_k (r_v x r_v)."""
        r_v = Z.shape[1]
        # Solve least squares for d = D 1 in R^{r_v}
        # Z d ~= t  => d = argmin ||Z d - t||^2
        d_hat, *_ = np.linalg.lstsq(Z, t, rcond=None)
        # Lift to a diagonal Delta that maps z -> (d_hat^T z)
        Delta_full = np.diag(d_hat)
        # Rank-k projection
        U, S, Vt = np.linalg.svd(Delta_full, full_matrices=False)
        k = max(1, min(k, r_v))
        Wk = U[:, :k] * S[:k]
        Vk = Vt[:k, :]
        Delta_k = Wk @ Vk
        return Delta_k

    def add_correction(self, phi_w: np.ndarray) -> float:
        z = phi_w @ self.H  # shape (r_v,)
        return float(z @ (self.Delta_k @ self.ones).reshape(-1))


@dataclass
class Selector:
    """Thresholded, constant-time overlay selector.
    Picks top-P by |score| among those with |score| >= theta.
    """

    P: int = 1
    theta: float = 0.0  # absolute score threshold

    def select_scores(self, scores: np.ndarray) -> np.ndarray:
        mask = np.abs(scores) >= self.theta
        idx = np.where(mask)[0]
        if idx.size == 0 or self.P <= 0:
            return np.empty((0,), dtype=int)
        order = np.argsort(np.abs(scores[idx]))[::-1]
        return idx[order[: self.P]]


@dataclass
class SeraCore:
    """Sera v2.0 pre-core: streaming ratio estimator with overlays and CCR.

    Core equation
      base(q) = (phi_w(q) dot R) / (phi_w(q) dot s + floor)

    where phi_w(q) = phi(q) / sqrt(var_r), and R, s are decayed sums of phi * v and phi.

    Numerics
      - Whitening via EW variance
      - Denominator floor to avoid small-divide
      - Optional Kahan-compensated decayed sums
    """

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
        """Update streaming state with key x and scalar value v."""
        phi = self.prf.features(x)
        # EW variance for whitening (update before using in query time)
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
        """Return (y_hat, num, den, phi_w)."""
        phi_q = self.prf.features(q)
        phi_w = phi_q / np.sqrt(self.ew.sig2 + 1e-12)
        num = float(phi_w @ self.R_acc)
        den = float(phi_w @ self.s_acc) + self.floor
        y = num / den
        if return_parts:
            return y, num, den, phi_w
        return y, 0.0, 0.0, phi_w  # parts ignored when return_parts=False

    def predict_with_overlays(self, q: np.ndarray) -> Tuple[float, dict]:
        """Apply base, then overlays: Type A (selected), then optional Type C."""
        y, num, den, phi_w = self.query(q, return_parts=True)

        # Type A overlays (numerator)
        if self.overlays_A:
            scores = np.array([float(phi_w @ ov.phi_anchor) for ov in self.overlays_A])
            idx = self.selector.select_scores(scores)
            if idx.size > 0:
                num2 = num + float(np.sum(scores[idx] * np.array([self.overlays_A[j].u for j in idx])))
                y = num2 / den

        # Type C overlay (value-space low-rank)
        if self.overlay_C is not None:
            y = y + self.overlay_C.add_correction(phi_w)

        debug = {"num": num, "den": den}
        return y, debug

    # Overlay management ----------------------------------------------------
    def add_overlays_A(self, overlays: List[OverlayA]) -> None:
        self.overlays_A.extend(overlays)

    def set_overlay_C(self, overlay_c: OverlayC) -> None:
        self.overlay_C = overlay_c

    def set_selector(self, P: int, theta: float) -> None:
        self.selector = Selector(P=P, theta=theta)


# CCR harmonization ----------------------------------------------------------
def ccr_harmonize(values: Sequence[float]) -> List[float]:
    """CCR-like correction for a set of scalar predictions.
    Solve a Laplacian system on a complete graph to reduce pairwise disagreements.
    Gauge: mean correction = 0.
    """
    y = np.asarray(values, dtype=float).reshape(-1)
    n = y.size
    if n <= 1:
        return y.tolist()
    # Complete graph Laplacian: L = n I - 1 1^T
    L = n * np.eye(n) - np.ones((n, n), dtype=float)
    # Right-hand side that encourages consensus toward mean
    # Here we use a simple projection: y_corr = y - (I - L^+) y_mean_component
    # Equivalent to subtracting deviations via pseudoinverse.
    L_pinv = np.linalg.pinv(L)
    # Target consensus is the average of y
    mean_y = float(np.mean(y))
    # Compute minimal-norm correction c that reduces disagreements
    # Solve L c = - (I - P) y, with P projecting to mean.
    P = np.ones((n, n), dtype=float) / n
    rhs = - (np.eye(n) - P) @ (y - mean_y)
    c = - L_pinv @ rhs
    return (y + c).tolist()


# DecGen policy helpers ------------------------------------------------------
def anytime_margin_policy(margin: float, thr: float, T_low: float, p_low: float, T_high: float = 1.0, p_high: float = 0.95) -> Tuple[float, float]:
    """Return (temperature, top_p) based on margin threshold."""
    if margin < thr:
        return T_low, p_low
    return T_high, p_high


# Minimal smoke test ---------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    d, r = 16, 256
    core = SeraCore(d=d, r=r, tau=1.2, gamma=0.99, clip_c=4.0, floor=1e-6, use_kahan=True, rng=rng)

    # Build a tiny synthetic stream
    T = 200
    q0 = rng.normal(size=d) / math.sqrt(d)
    k = rng.normal(size=(T, d)) / math.sqrt(d)
    w = rng.normal(size=d)
    v = k @ w + 0.05 * rng.normal(size=T)

    # Ingest
    for t in range(T):
        core.ingest(k[t], float(v[t]))

    # Base query
    y_base, dbg = core.predict_with_overlays(q0)[0], {}
    print("Base prediction:", y_base)

    # Fit a single Overlay A near q0 using a few calibrations with synthetic targets
    calib_q = [q0 + 0.1 * rng.normal(size=d) / math.sqrt(d) for _ in range(8)]
    calib_y = []
    for q in calib_q:
        # Construct synthetic targets by nudging base with a small ground-truth offset
        y_b, _, _, _ = core.query(q, return_parts=True)
        calib_y.append(y_b + 0.05 * float(rng.normal()))

    overlays = OverlayA.fit(core, anchors=[q0], calib_queries=calib_q, calib_targets=calib_y, lam=1e-6)
    core.add_overlays_A(overlays)
    core.set_selector(P=1, theta=0.0)

    y_with_A, _ = core.predict_with_overlays(q0)
    print("With Overlay A:", y_with_A)

    # CCR smoke
    vals = [y_base, y_with_A, y_base + 0.02]
    print("CCR:", ccr_harmonize(vals))
