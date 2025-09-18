#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SERA: Self-normalized Exponential-kernel Ratio Attention
--------------------------------------------------------
Constant-time (per token) softmax attention via exponential-kernel sufficient statistics.

This module provides:
  * Positive random features for the exponential kernel (softmax kernel)
  * Sufficient-statistics builder: Z = sum φ(k) v^T, z = sum φ(k)
  * SERA query: y_hat(q) = (φ(q)^T Z) / (φ(q)^T z + λ)
  * Optional streaming update with exponential decay γ
  * Exact softmax attention (for benchmarking)
  * Lambda calibration, diagnostics, and empirical experiments:
        - Error vs feature dimension r (log–log)
        - Hot-path latency vs sequence length n
        - Constant-value test, unbiasedness sanity

Notes
-----
* Uses double precision (float64) throughout.
* Exponent clipping is provided for numerical stability (configurable).
* No external ML frameworks; NumPy + Matplotlib + Pandas only.
* Plots follow the policy: matplotlib only, one chart per figure, no explicit colors.

MIT License
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import time
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Core: Positive Random Features for Exponential Kernel (Softmax Kernel)
# -----------------------------------------------------------------------------

def positive_random_features(X: np.ndarray, W: np.ndarray, tau: float, clip: float = 40.0) -> np.ndarray:
    """
    Compute nonnegative features φ(X) for the exponential kernel:
        φ_i(x) = 1/sqrt(r) * exp( w_i^T x / sqrt(tau) - ||x||^2/(2 tau) )
    where columns of W are i.i.d. N(0, I_d) random vectors.

    Args:
        X:   (N, d) array of inputs
        W:   (d, r) array of random weights
        tau: temperature parameter (typically sqrt(d) for scaled dot-product attention)
        clip: exponent clipping bound to avoid overflow/underflow

    Returns:
        phi: (N, r) nonnegative features
    """
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    N, d = X.shape
    r = W.shape[1]
    # Linear term (N, r)
    lin = (X @ W) / np.sqrt(tau)
    # Norm correction term (N, 1)
    norms = np.sum(X * X, axis=1, keepdims=True) / (2.0 * tau)
    expo = lin - norms
    if clip is not None:
        expo = np.clip(expo, -float(clip), float(clip))
    phi = np.exp(expo, dtype=np.float64) / np.sqrt(r)
    return phi


# -----------------------------------------------------------------------------
# SERA class
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class SERA:
    d: int
    dv: int
    r: int
    tau: Optional[float] = None
    clip: float = 40.0
    rng: Optional[np.random.Generator] = None

    def __post_init__(self):
        if self.tau is None:
            self.tau = math.sqrt(self.d)
        if self.rng is None:
            self.rng = np.random.default_rng(0)
        # Random feature weights (d, r)
        self.W = self.rng.standard_normal(size=(self.d, self.r)).astype(np.float64)
        # Sufficient statistics
        self.Z = np.zeros((self.r, self.dv), dtype=np.float64)
        self.z = np.zeros((self.r,), dtype=np.float64)

    # ----- Build from batch of keys/values -----
    def build(self, K: np.ndarray, V: np.ndarray):
        """
        Build sufficient statistics from full key/value sets.
        """
        K = np.asarray(K, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        phiK = positive_random_features(K, self.W, self.tau, self.clip)  # (n, r)
        self.Z = phiK.T @ V  # (r, dv)
        self.z = phiK.sum(axis=0)  # (r,)

    # ----- Streaming update -----
    def update(self, k: np.ndarray, v: np.ndarray, gamma: float = 1.0):
        """
        Update sufficient statistics with a single (k,v) pair.
        If gamma < 1, applies exponential decay (streaming).
        """
        k = np.asarray(k, dtype=np.float64).reshape(1, -1)
        v = np.asarray(v, dtype=np.float64).reshape(1, -1)
        phi = positive_random_features(k, self.W, self.tau, self.clip)[0]  # (r,)
        if gamma != 1.0:
            self.Z *= float(gamma)
            self.z *= float(gamma)
        # Rank-1 update
        self.Z += np.outer(phi, v.astype(np.float64))
        self.z += phi

    # ----- Query -----
    def query(self, Q: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SERA outputs for a batch of queries Q.
        Returns (y_hat, denominators).
        """
        Q = np.asarray(Q, dtype=np.float64)
        phiQ = positive_random_features(Q, self.W, self.tau, self.clip)  # (m, r)
        num = phiQ @ self.Z  # (m, dv)
        den = phiQ @ self.z  # (m,)
        y_hat = num / (den[:, None] + float(lam))
        return y_hat, den

    # ----- Lambda calibration -----
    def calibrate_lambda(self, Q_calib: np.ndarray, frac: float = 0.01, eps: float = 1e-12) -> float:
        """
        Calibrate ridge λ as a fraction of the median denominator over a small query set.
        """
        phiQ = positive_random_features(np.asarray(Q_calib, dtype=np.float64), self.W, self.tau, self.clip)
        den = phiQ @ self.z
        med = float(np.median(den))
        lam = max(frac * med, float(eps))
        return lam


# -----------------------------------------------------------------------------
# Exact softmax attention (baseline for evaluation)
# -----------------------------------------------------------------------------

def exact_softmax_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Exact scaled dot-product attention for a batch of queries.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d = K.shape[1]
    scores = (Q @ K.T) / math.sqrt(d)  # (m, n)
    scores = scores - scores.max(axis=1, keepdims=True)  # stabilize
    w = np.exp(scores, dtype=np.float64)
    w = w / w.sum(axis=1, keepdims=True)
    return w @ V


# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------

def experiment_error_vs_r(d: int = 64, dv: int = 32, n: int = 4096, m: int = 256,
                          r_list: Tuple[int, ...] = (16, 32, 64, 128, 256),
                          alpha: float = 0.25, seed: int = 0,
                          outdir: Optional[str] = None) -> pd.DataFrame:
    """
    Evaluate relative error vs feature dimension r on log–log scale.
    Scales inputs by alpha to control the log-normal variance.
    """
    rng = np.random.default_rng(seed)
    K = rng.standard_normal(size=(n, d)) * alpha
    V = rng.standard_normal(size=(n, dv)) / math.sqrt(dv)
    Q = rng.standard_normal(size=(m, d)) * alpha

    y_true = exact_softmax_attention(Q, K, V)
    y_true_norm = float(np.sqrt(np.mean(np.sum(y_true * y_true, axis=1))))

    rows = []
    for r in r_list:
        sera = SERA(d=d, dv=dv, r=r, tau=math.sqrt(d), clip=40.0, rng=rng)
        sera.build(K, V)
        lam = sera.calibrate_lambda(Q[: min(32, m)], frac=0.01, eps=1e-12)

        y_hat, den = sera.query(Q, lam=lam)
        err = y_hat - y_true
        rmse = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
        rel_rmse = rmse / (y_true_norm + 1e-12)
        mean_l2 = float(np.mean(np.linalg.norm(err, axis=1)))
        max_l2 = float(np.max(np.linalg.norm(err, axis=1)))
        rows.append(dict(r=r, lambda_=lam, rel_RMSE=rel_rmse, mean_L2=mean_l2, max_L2=max_l2))

    df = pd.DataFrame(rows).sort_values("r").reset_index(drop=True)
    # Log–log slope
    x = np.log(df["r"].values.astype(np.float64))
    y = np.log(df["rel_RMSE"].values.astype(np.float64))
    slope, intercept = np.polyfit(x, y, 1)
    df["loglog_slope_estimate"] = [float(slope)] * len(df)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.loglog(df["r"], df["rel_RMSE"], marker="o", label="SERA rel_RMSE")
    c_ref = df["rel_RMSE"].iloc[0] * math.sqrt(df["r"].iloc[0])
    ref = c_ref / np.sqrt(df["r"])
    plt.loglog(df["r"], ref, linestyle="--", label="~ r^{-1/2} reference")
    plt.xlabel("feature dimension r")
    plt.ylabel("relative RMSE")
    plt.title(f"Error vs r (alpha={alpha})   slope≈{slope:.3f}")
    plt.legend()
    plt.tight_layout()
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, "sera_error_vs_r.png"), dpi=160)
    plt.show()

    if outdir:
        df.to_csv(os.path.join(outdir, "sera_error_vs_r.csv"), index=False)
    return df


def experiment_latency_vs_n(d: int = 64, dv: int = 32, n_list: Tuple[int, ...] = (512, 1024, 2048, 4096, 8192),
                            m: int = 64, r: int = 128, alpha: float = 0.25,
                            seed: int = 1, outdir: Optional[str] = None) -> pd.DataFrame:
    """
    Measure per-query latency vs sequence length n for SERA (hot path) and exact softmax.
    """
    rng = np.random.default_rng(seed)
    n_max = max(n_list)
    K_full = rng.standard_normal(size=(n_max, d)) * alpha
    V_full = rng.standard_normal(size=(n_max, dv)) / math.sqrt(dv)
    Q = rng.standard_normal(size=(m, d)) * alpha

    rows = []
    for n in n_list:
        K = K_full[:n]
        V = V_full[:n]

        sera = SERA(d=d, dv=dv, r=r, tau=math.sqrt(d), clip=40.0, rng=rng)
        sera.build(K, V)
        lam = sera.calibrate_lambda(Q[: min(16, m)], frac=0.01)

        # SERA hot-path
        t0 = time.time()
        y_hat, den = sera.query(Q, lam=lam)
        t1 = time.time()
        sera_ms_per_query = (t1 - t0) * 1000.0 / m

        # exact softmax
        t0 = time.time()
        y_true = exact_softmax_attention(Q, K, V)
        t1 = time.time()
        exact_ms_per_query = (t1 - t0) * 1000.0 / m

        rows.append(dict(n=n, sera_ms_per_query=sera_ms_per_query,
                         exact_ms_per_query=exact_ms_per_query, r=r, lambda_=lam))

    df = pd.DataFrame(rows)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(df["n"], df["sera_ms_per_query"], marker="o", label="SERA hot-path")
    plt.plot(df["n"], df["exact_ms_per_query"], marker="s", label="Exact softmax")
    plt.xlabel("sequence length n")
    plt.ylabel(f"milliseconds per query (batch m={m})")
    plt.title("Hot-path latency vs n")
    plt.legend()
    plt.tight_layout()
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, "sera_latency_vs_n.png"), dpi=160)
    plt.show()

    if outdir:
        df.to_csv(os.path.join(outdir, "sera_latency_vs_n.csv"), index=False)
    return df


# -----------------------------------------------------------------------------
# Diagnostics / sanity checks
# -----------------------------------------------------------------------------

def constant_value_test(d: int = 64, dv: int = 16, n: int = 2048, m: int = 64,
                        r: int = 128, alpha: float = 0.25, lam_frac: float = 0.01,
                        seed: int = 123) -> Dict[str, float]:
    """
    If all v_j are identical, attention must return that constant vector.
    """
    rng = np.random.default_rng(seed)
    K = rng.standard_normal(size=(n, d)) * alpha
    v0 = rng.standard_normal(size=(dv,)) / math.sqrt(dv)
    V = np.tile(v0[None, :], (n, 1))
    Q = rng.standard_normal(size=(m, d)) * alpha

    y_true = exact_softmax_attention(Q, K, V)
    # Should be equal to v0 across queries
    assert np.allclose(y_true, np.tile(v0, (m, 1)), atol=1e-6)

    sera = SERA(d=d, dv=dv, r=r, tau=math.sqrt(d), clip=40.0, rng=rng)
    sera.build(K, V)
    lam = sera.calibrate_lambda(Q[: min(16, m)], frac=lam_frac, eps=1e-12)
    y_hat, den = sera.query(Q, lam=lam)

    err = y_hat - v0[None, :]
    mean_l2 = float(np.mean(np.linalg.norm(err, axis=1)))
    max_l2 = float(np.max(np.linalg.norm(err, axis=1)))
    shrink = float(np.median(den / (den + lam)))
    return dict(mean_L2=mean_l2, max_L2=max_l2, median_shrink_factor=shrink,
                r=r, lambda_=lam)


def unbiasedness_sanity(d: int = 64, dv: int = 8, n: int = 1024,
                        r: int = 64, alpha: float = 0.25,
                        repeats: int = 128, seed: int = 77) -> Dict[str, float]:
    """
    Monte Carlo sanity: averages of A_r and B_r across different random weights W
    approach the true A and B.
    """
    rng = np.random.default_rng(seed)
    K = rng.standard_normal(size=(n, d)) * alpha
    V = rng.standard_normal(size=(n, dv)) / math.sqrt(dv)
    q = rng.standard_normal(size=(d,)) * alpha

    # True A (unnormalized numerator) and B (denominator)
    dscale = math.sqrt(d)
    scores = (K @ q) / dscale
    w = np.exp(scores - scores.max())
    B_true = float(np.sum(w))
    A_true_vec = (w[:, None] * V).sum(axis=0)

    # Monte Carlo over different W
    tau = math.sqrt(d)
    As, Bs = [], []
    for _ in range(repeats):
        sera = SERA(d=d, dv=dv, r=r, tau=tau, clip=40.0, rng=rng)
        sera.build(K, V)
        phi_q = positive_random_features(q.reshape(1, -1), sera.W, tau, clip=40.0)[0]
        As.append(phi_q @ sera.Z)
        Bs.append(phi_q @ sera.z)

    A_mean = np.mean(np.stack(As, axis=0), axis=0)
    B_mean = float(np.mean(Bs))
    A_rel = float(np.linalg.norm(A_mean - A_true_vec) / (np.linalg.norm(A_true_vec) + 1e-12))
    B_rel = float(abs(B_mean - B_true) / (abs(B_true) + 1e-12))
    return dict(A_rel_error=A_rel, B_rel_error=B_rel, r=r, repeats=repeats)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SERA: constant-time softmax attention via exponential-kernel sufficient statistics")
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_all = sub.add_parser("all", help="Run all default experiments")
    p_all.add_argument("--outdir", type=str, default=None)

    p_err = sub.add_parser("err", help="Error vs r")
    p_err.add_argument("--d", type=int, default=64)
    p_err.add_argument("--dv", type=int, default=32)
    p_err.add_argument("--n", type=int, default=4096)
    p_err.add_argument("--m", type=int, default=256)
    p_err.add_argument("--r_list", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    p_err.add_argument("--alpha", type=float, default=0.25)
    p_err.add_argument("--seed", type=int, default=0)
    p_err.add_argument("--outdir", type=str, default=None)

    p_lat = sub.add_parser("lat", help="Latency vs n")
    p_lat.add_argument("--d", type=int, default=64)
    p_lat.add_argument("--dv", type=int, default=32)
    p_lat.add_argument("--n_list", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    p_lat.add_argument("--m", type=int, default=64)
    p_lat.add_argument("--r", type=int, default=128)
    p_lat.add_argument("--alpha", type=float, default=0.25)
    p_lat.add_argument("--seed", type=int, default=1)
    p_lat.add_argument("--outdir", type=str, default=None)

    p_chk = sub.add_parser("check", help="Consistency checks")
    p_chk.add_argument("--alpha", type=float, default=0.25)

    args = parser.parse_args()

    # Default action: run "all" with no outdir
    if args.cmd in (None, "all"):
        outdir = getattr(args, "outdir", None)
        print("[SERA] Running default experiments…")
        df_err = experiment_error_vs_r(outdir=outdir)
        df_lat = experiment_latency_vs_n(outdir=outdir)
        print("\n[Error vs r]\n", df_err)
        print("\n[Latency vs n]\n", df_lat)
        return

    if args.cmd == "err":
        df = experiment_error_vs_r(d=args.d, dv=args.dv, n=args.n, m=args.m,
                                   r_list=tuple(args.r_list), alpha=args.alpha,
                                   seed=args.seed, outdir=args.outdir)
        print(df)
        return

    if args.cmd == "lat":
        df = experiment_latency_vs_n(d=args.d, dv=args.dv, n_list=tuple(args.n_list),
                                     m=args.m, r=args.r, alpha=args.alpha,
                                     seed=args.seed, outdir=args.outdir)
        print(df)
        return

    if args.cmd == "check":
        res_const = constant_value_test(alpha=args.alpha)
        res_unbias = unbiasedness_sanity(alpha=args.alpha)
        print("[Constant-value test]", res_const)
        print("[Unbiasedness sanity]", res_unbias)
        return


if __name__ == "__main__":
    main()
