#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SERA: Self-normalized Exponential-kernel Ratio Attention — reference + robust experiments

High-quality refactor goals (names preserved)
--------------------------------------------
- Keep public names: SERAConfig, SERA, PosExpFeature, KahanVec, KahanMat, bench_ms, experiments.
- Improve numerical robustness and performance:
  - Optional lazy decay (avoid O(r*d_v) scaling per token when gamma!=1)
  - Vectorized ingest fast path (phi_batch + GEMM) when gamma==1
  - Better percentile computation and benchmarking hygiene
- Improve experiment credibility:
  - Multi-seed aggregation option for error-vs-r
  - Optional lam parity in exact baseline
"""

import os
import math
import argparse
import csv
import contextlib
from dataclasses import dataclass
from typing import Tuple, Callable, Dict, Iterable, Optional, List

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


# -------------------- Determinism & BLAS control --------------------

@contextlib.contextmanager
def single_thread_blas():
    """Force single-thread BLAS to reduce noise on small/medium problems."""
    keys = [
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    old = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = "1"
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def bench_ms(fn: Callable[[], None], reps: int = 200, warmup: int = 5) -> Dict[str, float]:
    """
    Robust micro-benchmark:
      - warmup calls (not recorded)
      - reps timed with perf_counter_ns()
      - returns median, p95, p99, min, max (in ms)

    Notes:
      - Keep allocations + RNG outside `fn`.
      - Use single_thread_blas() around outer loops when comparing methods.
    """
    import time

    if reps <= 0:
        raise ValueError("reps must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    pc = time.perf_counter_ns
    for _ in range(warmup):
        fn()

    xs = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        t0 = pc()
        fn()
        t1 = pc()
        xs[i] = (t1 - t0) / 1e6  # ms

    xs.sort()
    # percentiles: stable + standard definition
    p50 = float(np.percentile(xs, 50.0, method="linear"))
    p95 = float(np.percentile(xs, 95.0, method="linear"))
    p99 = float(np.percentile(xs, 99.0, method="linear"))
    return {"p50": p50, "p95": p95, "p99": p99, "min": float(xs[0]), "max": float(xs[-1])}


# -------------------- Numerics: Kahan compensated sums --------------------

class KahanVec:
    def __init__(self, dim: int, dtype=np.float64):
        self.sum = np.zeros(dim, dtype=dtype)
        self.c = np.zeros(dim, dtype=dtype)

    def add(self, x: np.ndarray):
        # Kahan summation: sum += x with compensation.
        y = x - self.c
        t = self.sum + y
        self.c = (t - self.sum) - y
        self.sum = t

    def scale(self, s: float):
        self.sum *= s
        self.c *= s

    def value(self):
        return self.sum

    def set_value(self, x: np.ndarray):
        # Replace sum exactly, clear compensation.
        self.sum[...] = x
        self.c[...] = 0


class KahanMat:
    def __init__(self, shape: Tuple[int, int], dtype=np.float64):
        self.sum = np.zeros(shape, dtype=dtype)
        self.c = np.zeros(shape, dtype=dtype)

    def add(self, X: np.ndarray):
        y = X - self.c
        t = self.sum + y
        self.c = (t - self.sum) - y
        self.sum = t

    def scale(self, s: float):
        self.sum *= s
        self.c *= s

    def value(self):
        return self.sum

    def set_value(self, X: np.ndarray):
        self.sum[...] = X
        self.c[...] = 0


# -------------------- Positive feature map for exp kernel --------------------

@dataclass
class PosExpFeature:
    d: int
    r: int
    tau: float
    clip: float = 40.0
    seed: int = 0

    def __post_init__(self):
        if self.d <= 0 or self.r <= 0:
            raise ValueError("d and r must be > 0")
        if self.tau <= 0:
            raise ValueError("tau must be > 0")
        self.rng = np.random.RandomState(self.seed)
        self.W = self.rng.randn(self.d, self.r).astype(np.float64, copy=False)
        self.rsqrt_r = 1.0 / math.sqrt(self.r)
        self.sqrt_tau = math.sqrt(self.tau)

    def phi(self, x: np.ndarray) -> np.ndarray:
        # s_i = (w_i^T x)/sqrt(tau) - ||x||^2/(2 tau)
        x = np.asarray(x, dtype=np.float64)
        s = (self.W.T @ x) / self.sqrt_tau - (x @ x) / (2.0 * self.tau)
        np.clip(s, -self.clip, self.clip, out=s)
        s = np.exp(s, dtype=np.float64)
        return s * self.rsqrt_r

    def phi_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        S = (X @ self.W) / self.sqrt_tau - (np.sum(X * X, axis=1, keepdims=True) / (2.0 * self.tau))
        np.clip(S, -self.clip, self.clip, out=S)
        S = np.exp(S, dtype=np.float64)
        return S * self.rsqrt_r


# -------------------- SERA core --------------------

@dataclass
class SERAConfig:
    d: int
    d_v: int
    r: int
    tau: float = None
    gamma: float = 1.0
    lam: float = 1e-3
    clip: float = 40.0
    seed: int = 0

    # Denominator accumulator dtype: longdouble reduces drift in z accumulation.
    den_dtype: any = np.longdouble

    # Performance knobs (names preserved; new fields are additive)
    lazy_decay: bool = True              # avoid O(r*d_v) scaling per token when gamma!=1
    decay_renorm_min: float = 1e-6       # renormalize scale if g gets too small
    decay_renorm_max: float = 1e+6       # renormalize scale if g gets too large
    vectorized_ingest: bool = True       # use phi_batch + GEMM when gamma==1


class SERA:
    """
    State:
      - Z: (r, d_v) sufficient statistic for numerator
      - z: (r,) sufficient statistic for denominator
      - optional lazy decay scale g: effective stats are (g*Z, g*z)
    """

    def __init__(self, cfg: SERAConfig):
        self.cfg = cfg
        if cfg.d <= 0 or cfg.d_v <= 0 or cfg.r <= 0:
            raise ValueError("d, d_v, r must be > 0")
        if cfg.tau is None:
            cfg.tau = math.sqrt(cfg.d)
        if cfg.tau <= 0:
            raise ValueError("tau must be > 0")
        if cfg.gamma <= 0:
            raise ValueError("gamma must be > 0")
        if cfg.lam < 0:
            raise ValueError("lam must be >= 0")

        self.feat = PosExpFeature(cfg.d, cfg.r, cfg.tau, cfg.clip, seed=cfg.seed)
        self.Z = KahanMat((cfg.r, cfg.d_v), dtype=np.float64)
        self.z = KahanVec(cfg.r, dtype=cfg.den_dtype)

        # lazy decay scale (effective stats = g * raw_stats)
        self._g = 1.0

    def _maybe_renorm_decay_scale(self):
        # Renormalize when g gets extreme to avoid underflow/overflow in (outer/g).
        if not self.cfg.lazy_decay:
            return
        g = self._g
        if g == 1.0:
            return
        if (g < self.cfg.decay_renorm_min) or (g > self.cfg.decay_renorm_max):
            # Apply g into raw stats, reset g=1.
            self.Z.scale(g)
            self.z.scale(g)
            self._g = 1.0

    def update(self, k_t: np.ndarray, v_t: np.ndarray):
        phi_t = self.feat.phi(k_t)  # float64
        v_t = np.asarray(v_t, dtype=np.float64)

        if self.cfg.gamma != 1.0:
            if self.cfg.lazy_decay:
                # raw: g <- gamma*g, then add outer/g to raw stats
                self._g *= float(self.cfg.gamma)
                self._maybe_renorm_decay_scale()
                invg = 1.0 / self._g
                self.Z.add(np.outer(phi_t, v_t) * invg)
                self.z.add(phi_t.astype(self.cfg.den_dtype) * invg)
            else:
                # exact scaling per step: O(r*d_v)
                self.Z.scale(self.cfg.gamma)
                self.z.scale(self.cfg.gamma)
                self.Z.add(np.outer(phi_t, v_t))
                self.z.add(phi_t.astype(self.cfg.den_dtype))
        else:
            self.Z.add(np.outer(phi_t, v_t))
            self.z.add(phi_t.astype(self.cfg.den_dtype))

    def ingest_sequence(self, K: np.ndarray, V: np.ndarray):
        """
        Ingest sequence.
        Fast path:
          - if gamma==1 and vectorized_ingest: use phi_batch + GEMM + sum
        """
        K = np.asarray(K, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        if K.ndim != 2 or V.ndim != 2:
            raise ValueError("K and V must be 2D arrays")
        if K.shape[0] != V.shape[0]:
            raise ValueError("K and V must have same length")
        if K.shape[1] != self.cfg.d or V.shape[1] != self.cfg.d_v:
            raise ValueError("K/V shape mismatch with config")

        if (self.cfg.gamma == 1.0) and self.cfg.vectorized_ingest:
            # One-shot build:
            #   Z = Phi^T V
            #   z = sum(Phi, axis=0)
            Phi = self.feat.phi_batch(K)  # (n, r)
            Z0 = Phi.T @ V                # (r, d_v)
            z0 = Phi.sum(axis=0)          # (r,)
            self.Z.set_value(Z0.astype(np.float64, copy=False))
            self.z.set_value(z0.astype(self.cfg.den_dtype, copy=False))
            self._g = 1.0
            return

        # streaming path
        for t in range(K.shape[0]):
            self.update(K[t], V[t])

    def query(self, q: np.ndarray) -> np.ndarray:
        # y_hat = (phi(q)^T (g*Z_raw)) / (phi(q)^T (g*z_raw) + lambda)
        phi_q = self.feat.phi(q).astype(self.cfg.den_dtype, copy=False)  # (r,)

        Zraw = self.Z.value()
        zraw = self.z.value()

        # numerator: float64 is enough; keep dot stable by casting once
        num = (phi_q.astype(np.float64) @ Zraw).astype(np.float64, copy=False)  # (d_v,)

        g = float(self._g)
        den = float((phi_q @ zraw) * g) + float(self.cfg.lam)

        # Apply g to numerator last (keeps raw stats small under lazy decay)
        return (num * g) / den


# -------------------- Baseline & data --------------------

def softmax_attention_exact(q, K, V, tau, lam: float = 0.0):
    """
    Numerically-stable exact softmax attention in float64.
    Optional lam parity: add lam to denominator to match SERA's stabilized division.
    """
    q = np.asarray(q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    s = (K @ q) / float(tau)
    s = s - np.max(s)
    w = np.exp(s, dtype=np.float64)
    den = float(np.sum(w)) + float(lam) + 1e-300
    return (w[:, None] * V).sum(axis=0) / den


def make_sequence(n, d, d_v, rng, noise=0.0):
    K = rng.randn(n, d).astype(np.float64)
    K /= (np.linalg.norm(K, axis=1, keepdims=True) + 1e-12)
    V = rng.randn(n, d_v).astype(np.float64)
    if noise > 0:
        V += noise * rng.randn(n, d_v)
    return K, V


# -------------------- Experiments --------------------

def _write_csv(path: str, rows: List[Dict[str, float]], fieldnames: List[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def experiment_error_vs_r(
    outdir,
    n=1024,
    d=64,
    d_v=128,
    tau=None,
    r_list=(16, 32, 64, 128),
    gamma=1.0,
    lam=1e-3,
    seed=0,
    n_queries: int = 1,
    n_seeds: int = 1,
    lam_parity_exact: bool = False,
):
    """
    Error vs r.
    Upgrades:
      - multi-seed aggregation (n_seeds)
      - multi-query aggregation per seed (n_queries)
      - optional lam parity in exact baseline
    """
    os.makedirs(outdir, exist_ok=True)
    if tau is None:
        tau = math.sqrt(d)

    rows = []
    # Deterministic seed schedule
    seeds = [int(seed) + i for i in range(int(n_seeds))]

    for r in r_list:
        errs = []
        for sd in seeds:
            rng = np.random.RandomState(sd)
            K, V = make_sequence(n, d, d_v, rng)
            for _ in range(int(n_queries)):
                q = rng.randn(d).astype(np.float64)
                q /= (np.linalg.norm(q) + 1e-12)

                y_true = softmax_attention_exact(q, K, V, tau, lam=(lam if lam_parity_exact else 0.0))

                cfg = SERAConfig(d=d, d_v=d_v, r=int(r), tau=float(tau), gamma=float(gamma), lam=float(lam), seed=sd)
                model = SERA(cfg)
                model.ingest_sequence(K, V)
                y_hat = model.query(q)

                err = LA.norm(y_hat - y_true) / (LA.norm(y_true) + 1e-12)
                errs.append(float(err))

        errs = np.asarray(errs, dtype=np.float64)
        rows.append(
            {
                "r": float(r),
                "rel_l2_err_mean": float(errs.mean()),
                "rel_l2_err_p50": float(np.percentile(errs, 50.0, method="linear")),
                "rel_l2_err_p95": float(np.percentile(errs, 95.0, method="linear")),
                "n_eval": float(errs.size),
            }
        )

    csv_path = os.path.join(outdir, "sera_error_vs_r.csv")
    _write_csv(csv_path, rows, ["r", "rel_l2_err_mean", "rel_l2_err_p50", "rel_l2_err_p95", "n_eval"])

    xs = np.array([row["r"] for row in rows], dtype=np.float64)
    ys = np.array([row["rel_l2_err_mean"] for row in rows], dtype=np.float64)

    plt.figure()
    plt.loglog(xs, ys, marker="o", label="SERA (mean relative L2)")
    # reference slope line
    ref = ys[0] * np.sqrt(xs[0] / xs)
    plt.loglog(xs, ref, linestyle="--", label="~ r^{-1/2} reference")
    plt.xlabel("r (feature dimension)")
    plt.ylabel("relative L2 error")
    plt.title("SERA: error vs r")
    plt.legend()
    png_path = os.path.join(outdir, "sera_error_vs_r.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    return csv_path, png_path


def experiment_latency_vs_n(
    outdir,
    d=64,
    d_v=128,
    tau=None,
    r=128,
    n_list=(256, 512, 1024, 2048, 4096, 8192),
    gamma=1.0,
    lam=1e-3,
    seed=0,
    reps=200,
    warmup=5,
    y_max_query=None,
    lam_parity_exact: bool = False,
    vectorized_ingest: bool = True,
):
    """
    Latency experiment:
      - Single-thread BLAS
      - Warmup + repetitions
      - Ingest and query plotted separately

    Upgrades:
      - vectorized ingest option (phi_batch+GEMM) for gamma==1
      - optional lam parity in exact query
    """
    os.makedirs(outdir, exist_ok=True)
    if tau is None:
        tau = math.sqrt(d)

    rng = np.random.RandomState(seed)

    rows = []
    with single_thread_blas():
        for n in n_list:
            # Outside timed region: data generation & query selection
            K, V = make_sequence(int(n), int(d), int(d_v), rng)
            q = rng.randn(d).astype(np.float64)
            q /= (np.linalg.norm(q) + 1e-12)

            # Ingest timing — build state from scratch each call
            def _ingest_once():
                cfg = SERAConfig(
                    d=d,
                    d_v=d_v,
                    r=r,
                    tau=tau,
                    gamma=gamma,
                    lam=lam,
                    seed=seed,
                    vectorized_ingest=vectorized_ingest,
                )
                m = SERA(cfg)
                m.ingest_sequence(K, V)

            ingest_stats = bench_ms(_ingest_once, reps=reps, warmup=warmup)

            # Build once for query timing
            cfg = SERAConfig(
                d=d,
                d_v=d_v,
                r=r,
                tau=tau,
                gamma=gamma,
                lam=lam,
                seed=seed,
                vectorized_ingest=vectorized_ingest,
            )
            model = SERA(cfg)
            model.ingest_sequence(K, V)

            sera_stats = bench_ms(lambda: model.query(q), reps=reps, warmup=warmup)
            exact_stats = bench_ms(
                lambda: softmax_attention_exact(q, K, V, tau, lam=(lam if lam_parity_exact else 0.0)),
                reps=reps,
                warmup=warmup,
            )

            rows.append(
                {
                    "n": float(n),
                    "sera_ing_p50_ms": ingest_stats["p50"],
                    "sera_ing_p95_ms": ingest_stats["p95"],
                    "sera_ing_p99_ms": ingest_stats["p99"],
                    "sera_q_p50_ms": sera_stats["p50"],
                    "sera_q_p95_ms": sera_stats["p95"],
                    "sera_q_p99_ms": sera_stats["p99"],
                    "exact_q_p50_ms": exact_stats["p50"],
                    "exact_q_p95_ms": exact_stats["p95"],
                    "exact_q_p99_ms": exact_stats["p99"],
                }
            )

    csv_path = os.path.join(outdir, "sera_latency_vs_n.csv")
    fieldnames = list(rows[0].keys())
    _write_csv(csv_path, rows, fieldnames)

    xs = np.array([r["n"] for r in rows], dtype=np.float64)

    # Plot 1: Ingest
    ing = np.array([r["sera_ing_p50_ms"] for r in rows], dtype=np.float64)
    plt.figure()
    plt.plot(xs, ing, marker="o", label="SERA ingest (median)")
    plt.xlabel("n (sequence length)")
    plt.ylabel("time (ms)")
    plt.title("Ingest vs n (SERA build state)")
    plt.legend()
    png_ing = os.path.join(outdir, "sera_ingest_vs_n.png")
    plt.savefig(png_ing, bbox_inches="tight")
    plt.close()

    # Plot 2: Query
    sera_q = np.array([r["sera_q_p50_ms"] for r in rows], dtype=np.float64)
    exact_q = np.array([r["exact_q_p50_ms"] for r in rows], dtype=np.float64)
    plt.figure()
    plt.plot(xs, sera_q, marker="o", label="SERA query (median, O(1))")
    plt.plot(xs, exact_q, marker="o", label="Exact softmax query (median, O(n))")
    plt.xlabel("n (sequence length)")
    plt.ylabel("time (ms)")
    plt.title("Query vs n (median over repetitions)")
    plt.legend()
    if y_max_query is not None:
        plt.ylim(0, float(y_max_query))
    png_q = os.path.join(outdir, "sera_query_vs_n.png")
    plt.savefig(png_q, bbox_inches="tight")
    plt.close()

    return csv_path, png_ing, png_q


def experiment_check(outdir, d=32, d_v=64, tau=None, r=64, n=256, gamma=1.0, lam=1e-3, seed=1):
    """Sanity: if V is constant, both SERA and exact should return ~ that constant."""
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.RandomState(seed)
    if tau is None:
        tau = math.sqrt(d)

    K = rng.randn(n, d).astype(np.float64)
    K /= (np.linalg.norm(K, axis=1, keepdims=True) + 1e-12)
    v_const = rng.randn(d_v).astype(np.float64)
    V = np.tile(v_const[None, :], (n, 1))

    cfg = SERAConfig(d=d, d_v=d_v, r=r, tau=tau, gamma=gamma, lam=lam, seed=seed)
    model = SERA(cfg)
    model.ingest_sequence(K, V)

    q = rng.randn(d).astype(np.float64)
    q /= (np.linalg.norm(q) + 1e-12)
    y = model.query(q)

    const_err = LA.norm(y - v_const) / (LA.norm(v_const) + 1e-12)

    csv_path = os.path.join(outdir, "sera_checks.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["const_err"])
        writer.writeheader()
        writer.writerow({"const_err": float(const_err)})
    return csv_path


# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(description="SERA reference implementation + robust experiments")
    sub = parser.add_subparsers(dest="cmd")

    p_all = sub.add_parser("all")
    p_all.add_argument("--outdir", type=str, default="./sera_runs")

    p1 = sub.add_parser("err_r")
    p1.add_argument("--outdir", type=str, default="./sera_runs")
    p1.add_argument("--n", type=int, default=1024)
    p1.add_argument("--d", type=int, default=64)
    p1.add_argument("--d_v", type=int, default=128)
    p1.add_argument("--r_list", type=int, nargs="+", default=[16, 32, 64, 128])
    p1.add_argument("--gamma", type=float, default=1.0)
    p1.add_argument("--lam", type=float, default=1e-3)
    p1.add_argument("--seed", type=int, default=0)
    # upgrades
    p1.add_argument("--n_queries", type=int, default=1)
    p1.add_argument("--n_seeds", type=int, default=1)
    p1.add_argument("--lam_parity_exact", action="store_true")

    p2 = sub.add_parser("lat")
    p2.add_argument("--outdir", type=str, default="./sera_runs")
    p2.add_argument("--d", type=int, default=64)
    p2.add_argument("--d_v", type=int, default=128)
    p2.add_argument("--r", type=int, default=128)
    p2.add_argument("--n_list", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192])
    p2.add_argument("--gamma", type=float, default=1.0)
    p2.add_argument("--lam", type=float, default=1e-3)
    p2.add_argument("--seed", type=int, default=0)
    p2.add_argument("--reps", type=int, default=200)
    p2.add_argument("--warmup", type=int, default=5)
    p2.add_argument("--y_max_query", type=float, default=None)
    # upgrades
    p2.add_argument("--lam_parity_exact", action="store_true")
    p2.add_argument("--no_vectorized_ingest", action="store_true")

    p3 = sub.add_parser("check")
    p3.add_argument("--outdir", type=str, default="./sera_runs")

    args = parser.parse_args()
    outdir = getattr(args, "outdir", "./sera_runs")
    os.makedirs(outdir, exist_ok=True)

    if args.cmd == "err_r":
        c, p = experiment_error_vs_r(
            outdir=outdir,
            n=args.n,
            d=args.d,
            d_v=args.d_v,
            r_list=tuple(args.r_list),
            gamma=args.gamma,
            lam=args.lam,
            seed=args.seed,
            n_queries=args.n_queries,
            n_seeds=args.n_seeds,
            lam_parity_exact=bool(args.lam_parity_exact),
        )
        print(c)
        print(p)
    elif args.cmd == "lat":
        c, p_ing, p_q = experiment_latency_vs_n(
            outdir=outdir,
            d=args.d,
            d_v=args.d_v,
            r=args.r,
            n_list=tuple(args.n_list),
            gamma=args.gamma,
            lam=args.lam,
            seed=args.seed,
            reps=args.reps,
            warmup=args.warmup,
            y_max_query=args.y_max_query,
            lam_parity_exact=bool(args.lam_parity_exact),
            vectorized_ingest=(not args.no_vectorized_ingest),
        )
        print(c)
        print(p_ing)
        print(p_q)
    elif args.cmd == "check":
        c = experiment_check(outdir=outdir)
        print(c)
    else:  # "all" or None
        c1, p1_png = experiment_error_vs_r(outdir=outdir, n_seeds=5, n_queries=3)
        c2, p_ing, p_q = experiment_latency_vs_n(outdir=outdir)
        c3 = experiment_check(outdir=outdir)
        print(c1)
        print(p1_png)
        print(c2)
        print(p_ing)
        print(p_q)
        print(c3)


if __name__ == "__main__":
    main()
