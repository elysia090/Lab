#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SERA: Self-normalized Exponential-kernel Ratio Attention — reference + robust experiments

What this script provides
-------------------------
1) A clean SERA reference implementation with:
   - Positive random features for the exponential (softmax) kernel
   - Stateful sufficient statistics Z (r x d_v) and z (r,)
   - Streaming decay gamma and stabilization lambda
   - Kahan-compensated accumulation; optional longdouble denominator

2) Reproducible experiments with sound benchmarking:
   - Error vs r (shows ~ r^{-1/2} behavior)
   - Latency vs n with separated plots:
       * Ingest/build-state (O(n))
       * Query: SERA (O(1)) vs Exact softmax (O(n))
   - Sanity check (constant-value test)

Benchmarking hygiene
--------------------
- Uses perf_counter_ns(), warmup + repetitions, and single-thread BLAS
- Keeps randomness and allocations out of the timed region
- Saves CSVs and PNG figures under --outdir (default: ./sera_runs)

Usage
-----
  python sera.py all
  python sera.py err_r --r_list 16 32 64 128 --n 1024
  python sera.py lat --n_list 256 512 1024 2048 4096 8192 --reps 200 --warmup 5
  python sera.py check

No seaborn; matplotlib only. Figures are saved, not shown interactively.
"""

import os, math, argparse, csv, statistics, contextlib
from dataclasses import dataclass
from typing import Tuple, Callable, Dict

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


# -------------------- Determinism & BLAS control --------------------

@contextlib.contextmanager
def single_thread_blas():
    """Force single-thread BLAS to reduce noise on small/medium problems."""
    old_mkl = os.environ.get("MKL_NUM_THREADS")
    old_omp = os.environ.get("OMP_NUM_THREADS")
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"]  = "1"
    try:
        yield
    finally:
        if old_mkl is None: os.environ.pop("MKL_NUM_THREADS", None)
        else: os.environ["MKL_NUM_THREADS"] = old_mkl
        if old_omp is None: os.environ.pop("OMP_NUM_THREADS", None)
        else: os.environ["OMP_NUM_THREADS"] = old_omp


def bench_ms(fn: Callable[[], None], reps: int = 200, warmup: int = 5) -> Dict[str, float]:
    """
    Robust micro-benchmark:
      - warmup calls (not recorded)
      - reps timed with perf_counter_ns()
      - returns median, p95, p99, min, max (in ms)
    Assumes `fn` has no externally visible side effects across calls.
    """
    import time
    pc = time.perf_counter_ns
    for _ in range(warmup): fn()
    xs = []
    for _ in range(reps):
        t0 = pc(); fn(); t1 = pc()
        xs.append((t1 - t0) / 1e6)  # ms
    xs.sort()

    def q(p: int) -> float:
        k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
        return xs[k]

    return {
        "p50": float(np.median(xs)),
        "p95": q(95),
        "p99": q(99),
        "min": xs[0],
        "max": xs[-1],
    }


# -------------------- Numerics: Kahan compensated sums --------------------

class KahanVec:
    def __init__(self, dim: int, dtype=np.float64):
        self.sum = np.zeros(dim, dtype=dtype)
        self.c   = np.zeros(dim, dtype=dtype)
    def add(self, x: np.ndarray):
        y = x - self.c
        t = self.sum + y
        self.c = (t - self.sum) - y
        self.sum = t
    def scale(self, gamma: float):
        self.sum *= gamma; self.c *= gamma
    def value(self):
        return self.sum


class KahanMat:
    def __init__(self, shape: Tuple[int, int], dtype=np.float64):
        self.sum = np.zeros(shape, dtype=dtype)
        self.c   = np.zeros(shape, dtype=dtype)
    def add(self, X: np.ndarray):
        y = X - self.c
        t = self.sum + y
        self.c = (t - self.sum) - y
        self.sum = t
    def scale(self, gamma: float):
        self.sum *= gamma; self.c *= gamma
    def value(self):
        return self.sum


# -------------------- Positive feature map for exp kernel --------------------

@dataclass
class PosExpFeature:
    d: int
    r: int
    tau: float
    clip: float = 40.0
    seed: int   = 0

    def __post_init__(self):
        self.rng = np.random.RandomState(self.seed)
        self.W   = self.rng.randn(self.d, self.r).astype(np.float64)
        self.rsqrt_r  = 1.0 / math.sqrt(self.r)
        self.sqrt_tau = math.sqrt(self.tau)

    def phi(self, x: np.ndarray) -> np.ndarray:
        # s_i = w_i^T x / sqrt(tau) - ||x||^2/(2 tau)
        s = (self.W.T @ x) / self.sqrt_tau - (x @ x) / (2.0 * self.tau)
        np.clip(s, -self.clip, self.clip, out=s)
        s = np.exp(s, dtype=np.float64)
        return s * self.rsqrt_r

    def phi_batch(self, X: np.ndarray) -> np.ndarray:
        S = (X @ self.W) / self.sqrt_tau \
            - (np.sum(X * X, axis=1, keepdims=True) / (2.0 * self.tau))
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
    seed: int   = 0
    den_dtype: any = np.longdouble  # high-precision denom accumulator


class SERA:
    def __init__(self, cfg: SERAConfig):
        self.cfg = cfg
        if cfg.tau is None:
            cfg.tau = math.sqrt(cfg.d)
        self.feat = PosExpFeature(cfg.d, cfg.r, cfg.tau, cfg.clip, seed=cfg.seed)
        self.Z = KahanMat((cfg.r, cfg.d_v), dtype=np.float64)
        self.z = KahanVec(cfg.r, dtype=cfg.den_dtype)

    def update(self, k_t: np.ndarray, v_t: np.ndarray):
        phi_t = self.feat.phi(k_t).astype(np.float64)
        if self.cfg.gamma != 1.0:
            self.Z.scale(self.cfg.gamma); self.z.scale(self.cfg.gamma)
        self.Z.add(np.outer(phi_t, v_t.astype(np.float64)))           # (r, d_v)
        self.z.add(phi_t.astype(self.cfg.den_dtype))                  # (r,)

    def ingest_sequence(self, K: np.ndarray, V: np.ndarray):
        for t in range(K.shape[0]):
            self.update(K[t], V[t])

    def query(self, q: np.ndarray) -> np.ndarray:
        # y_hat = (phi(q)^T Z) / (phi(q)^T z + lambda)
        phi_q = self.feat.phi(q).astype(self.cfg.den_dtype)           # (r,)
        num   = (phi_q.astype(np.float64) @ self.Z.value()).astype(np.float64)  # (d_v,)
        den   = float(phi_q @ self.z.value()) + float(self.cfg.lam)
        return num / den


# -------------------- Baseline & data --------------------

def softmax_attention_exact(q, K, V, tau):
    """Numerically-stable exact softmax attention in float64."""
    s = (K @ q) / tau
    s = s - np.max(s)
    w = np.exp(s, dtype=np.float64)
    den = np.sum(w) + 1e-300
    return (w[:, None] * V).sum(axis=0) / den


def make_sequence(n, d, d_v, rng, noise=0.0):
    K = rng.randn(n, d).astype(np.float64)
    K /= (np.linalg.norm(K, axis=1, keepdims=True) + 1e-12)
    V = rng.randn(n, d_v).astype(np.float64)
    if noise > 0: V += noise * rng.randn(n, d_v)
    return K, V


# -------------------- Experiments --------------------

def experiment_error_vs_r(outdir, n=1024, d=64, d_v=128, tau=None,
                          r_list=(16, 32, 64, 128), gamma=1.0, lam=1e-3, seed=0):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.RandomState(seed)
    if tau is None: tau = math.sqrt(d)

    K, V = make_sequence(n, d, d_v, rng)
    q = rng.randn(d).astype(np.float64); q /= (np.linalg.norm(q) + 1e-12)
    y_true = softmax_attention_exact(q, K, V, tau)

    rows = []
    for r in r_list:
        cfg = SERAConfig(d=d, d_v=d_v, r=r, tau=tau, gamma=gamma, lam=lam, seed=seed)
        model = SERA(cfg); model.ingest_sequence(K, V)
        y_hat = model.query(q)
        err = LA.norm(y_hat - y_true) / (LA.norm(y_true) + 1e-12)
        rows.append({"r": r, "rel_l2_err": float(err)})

    csv_path = os.path.join(outdir, "sera_error_vs_r.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["r", "rel_l2_err"])
        writer.writeheader(); [writer.writerow(row) for row in rows]

    xs = np.array([row["r"] for row in rows], dtype=np.float64)
    ys = np.array([row["rel_l2_err"] for row in rows], dtype=np.float64)

    plt.figure()
    plt.loglog(xs, ys, marker="o", label="SERA (relative L2)")
    ref = ys[0] * np.sqrt(xs[0] / xs)
    plt.loglog(xs, ref, linestyle="--", label="~ r^{-1/2} reference")
    plt.xlabel("r (feature dimension)")
    plt.ylabel("relative L2 error")
    plt.title("SERA: error vs r")
    plt.legend()
    png_path = os.path.join(outdir, "sera_error_vs_r.png")
    plt.savefig(png_path, bbox_inches="tight"); plt.close()

    return csv_path, png_path


def experiment_latency_vs_n(outdir, d=64, d_v=128, tau=None, r=128,
                            n_list=(256, 512, 1024, 2048, 4096, 8192),
                            gamma=1.0, lam=1e-3, seed=0,
                            reps=200, warmup=5, y_max_query=None):
    """
    Robust latency experiment:
      - single-thread BLAS
      - warmup + repetitions
      - ingest and query plotted separately
    """
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.RandomState(seed)
    if tau is None: tau = math.sqrt(d)

    rows = []
    with single_thread_blas():
        for n in n_list:
            # Outside timed region: data generation & query selection
            K, V = make_sequence(n, d, d_v, rng)
            q = rng.randn(d).astype(np.float64); q /= (np.linalg.norm(q) + 1e-12)

            # Ingest timing — build state from scratch each time
            def _ingest_once():
                m = SERA(SERAConfig(d=d, d_v=d_v, r=r, tau=tau, gamma=gamma, lam=lam, seed=seed))
                m.ingest_sequence(K, V)
            ingest_stats = bench_ms(_ingest_once, reps=reps, warmup=warmup)

            # Build once for query timing
            model = SERA(SERAConfig(d=d, d_v=d_v, r=r, tau=tau, gamma=gamma, lam=lam, seed=seed))
            model.ingest_sequence(K, V)

            sera_stats  = bench_ms(lambda: model.query(q), reps=reps, warmup=warmup)
            exact_stats = bench_ms(lambda: softmax_attention_exact(q, K, V, tau), reps=reps, warmup=warmup)

            rows.append({
                "n": n,
                "sera_ing_p50_ms": ingest_stats["p50"],
                "sera_ing_p95_ms": ingest_stats["p95"],
                "sera_ing_p99_ms": ingest_stats["p99"],
                "sera_q_p50_ms":   sera_stats["p50"],
                "sera_q_p95_ms":   sera_stats["p95"],
                "sera_q_p99_ms":   sera_stats["p99"],
                "exact_q_p50_ms":  exact_stats["p50"],
                "exact_q_p95_ms":  exact_stats["p95"],
                "exact_q_p99_ms":  exact_stats["p99"],
            })

    # CSV
    csv_path = os.path.join(outdir, "sera_latency_vs_n.csv")
    with open(csv_path, "w", newline="") as f:
        fnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader(); [writer.writerow(rw) for rw in rows]

    # Plot 1: Ingest (O(n))
    xs = np.array([r["n"] for r in rows], dtype=np.float64)
    ing = np.array([r["sera_ing_p50_ms"] for r in rows], dtype=np.float64)
    plt.figure()
    plt.plot(xs, ing, marker="o", label="SERA ingest (median)")
    plt.xlabel("n (sequence length)"); plt.ylabel("time (ms)")
    plt.title("Ingest vs n (SERA build state)")
    plt.legend()
    png_ing = os.path.join(outdir, "sera_ingest_vs_n.png")
    plt.savefig(png_ing, bbox_inches="tight"); plt.close()

    # Plot 2: Query — SERA O(1) vs Exact O(n)
    sera_q  = np.array([r["sera_q_p50_ms"]  for r in rows], dtype=np.float64)
    exact_q = np.array([r["exact_q_p50_ms"] for r in rows], dtype=np.float64)
    plt.figure()
    plt.plot(xs, sera_q,  marker="o", label="SERA query (median, O(1))")
    plt.plot(xs, exact_q, marker="o", label="Exact softmax query (median, O(n))")
    plt.xlabel("n (sequence length)"); plt.ylabel("time (ms)")
    plt.title("Query vs n (median over repetitions)")
    plt.legend()
    if y_max_query is not None:
        plt.ylim(0, y_max_query)
    png_q = os.path.join(outdir, "sera_query_vs_n.png")
    plt.savefig(png_q, bbox_inches="tight"); plt.close()

    return csv_path, png_ing, png_q


def experiment_check(outdir, d=32, d_v=64, tau=None, r=64, n=256,
                     gamma=1.0, lam=1e-3, seed=1):
    """Sanity: if V is constant, both SERA and exact should return ~ that constant."""
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.RandomState(seed)
    if tau is None: tau = math.sqrt(d)

    K = rng.randn(n, d).astype(np.float64)
    K /= (np.linalg.norm(K, axis=1, keepdims=True) + 1e-12)
    v_const = rng.randn(d_v).astype(np.float64)
    V = np.tile(v_const[None, :], (n, 1))

    cfg = SERAConfig(d=d, d_v=d_v, r=r, tau=tau, gamma=gamma, lam=lam, seed=seed)
    model = SERA(cfg); model.ingest_sequence(K, V)

    q = rng.randn(d).astype(np.float64); q /= (np.linalg.norm(q) + 1e-12)
    y = model.query(q)
    const_err = LA.norm(y - v_const) / (LA.norm(v_const) + 1e-12)

    csv_path = os.path.join(outdir, "sera_checks.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["const_err"])
        writer.writeheader(); writer.writerow({"const_err": float(const_err)})
    return csv_path


# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(description="SERA reference implementation + robust experiments")
    sub = parser.add_subparsers(dest="cmd")

    p_all = sub.add_parser("all"); p_all.add_argument("--outdir", type=str, default="./sera_runs")

    p1 = sub.add_parser("err_r"); p1.add_argument("--outdir", type=str, default="./sera_runs")
    p1.add_argument("--n", type=int, default=1024)
    p1.add_argument("--d", type=int, default=64)
    p1.add_argument("--d_v", type=int, default=128)
    p1.add_argument("--r_list", type=int, nargs="+", default=[16, 32, 64, 128])
    p1.add_argument("--seed", type=int, default=0)

    p2 = sub.add_parser("lat"); p2.add_argument("--outdir", type=str, default="./sera_runs")
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

    p3 = sub.add_parser("check"); p3.add_argument("--outdir", type=str, default="./sera_runs")

    args = parser.parse_args()
    outdir = getattr(args, "outdir", "./sera_runs")
    os.makedirs(outdir, exist_ok=True)

    if args.cmd == "err_r":
        c, p = experiment_error_vs_r(outdir=outdir, n=args.n, d=args.d, d_v=args.d_v,
                                     r_list=tuple(args.r_list), seed=args.seed)
        print(c); print(p)
    elif args.cmd == "lat":
        c, p_ing, p_q = experiment_latency_vs_n(outdir=outdir, d=args.d, d_v=args.d_v, r=args.r,
                                                n_list=tuple(args.n_list), gamma=args.gamma, lam=args.lam,
                                                seed=args.seed, reps=args.reps, warmup=args.warmup,
                                                y_max_query=args.y_max_query)
        print(c); print(p_ing); print(p_q)
    elif args.cmd == "check":
        c = experiment_check(outdir=outdir); print(c)
    else:  # "all" or None
        c1, p1 = experiment_error_vs_r(outdir=outdir)
        c2, p_ing, p_q = experiment_latency_vs_n(outdir=outdir)
        c3 = experiment_check(outdir=outdir)
        print(c1); print(p1); print(c2); print(p_ing); print(p_q); print(c3)


if __name__ == "__main__":
    main()


