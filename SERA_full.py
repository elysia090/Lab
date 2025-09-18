#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SERA: Self-normalized Exponential-kernel Ratio Attention
- Positive random features for softmax kernel
- Stateful sufficient statistics: Z (r x d_v), z (r,)
- Streaming decay gamma, stabilization lambda
- Kahan-compensated accumulation; optional longdouble denominator
- CLI experiments: error vs r, latency vs n, sanity check
"""

import os, sys, time, math, argparse, csv
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

# ---------- Numerics: Kahan compensated sums ----------

class KahanVec:
    def __init__(self, dim: int, dtype=np.float64):
        self.sum = np.zeros(dim, dtype=dtype)
        self.c = np.zeros(dim, dtype=dtype)
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
    def __init__(self, shape: Tuple[int,int], dtype=np.float64):
        self.sum = np.zeros(shape, dtype=dtype)
        self.c = np.zeros(shape, dtype=dtype)
    def add(self, X: np.ndarray):
        y = X - self.c
        t = self.sum + y
        self.c = (t - self.sum) - y
        self.sum = t
    def scale(self, gamma: float):
        self.sum *= gamma; self.c *= gamma
    def value(self):
        return self.sum

# ---------- Positive feature map for exp kernel ----------

@dataclass
class PosExpFeature:
    d: int; r: int; tau: float; clip: float = 40.0; seed: int = 0
    def __post_init__(self):
        self.rng = np.random.RandomState(self.seed)
        self.W = self.rng.randn(self.d, self.r).astype(np.float64)
    def phi(self, x: np.ndarray):
        s = (self.W.T @ x) / math.sqrt(self.tau) - (x @ x) / (2.0*self.tau)
        np.clip(s, -self.clip, self.clip, out=s)
        return np.exp(s, dtype=np.float64) / math.sqrt(self.r)
    def phi_batch(self, X: np.ndarray):
        S = (X @ self.W) / math.sqrt(self.tau) - (np.sum(X*X, axis=1, keepdims=True)/(2.0*self.tau))
        np.clip(S, -self.clip, self.clip, out=S)
        return np.exp(S, dtype=np.float64) / math.sqrt(self.r)

# ---------- SERA core ----------

@dataclass
class SERAConfig:
    d: int; d_v: int; r: int
    tau: float = None; gamma: float = 1.0; lam: float = 1e-3
    clip: float = 40.0; seed: int = 0
    den_dtype: any = np.longdouble  # high-precision denom

class SERA:
    def __init__(self, cfg: SERAConfig):
        self.cfg = cfg
        if cfg.tau is None: cfg.tau = math.sqrt(cfg.d)
        self.feat = PosExpFeature(cfg.d, cfg.r, cfg.tau, cfg.clip, seed=cfg.seed)
        self.Z = KahanMat((cfg.r, cfg.d_v), dtype=np.float64)
        self.z = KahanVec(cfg.r, dtype=cfg.den_dtype)

    def update(self, k_t: np.ndarray, v_t: np.ndarray):
        phi_t = self.feat.phi(k_t).astype(np.float64)
        if self.cfg.gamma != 1.0:
            self.Z.scale(self.cfg.gamma); self.z.scale(self.cfg.gamma)
        self.Z.add(np.outer(phi_t, v_t.astype(np.float64)))   # (r, d_v)
        self.z.add(phi_t.astype(self.cfg.den_dtype))          # (r,)

    def ingest_sequence(self, K: np.ndarray, V: np.ndarray):
        for t in range(K.shape[0]):
            self.update(K[t], V[t])

    def query(self, q: np.ndarray) -> np.ndarray:
        phi_q = self.feat.phi(q).astype(self.cfg.den_dtype)   # (r,)
        num = (phi_q.astype(np.float64) @ self.Z.value()).astype(np.float64)  # (d_v,)
        den = float(phi_q @ self.z.value()) + float(self.cfg.lam)
        return num / den

# ---------- Baseline & data ----------

def softmax_attention_exact(q, K, V, tau):
    s = (K @ q) / tau
    s = s - np.max(s)
    w = np.exp(s).astype(np.float64)
    den = np.sum(w) + 1e-300
    return (w[:,None]*V).sum(axis=0) / den

def make_sequence(n, d, d_v, rng, noise=0.0):
    K = rng.randn(n, d).astype(np.float64)
    K /= (np.linalg.norm(K, axis=1, keepdims=True)+1e-12)
    V = rng.randn(n, d_v).astype(np.float64)
    if noise>0: V += noise*rng.randn(n, d_v)
    return K, V

# ---------- Experiments ----------

def experiment_error_vs_r(outdir, n=1024, d=64, d_v=128, tau=None, r_list=(16,32,64,128), gamma=1.0, lam=1e-3, seed=0):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.RandomState(seed)
    if tau is None: tau = math.sqrt(d)
    K, V = make_sequence(n, d, d_v, rng)
    q = rng.randn(d).astype(np.float64); q /= (np.linalg.norm(q)+1e-12)
    y_true = softmax_attention_exact(q, K, V, tau)

    rows = []
    for r in r_list:
        cfg = SERAConfig(d=d, d_v=d_v, r=r, tau=tau, gamma=gamma, lam=lam, seed=seed)
        model = SERA(cfg)
        model.ingest_sequence(K, V)
        y_hat = model.query(q)
        err = LA.norm(y_hat - y_true) / (LA.norm(y_true)+1e-12)
        rows.append({"r": r, "rel_l2_err": float(err)})

    csv_path = os.path.join(outdir, "sera_error_vs_r.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["r","rel_l2_err"]); writer.writeheader()
        for row in rows: writer.writerow(row)

    # Plot
    xs = np.array([row["r"] for row in rows], dtype=np.float64)
    ys = np.array([row["rel_l2_err"] for row in rows], dtype=np.float64)
    plt.figure(); plt.loglog(xs, ys, marker="o")
    ref = ys[0]*np.sqrt(xs[0]/xs)
    plt.loglog(xs, ref, linestyle="--")
    plt.xlabel("r (feature dimension)"); plt.ylabel("relative L2 error"); plt.title("SERA: error vs r")
    png_path = os.path.join(outdir, "sera_error_vs_r.png")
    plt.savefig(png_path, bbox_inches="tight"); plt.close()
    return csv_path, png_path

def experiment_latency_vs_n(outdir, d=64, d_v=128, tau=None, r=128, n_list=(256,512,1024,2048), gamma=1.0, lam=1e-3, seed=0):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.RandomState(seed)
    if tau is None: tau = math.sqrt(d)

    rows = []
    for n in n_list:
        K, V = make_sequence(n, d, d_v, rng)

        cfg = SERAConfig(d=d, d_v=d_v, r=r, tau=tau, gamma=gamma, lam=lam, seed=seed)
        model = SERA(cfg)

        # ingest
        t0 = time.time(); model.ingest_sequence(K, V); t1 = time.time()
        sera_ing_ms = (t1 - t0)*1000.0

        # query
        q = rng.randn(d).astype(np.float64); q /= (np.linalg.norm(q)+1e-12)
        t0 = time.time(); _ = model.query(q); t1 = time.time()
        sera_q_ms = (t1 - t0)*1000.0

        # exact softmax query
        t0 = time.time(); _ = softmax_attention_exact(q, K, V, tau); t1 = time.time()
        exact_q_ms = (t1 - t0)*1000.0

        rows.append({"n": n, "sera_ingest_ms": sera_ing_ms, "sera_query_ms": sera_q_ms, "exact_query_ms": exact_q_ms})

    csv_path = os.path.join(outdir, "sera_latency_vs_n.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n","sera_ingest_ms","sera_query_ms","exact_query_ms"]); writer.writeheader()
        for row in rows: writer.writerow(row)

    # Plot
    xs = np.array([row["n"] for row in rows], dtype=np.float64)
    ys_i = np.array([row["sera_ingest_ms"] for row in rows], dtype=np.float64)
    ys_q = np.array([row["sera_query_ms"] for row in rows], dtype=np.float64)
    ys_e = np.array([row["exact_query_ms"] for row in rows], dtype=np.float64)
    plt.figure(); plt.plot(xs, ys_i, marker="o", label="SERA ingest (build state)")
    plt.plot(xs, ys_q, marker="o", label="SERA query (O(1))")
    plt.plot(xs, ys_e, marker="o", label="Exact softmax query (O(n))")
    plt.xlabel("n (sequence length)"); plt.ylabel("time (ms)"); plt.title("SERA latency vs n: breakdown")
    plt.legend()
    png_path = os.path.join(outdir, "sera_latency_vs_n.png"); plt.savefig(png_path, bbox_inches="tight"); plt.close()
    return csv_path, png_path

def experiment_check(outdir, d=32, d_v=64, tau=None, r=64, n=256, gamma=1.0, lam=1e-3, seed=1):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.RandomState(seed)
    if tau is None: tau = math.sqrt(d)
    # constant-value test
    K = rng.randn(n, d).astype(np.float64); K /= (np.linalg.norm(K, axis=1, keepdims=True)+1e-12)
    v_const = rng.randn(d_v).astype(np.float64)
    V = np.tile(v_const[None,:], (n,1))
    cfg = SERAConfig(d=d, d_v=d_v, r=r, tau=tau, gamma=gamma, lam=lam, seed=seed)
    model = SERA(cfg); model.ingest_sequence(K, V)
    q = rng.randn(d).astype(np.float64); q /= (np.linalg.norm(q)+1e-12)
    y = model.query(q)
    const_err = LA.norm(y - v_const) / (LA.norm(v_const)+1e-12)
    csv_path = os.path.join(outdir, "sera_checks.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["const_err"]); writer.writeheader(); writer.writerow({"const_err": float(const_err)})
    return csv_path

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="SERA reference implementation + experiments")
    sub = parser.add_subparsers(dest="cmd")
    p_all = sub.add_parser("all"); p_all.add_argument("--outdir", type=str, default="./sera_runs")
    p1 = sub.add_parser("err_r"); p1.add_argument("--outdir", type=str, default="./sera_runs")
    p2 = sub.add_parser("lat"); p2.add_argument("--outdir", type=str, default="./sera_runs")
    p3 = sub.add_parser("check"); p3.add_argument("--outdir", type=str, default="./sera_runs")
    args = parser.parse_args()
    outdir = getattr(args, "outdir", "./sera_runs")
    os.makedirs(outdir, exist_ok=True)
    if args.cmd == "err_r":
        c,p = experiment_error_vs_r(outdir=outdir); print(c); print(p)
    elif args.cmd == "lat":
        c,p = experiment_latency_vs_n(outdir=outdir); print(c); print(p)
    elif args.cmd == "check":
        c = experiment_check(outdir=outdir); print(c)
    elif args.cmd in ["all", None]:
        c1,p1 = experiment_error_vs_r(outdir=outdir)
        c2,p2 = experiment_latency_vs_n(outdir=outdir)
        c3 = experiment_check(outdir=outdir)
        print(c1); print(p1); print(c2); print(p2); print(c3)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

