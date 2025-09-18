
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time, math, argparse, csv
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

# ---------------- Utilities ----------------

class KahanVec:
    def __init__(self, dim: int, dtype=np.float64):
        self.sum = np.zeros(dim, dtype=dtype)
        self.c = np.zeros(dim, dtype=dtype)
    def add(self, x):
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
    def add(self, X):
        y = X - self.c
        t = self.sum + y
        self.c = (t - self.sum) - y
        self.sum = t
    def scale(self, gamma: float):
        self.sum *= gamma; self.c *= gamma
    def value(self):
        return self.sum

# ------------- Positive exp features -------------

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
        S = (X @ self.W) / math.sqrt(self.tau) - (np.sum(X*X, axis=1, keepdims=True) / (2.0*self.tau))
        np.clip(S, -self.clip, self.clip, out=S)
        return np.exp(S, dtype=np.float64) / math.sqrt(self.r)

# ------------- SLUM core -------------

@dataclass
class SLUMConfig:
    d: int; d_v: int; r: int; r_v: int
    tau: float = None; gamma: float = 1.0; lam: float = 1e-3; mu: float = 1e-3
    clip: float = 40.0; basis_update_B: int = 200; seed: int = 0
    den_dtype: any = np.longdouble  # improved denominator precision

class SLUM:
    def __init__(self, cfg: SLUMConfig):
        self.cfg = cfg
        if cfg.tau is None: cfg.tau = math.sqrt(cfg.d)
        self.rng = np.random.RandomState(cfg.seed)
        self.feat = PosExpFeature(cfg.d, cfg.r, cfg.tau, cfg.clip, seed=cfg.seed)

        # Initialize value basis U (orthonormal columns)
        U0 = self.rng.randn(cfg.d_v, cfg.r_v)
        self.U, _ = LA.qr(U0)

        # Ridge inverse (U^TU + mu I)^{-1}
        self._update_ridge_matrix()

        # Sufficient stats: A (r x r_v), s (r)
        self.A = KahanMat((cfg.r, cfg.r_v), dtype=np.float64)
        self.s = KahanVec(cfg.r, dtype=cfg.den_dtype)

        # Buffers for basis update
        self._buf_V = []
        self._steps = 0

    def _update_ridge_matrix(self):
        UtU = self.U.T @ self.U
        self.G = LA.inv(UtU + self.cfg.mu * np.eye(self.cfg.r_v))

    def _project_coeff(self, v):
        # r_hat = (U^T U + mu I)^{-1} U^T v
        return self.G @ (self.U.T @ v)

    def update(self, k_t, v_t):
        phi_t = self.feat.phi(k_t).astype(np.float64)
        r_t = self._project_coeff(v_t.astype(np.float64))

        if self.cfg.gamma != 1.0:
            self.A.scale(self.cfg.gamma)
            self.s.scale(self.cfg.gamma)

        self.A.add(np.outer(phi_t, r_t))
        self.s.add(phi_t.astype(self.cfg.den_dtype))

        # buffer for basis update
        self._buf_V.append(v_t.astype(np.float64))
        if len(self._buf_V) > self.cfg.basis_update_B:
            self._buf_V.pop(0)

        self._steps += 1
        if self.cfg.basis_update_B > 0 and (self._steps % self.cfg.basis_update_B == 0):
            self._maybe_update_basis()

    def _maybe_update_basis(self):
        if len(self._buf_V) < self.cfg.r_v:
            return
        V = np.stack(self._buf_V, axis=1)  # (d_v, B)

        # SVD on buffer to refresh basis
        try:
            U_new, _, _ = LA.svd(V, full_matrices=False)
            U_new = U_new[:, :self.cfg.r_v]
        except LA.LinAlgError:
            return

        # Procrustes alignment: find orthogonal Q s.t. U_old Q ≈ U_new
        M = self.U.T @ U_new
        Uq, _, Vq = LA.svd(M, full_matrices=False)
        Q = Uq @ Vq  # orthogonal

        # Consistent state transform (see paper derivation):
        # With U' = U Q and r' = Q^T r, we must keep A' = A Q, so that
        # c' = A'^T φ = Q^T A^T φ and y' = U' c' = U A^T φ = y.
        self.A.sum = self.A.sum @ Q
        self.A.c   = self.A.c   @ Q
        self.U = self.U @ Q
        self._update_ridge_matrix()

    def query(self, q):
        phi_q = self.feat.phi(q).astype(self.cfg.den_dtype)  # high precision dot with s
        # num_small in float64 is fine; denominator in den_dtype
        num_small = (phi_q.astype(np.float64) @ self.A.value()).astype(np.float64)  # (r_v,)
        den = float(phi_q @ self.s.value()) + float(self.cfg.lam)
        y = self.U @ (num_small / den)
        return y

    def ingest_sequence(self, K, V):
        for t in range(K.shape[0]):
            self.update(K[t], V[t])

# ------------- Baselines / data -------------

def softmax_attention_exact(q, K, V, tau):
    s = (K @ q) / tau
    s = s - np.max(s)
    w = np.exp(s).astype(np.float64)
    den = np.sum(w) + 1e-300
    y = (w[:,None] * V).sum(axis=0) / den
    return y

def generate_lowrank_sequence(n, d, d_v, r_v, rng, noise=0.0):
    K = rng.randn(n, d).astype(np.float64)
    K /= (np.linalg.norm(K, axis=1, keepdims=True) + 1e-12)
    U_true, _ = LA.qr(rng.randn(d_v, r_v))
    R = rng.randn(n, r_v).astype(np.float64)
    V = (U_true @ R.T).T
    if noise > 0.0:
        V += noise * rng.randn(n, d_v)
    return K, V, U_true

# ------------- Experiments -------------

def experiment_error_vs_r(outdir, n=1024, d=64, d_v=128, tau=None, r_list=(32,64,128,256), r_v=16, gamma=1.0, lam=1e-3, mu=1e-3, seed=0):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.RandomState(seed)
    if tau is None: tau = math.sqrt(d)

    K, V, _ = generate_lowrank_sequence(n, d, d_v, r_v, rng)
    q = rng.randn(d).astype(np.float64); q /= (np.linalg.norm(q)+1e-12)
    y_true = softmax_attention_exact(q, K, V, tau)

    rows = []
    for r in r_list:
        cfg = SLUMConfig(d=d, d_v=d_v, r=r, r_v=r_v, tau=tau, gamma=gamma, lam=lam, mu=mu, basis_update_B=200, seed=seed)
        model = SLUM(cfg)
        model.ingest_sequence(K, V)
        y_hat = model.query(q)
        err = LA.norm(y_hat - y_true) / (LA.norm(y_true)+1e-12)
        rows.append({"r": r, "r_v": r_v, "rel_l2_err": float(err)})

    csv_path = os.path.join(outdir, "slum_error_vs_r.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["r","r_v","rel_l2_err"]); writer.writeheader()
        for row in rows: writer.writerow(row)

    xs = np.array([row["r"] for row in rows], dtype=np.float64)
    ys = np.array([row["rel_l2_err"] for row in rows], dtype=np.float64)
    plt.figure(); plt.loglog(xs, ys, marker="o")
    ref = ys[0] * np.sqrt(xs[0]/xs)
    plt.loglog(xs, ref, linestyle="--")
    plt.xlabel("r (feature dimension)"); plt.ylabel("relative L2 error"); plt.title(f"SLUM: error vs r (r_v={r_v})")
    png_path = os.path.join(outdir, "slum_error_vs_r.png"); plt.savefig(png_path, bbox_inches="tight"); plt.close()
    return csv_path, png_path

def experiment_error_vs_rv(outdir, n=1024, d=64, d_v=128, tau=None, r=256, r_v_list=(4,8,16,32), gamma=1.0, lam=1e-3, mu=1e-3, seed=0):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.RandomState(seed)
    if tau is None: tau = math.sqrt(d)

    K, V, _ = generate_lowrank_sequence(n, d, d_v, max(r_v_list), rng)
    q = rng.randn(d).astype(np.float64); q /= (np.linalg.norm(q)+1e-12)
    y_true = softmax_attention_exact(q, K, V, tau)

    rows = []
    for r_v in r_v_list:
        cfg = SLUMConfig(d=d, d_v=d_v, r=r, r_v=r_v, tau=tau, gamma=gamma, lam=lam, mu=mu, basis_update_B=200, seed=seed)
        model = SLUM(cfg)
        model.ingest_sequence(K, V)
        y_hat = model.query(q)
        err = LA.norm(y_hat - y_true) / (LA.norm(y_true)+1e-12)
        rows.append({"r": r, "r_v": r_v, "rel_l2_err": float(err)})

    csv_path = os.path.join(outdir, "slum_error_vs_rv.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["r","r_v","rel_l2_err"]); writer.writeheader()
        for row in rows: writer.writerow(row)

    xs = np.array([row["r_v"] for row in rows], dtype=np.float64)
    ys = np.array([row["rel_l2_err"] for row in rows], dtype=np.float64)
    plt.figure(); plt.loglog(xs, ys, marker="o")
    plt.xlabel("r_v (value rank)"); plt.ylabel("relative L2 error"); plt.title(f"SLUM: error vs r_v (r={r})")
    png_path = os.path.join(outdir, "slum_error_vs_rv.png"); plt.savefig(png_path, bbox_inches="tight"); plt.close()
    return csv_path, png_path

def experiment_latency_vs_n(outdir, d=64, d_v=128, tau=None, r=256, r_v=16, n_list=(512,1024,2048,4096,8192), gamma=1.0, lam=1e-3, mu=1e-3, seed=0):
    """Fair breakdown: SLUM ingest_only, SLUM query_only, Exact softmax query."""
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.RandomState(seed)
    if tau is None: tau = math.sqrt(d)

    rows = []
    for n in n_list:
        K, V, _ = generate_lowrank_sequence(n, d, d_v, r_v, rng)

        cfg = SLUMConfig(d=d, d_v=d_v, r=r, r_v=r_v, tau=tau, gamma=gamma, lam=lam, mu=mu, basis_update_B=200, seed=seed)
        model = SLUM(cfg)

        # Ingest only
        t0 = time.time(); model.ingest_sequence(K, V); t1 = time.time()
        slum_ingest_ms = (t1 - t0) * 1000.0

        # Query only (state already built)
        q = rng.randn(d).astype(np.float64); q /= (np.linalg.norm(q)+1e-12)
        t0 = time.time(); _ = model.query(q); t1 = time.time()
        slum_query_ms = (t1 - t0) * 1000.0

        # Exact softmax query
        t0 = time.time(); _ = softmax_attention_exact(q, K, V, tau); t1 = time.time()
        exact_query_ms = (t1 - t0) * 1000.0

        rows.append({"n": n, "slum_ingest_ms": slum_ingest_ms, "slum_query_ms": slum_query_ms, "exact_query_ms": exact_query_ms})

    csv_path = os.path.join(outdir, "slum_latency_vs_n.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n","slum_ingest_ms","slum_query_ms","exact_query_ms"]); writer.writeheader()
        for row in rows: writer.writerow(row)

    # Plot: per requirement, single-figure, no explicit colors
    xs = np.array([row["n"] for row in rows], dtype=np.float64)
    ys_ing = np.array([row["slum_ingest_ms"] for row in rows], dtype=np.float64)
    ys_q = np.array([row["slum_query_ms"] for row in rows], dtype=np.float64)
    ys_ex = np.array([row["exact_query_ms"] for row in rows], dtype=np.float64)

    plt.figure(); plt.plot(xs, ys_ing, marker="o", label="SLUM ingest (build state)")
    plt.plot(xs, ys_q, marker="o", label="SLUM query (O(1))")
    plt.plot(xs, ys_ex, marker="o", label="Exact softmax query")
    plt.xlabel("n (sequence length)"); plt.ylabel("time (ms)"); plt.title("Latency vs n: breakdown")
    plt.legend()
    png_path = os.path.join(outdir, "slum_latency_vs_n.png"); plt.savefig(png_path, bbox_inches="tight"); plt.close()

    return csv_path, png_path

# ------------- CLI -------------

def main():
    parser = argparse.ArgumentParser(description="SLUM: reference implementation + experiments (fixed)")
    sub = parser.add_subparsers(dest="cmd")
    p_all = sub.add_parser("all"); p_all.add_argument("--outdir", type=str, default="./slum_runs")
    p1 = sub.add_parser("err_r"); p1.add_argument("--outdir", type=str, default="./slum_runs")
    p2 = sub.add_parser("err_rv"); p2.add_argument("--outdir", type=str, default="./slum_runs")
    p3 = sub.add_parser("lat"); p3.add_argument("--outdir", type=str, default="./slum_runs")
    p4 = sub.add_parser("check"); p4.add_argument("--outdir", type=str, default="./slum_runs")
    args = parser.parse_args()
    outdir = getattr(args, "outdir", "./slum_runs")
    os.makedirs(outdir, exist_ok=True)
    if args.cmd == "err_r":
        c,p = experiment_error_vs_r(outdir=outdir); print(c); print(p)
    elif args.cmd == "err_rv":
        c,p = experiment_error_vs_rv(outdir=outdir); print(c); print(p)
    elif args.cmd == "lat":
        c,p = experiment_latency_vs_n(outdir=outdir); print(c); print(p)
    elif args.cmd == "check":
        c = experiment_error_vs_r(outdir=outdir); print(c)  # simple smoke
    elif args.cmd in ["all", None]:
        c1,p1 = experiment_error_vs_r(outdir=outdir)
        c2,p2 = experiment_error_vs_rv(outdir=outdir)
        c3,p3 = experiment_latency_vs_n(outdir=outdir)
        print(c1); print(p1); print(c2); print(p2); print(c3); print(p3)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

