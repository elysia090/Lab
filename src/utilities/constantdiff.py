#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Constantdiff — Constant-Time Differentiation Templates (CCR-U Minimal Implementation).

This module provides a self-contained implementation of the constant-size
template machinery used for the CCR-U experiments.  It features:

* Construction of constant-size templates on a three-patch cover of S^1 using
  a real Fourier band limit ``K``.
* O(1) forward- and reverse-mode automatic differentiation kernels for
  ``y = R0 ∘ Φ ∘ J_c ∘ L_patch``.
* A curvature identity unit test enforcing ``D_R^2 = 0`` on the Čech slice
  ``q:0→2`` using ``R1 := − (δ1 R0) · (δ0 + R0)^+``.
* Gamma-style certification helper routines and Neumann truncation order
  utilities.
* A minimal CLI providing quick access to curvature checks, certification
  numbers, differentiation demos, and template persistence.

The implementation depends only on NumPy and is intentionally compact so it
can be embedded directly into research notebooks or lightweight scripts.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, Tuple

import numpy as np
from numpy.linalg import norm, pinv


# ------------------------------
# S^1 real Fourier basis (band K)
# ------------------------------
def make_s1_basis(K: int) -> Tuple[int, np.ndarray]:
    """Return ``(d, LX)`` where ``d = 1 + 2K`` and ``LX`` is ``d/dθ`` in this basis."""

    d = 1 + 2 * K
    LX = np.zeros((d, d), dtype=float)
    for n in range(1, K + 1):
        idx_c = n  # cos(nθ)
        idx_s = K + n  # sin(nθ)
        LX[idx_s, idx_c] = -n  # d/dθ cos = -n sin
        LX[idx_c, idx_s] = n  # d/dθ sin = n cos
    return d, LX


# ------------------------------
# Čech structure (3 patches)
# ------------------------------
EDGES = [(0, 1), (0, 2), (1, 2)]
TRI = [(0, 1, 2)]
EIDX = {e: i for i, e in enumerate(EDGES)}


def build_E_S(d: int, P: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Return the ``E`` and ``S`` operators for the three-patch cover."""

    E = np.zeros((len(EDGES) * d, P * d))
    S = np.zeros_like(E)
    for (i, j), idx in EIDX.items():
        rs = slice(idx * d, (idx + 1) * d)
        E[rs, i * d : (i + 1) * d] += np.eye(d)
        E[rs, j * d : (j + 1) * d] += -np.eye(d)
        S[rs, i * d : (i + 1) * d] += np.eye(d)
        S[rs, j * d : (j + 1) * d] += np.eye(d)
    return E, S


def build_delta0(d: int, P: int = 3) -> np.ndarray:
    """Construct ``δ0 : C^0 → C^1`` mapping each edge to ``ψ_j − ψ_i``."""

    D0 = np.zeros((len(EDGES) * d, P * d))
    for (i, j), idx in EIDX.items():
        rs = slice(idx * d, (idx + 1) * d)
        D0[rs, j * d : (j + 1) * d] += np.eye(d)
        D0[rs, i * d : (i + 1) * d] += -np.eye(d)
    return D0


def build_delta1(d: int) -> np.ndarray:
    """Construct ``δ1 : C^1 → C^2`` for the unique Čech triangle."""

    D1 = np.zeros((1 * d, len(EDGES) * d))
    (i, j, k) = TRI[0]

    def eidx(a: int, b: int) -> int:
        a, b = min(a, b), max(a, b)
        return EIDX[(a, b)]

    ejk = eidx(j, k)
    eik = eidx(i, k)
    eij = eidx(i, j)
    D1[0:d, ejk * d : (ejk + 1) * d] += np.eye(d)
    D1[0:d, eik * d : (eik + 1) * d] += -np.eye(d)
    D1[0:d, eij * d : (eij + 1) * d] += np.eye(d)
    return D1


# ------------------------------
# Template builder
# ------------------------------
def build_templates(
    K: int = 3,
    alpha: float = 1.0,
    beta_vals: Tuple[float, float, float] = (1.2, 0.8, 1.5),
    Jc_scales: Tuple[float, float, float] = (1.1, 0.9, 1.05),
) -> Dict[str, np.ndarray]:
    """Build constant-size templates for the specified parameters."""

    d, LX = make_s1_basis(K)
    P = 3
    E = len(EDGES)
    L_patch = np.kron(np.eye(P), LX)
    L_edge = np.kron(np.eye(E), LX)
    E_mat, S_mat = build_E_S(d, P)
    D0 = build_delta0(d, P)
    D1 = build_delta1(d)

    B_edge = np.zeros((E * d, E * d))
    for (i, j), idx in EIDX.items():
        _ = (i, j)  # Unused but documents the tuple unpacking for clarity.
        B_edge[idx * d : (idx + 1) * d, idx * d : (idx + 1) * d] = float(
            beta_vals[idx]
        ) * np.eye(d)

    R0 = alpha * E_mat + (L_edge @ (B_edge @ S_mat))

    A = D0 + R0
    C = D1 @ R0
    R1 = -C @ pinv(A)

    Jc = np.kron(np.diag(np.array(Jc_scales, dtype=float)), np.eye(d))

    return dict(
        K=K,
        d=d,
        P=P,
        E=E,
        LX=LX,
        L_patch=L_patch,
        Jc=Jc,
        R0=R0,
        R1=R1,
        delta0=D0,
        delta1=D1,
    )


# ------------------------------
# O(1) AD kernels
# ------------------------------
def assemble_Dphi(preact: np.ndarray) -> np.ndarray:
    """Return the diagonal Jacobian of the ``tanh`` activation."""

    z = np.tanh(preact)
    dz = 1.0 - z * z
    return np.diag(dz)


def jvp(tpl: Dict[str, np.ndarray], psi_patch_vec: np.ndarray, v_patch_vec: np.ndarray) -> np.ndarray:
    """Jacobian-vector product for the template map."""

    Lb = tpl["L_patch"]
    Jc = tpl["Jc"]
    R0 = tpl["R0"]
    u = Jc @ (Lb @ psi_patch_vec)
    Dp = assemble_Dphi(u)
    return R0 @ (Dp @ (Jc @ (Lb @ v_patch_vec)))


def vjp(tpl: Dict[str, np.ndarray], psi_patch_vec: np.ndarray, w_edge_vec: np.ndarray) -> np.ndarray:
    """Vector-Jacobian product (adjoint) for the template map."""

    Lb = tpl["L_patch"]
    Jc = tpl["Jc"]
    R0 = tpl["R0"]
    u = Jc @ (Lb @ psi_patch_vec)
    Dp = assemble_Dphi(u)
    return Lb.T @ (Jc.T @ (Dp.T @ (R0.T @ w_edge_vec)))


# ------------------------------
# Curvature test (q:0→2) for D_R^2=0 on Čech slice
# ------------------------------
def curvature_residual(tpl: Dict[str, np.ndarray], psi_patch_vec: np.ndarray) -> np.ndarray:
    """Return the curvature residual ``δ(R0 ψ) + R1(δ ψ) + R1(R0 ψ)``."""

    D1 = tpl["delta1"]
    R1 = tpl["R1"]
    R0 = tpl["R0"]
    D0 = tpl["delta0"]
    return (D1 @ (R0 @ psi_patch_vec)) + (R1 @ (D0 @ psi_patch_vec)) + (R1 @ (R0 @ psi_patch_vec))


def curvature_test(tpl: Dict[str, np.ndarray], trials: int = 200, seed: int = 0) -> Tuple[float, float]:
    """Monte Carlo test for the curvature residual norm statistics."""

    rng = np.random.default_rng(seed)
    P = tpl["P"]
    d = tpl["d"]
    mx = 0.0
    av = 0.0
    for _ in range(trials):
        psi = rng.normal(size=(P * d,))
        r = curvature_residual(tpl, psi)
        n = norm(r)
        mx = max(mx, n)
        av += n
    av /= trials
    return mx, av


# ------------------------------
# Gamma certification and Neumann truncation
# ------------------------------
def op_norm_2(A: np.ndarray) -> float:
    """Return the spectral norm (σₘₐₓ)."""

    return np.linalg.svd(A, compute_uv=False)[0]


def certification_numbers(tpl: Dict[str, np.ndarray], alpha: float, beta_max: float, h_norm: float) -> Dict[str, float]:
    """Compute helper constants used in the gamma-style certification."""

    d = tpl["d"]
    P = tpl["P"]
    E_mat, S_mat = build_E_S(d, P)
    L_patch = tpl["L_patch"]
    bE = op_norm_2(E_mat)
    bS = op_norm_2(S_mat)
    bL = op_norm_2(L_patch)
    gamma = h_norm * (alpha * bE + beta_max * bS * bL)
    return dict(bE=bE, bS=bS, bL=bL, gamma=gamma)


def truncation_order(eps: float, gamma: float) -> int | None:
    """Return the minimal ``m`` with tail ``≤ eps`` for the Neumann series."""

    if not (0.0 < gamma < 1.0):
        return None
    m = math.ceil(math.log(eps * (1.0 - gamma), gamma)) - 1
    return max(0, m)


# ------------------------------
# CLI
# ------------------------------
def main() -> None:
    """Entry point for the minimal command-line interface."""

    ap = argparse.ArgumentParser(description="Constant-Time Differentiation Minimal Kernel (CCR-U)")
    ap.add_argument("--K", type=int, default=3, help="Fourier band (default: 3)")
    ap.add_argument("--alpha", type=float, default=1.0, help="alpha (Robin)")
    ap.add_argument("--betas", type=float, nargs=3, default=(1.2, 0.8, 1.5), help="beta_01 beta_02 beta_12")
    ap.add_argument(
        "--jcs",
        type=float,
        nargs=3,
        default=(1.1, 0.9, 1.05),
        help="degree-0 incidence scaling per patch",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("curvature", help="Run curvature unit test (q:0→2)")
    sp_cert = sub.add_parser("cert", help="Gamma-style certification numbers")
    sp_cert.add_argument("--beta-max", type=float, default=1.5)
    sp_cert.add_argument("--h-norm", type=float, default=0.05)
    sp_cert.add_argument("--eps", type=float, default=1e-6, help="target truncation error")

    sp_jvp = sub.add_parser("jvp", help="Run a demo JVP/VJP and report inner-product match")
    sp_jvp.add_argument("--seed", type=int, default=0)

    sp_save = sub.add_parser("save", help="Save templates to NPZ")
    sp_save.add_argument("--out", type=str, default="o1_templates.npz")

    ap.add_argument("--trials", type=int, default=200, help="trials for curvature")
    ap.add_argument("--seed", type=int, default=0, help="rng seed")

    args = ap.parse_args()
    tpl = build_templates(
        K=args.K,
        alpha=args.alpha,
        beta_vals=tuple(args.betas),
        Jc_scales=tuple(args.jcs),
    )

    if args.cmd == "curvature":
        mx, av = curvature_test(tpl, trials=args.trials, seed=args.seed)
        print(f"[Curvature q:0→2] max_residual={mx:.3e}  mean_residual={av:.3e}")

    elif args.cmd == "cert":
        cert = certification_numbers(tpl, alpha=args.alpha, beta_max=args.beta_max, h_norm=args.h_norm)
        gamma = cert["gamma"]
        m_eps = truncation_order(args.eps, gamma)
        print(f"[Certification] bE={cert['bE']:.3f}  bS={cert['bS']:.3f}  bL={cert['bL']:.3f}")
        print(f"gamma = {gamma:.3f}   (must be < 1 for Neumann convergence)")
        if m_eps is None:
            print("Neumann truncation: not applicable (gamma ≥ 1).")
        else:
            print(f"Neumann truncation m for eps={args.eps:g}:  m = {m_eps}")
            tail = (gamma ** (m_eps + 1)) / (1.0 - gamma)
            print(f"Tail bound ≤ {tail:.3e}")

    elif args.cmd == "jvp":
        rng = np.random.default_rng(args.seed)
        P, d, E = tpl["P"], tpl["d"], tpl["E"]
        psi = rng.normal(size=(P * d,))
        v = rng.normal(size=(P * d,))
        w = rng.normal(size=(E * d,))
        dy = jvp(tpl, psi, v)
        g = vjp(tpl, psi, w)
        lhs = float(np.dot(dy, w))
        rhs = float(np.dot(v, g))
        rel = abs(lhs - rhs) / (1e-12 + max(abs(lhs), abs(rhs)))
        print(f"[JVP/VJP]  <J v, w>={lhs:.6e}   <v, J^T w>={rhs:.6e}   rel_err={rel:.3e}")
        print(f"Shapes: psi={psi.shape}, v={v.shape}, y={dy.shape}, g={g.shape}")

    elif args.cmd == "save":
        np.savez(
            args.out,
            R0=tpl["R0"],
            R1=tpl["R1"],
            delta0=tpl["delta0"],
            delta1=tpl["delta1"],
            K=np.array([tpl["K"]]),
            d=np.array([tpl["d"]]),
            P=np.array([tpl["P"]]),
            E=np.array([tpl["E"]]),
        )
        print(f"Saved templates to {args.out}")


if __name__ == "__main__":
    main()

