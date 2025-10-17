
# lowrank_o1.py
# O(1) Low-Rank Update Kernel (rank-k), CCR-Compatible
# Author: ChatGPT (GPT-5 Thinking)
# License: MIT
"""
O(1) Low-Rank Update Kernel (rank-k)

This module implements the constant-time (O(1) w.r.t. ambient n) online queries
for matrices subject to a fixed-rank update A_u := A + U V^T, under the assumption
that all n-dependent work is performed offline against fixed dictionaries.

Supported online queries (constant-time, dependent only on fixed k and small ranks):
  - logdet(A + U V^T)
  - projected solves: P^T y where y = (A + U V^T)^{-1} b, for b from a fixed dictionary
  - selected traces: tr((A + U V^T)^{-1} B) for low-rank B = P Q^T
  - quadratic forms: c^T (A + U V^T)^{-1} c for c from a fixed set

Assumptions:
  - A is SPD (preferred) or at least invertible. SPD path uses Cholesky.
  - Rank k is fixed. Dictionaries (RHS b_j, test matrices B_l, projections P_ell, vectors c_i)
    are fixed offline.
  - All n-dependent products are precomputed offline; online work is k-by-k only.

Notation:
  T := A^{-1}, S := T U, K := I_k + V^T S.

References (standard identities):
  Woodbury: (A + U V^T)^{-1} = T - S K^{-1} V^T T
  det lemma: det(A + U V^T) = det(A) det(K)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np


def _cholesky_spd(A: np.ndarray) -> np.ndarray:
    """Return lower-triangular L with A = L L^T. Raises LinAlgError if not SPD."""
    return np.linalg.cholesky(A)


def _solve_spd_from_chol(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve A X = B for SPD A given its Cholesky factor L (lower), i.e., A = L L^T."""
    # Solve L Y = B
    Y = np.linalg.solve(L, B)
    # Solve L^T X = Y
    X = np.linalg.solve(L.T, Y)
    return X


def _is_symmetric(M: np.ndarray, tol: float = 1e-10) -> bool:
    return np.allclose(M, M.T, atol=tol, rtol=0.0)


@dataclass
class LowRankTraceSpec:
    """Specification for a low-rank test matrix B = P Q^T."""
    P: np.ndarray  # (n x r)
    Q: np.ndarray  # (n x r)

    def rank(self) -> int:
        return self.P.shape[1]


@dataclass
class ProjectionSpec:
    """Projection matrix P_ell in R^{n x p} for reporting P_ell^T y."""
    P: np.ndarray  # (n x p)

    @property
    def dim(self) -> int:
        return self.P.shape[1]


@dataclass
class OfflineData:
    # Scalars
    logdet_A: float

    # Core small matrix and factorization data
    K: np.ndarray                  # (k x k)
    chol_K: Optional[np.ndarray]   # lower-triangular if SPD; else None

    # Precomputed dictionary products
    PTY: List[np.ndarray]          # per projection ell: (p_ell x J)
    PT_S: List[np.ndarray]         # per projection ell: (p_ell x k)
    M_b: np.ndarray                # (k x J)   with columns m_j = V^T T b_j
    quad_baseline: np.ndarray      # (I,)      with entries c_i^T T c_i
    M_c: np.ndarray                # (k x I)   with columns m_i = V^T T c_i

    # Selected-trace components
    trace_baseline: List[float]    # tr(T B_l)
    C1_list: List[np.ndarray]      # V^T T P_l  shape (k x r_l)
    C2_list: List[np.ndarray]      # Q_l^T T U  shape (r_l x k), equivalently Q_l^T S

    # Metadata
    k: int
    J: int
    I: int
    proj_dims: List[int]           # [p_ell]


class LowRankO1Kernel:
    """
    O(1) low-rank update kernel.

    Build offline with:
        kernel = LowRankO1Kernel.build_spd(
            A=A, U=U, V=V, rhs_list=rhs_list, proj_list=proj_list,
            trace_list=trace_list, quad_list=quad_list, symmetric_update=True
        )

    Online queries:
        kernel.logdet()
        kernel.solve_projected(b_index, proj_index)  -> vector in R^{p_proj}
        kernel.selected_trace(trace_index)            -> scalar
        kernel.quadratic_form(c_index)               -> scalar

    All online methods run in time/memory independent of n.
    """

    def __init__(self, offline: OfflineData):
        self.off = offline

    @staticmethod
    def build_spd(
        A: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        rhs_list: List[np.ndarray],
        proj_list: List[ProjectionSpec],
        trace_list: List[LowRankTraceSpec],
        quad_list: List[np.ndarray],
        symmetric_update: bool = True,
    ) -> "LowRankO1Kernel":
        """
        Offline build for SPD A. Computes Cholesky of A and all n-dependent products.

        Args:
            A: SPD matrix (n x n).
            U: (n x k).
            V: (n x k). If symmetric_update, pass V = U for best stability.
            rhs_list: list of b_j (n,).
            proj_list: list of ProjectionSpec for P_ell (n x p_ell).
            trace_list: list of LowRankTraceSpec for B_l = P_l Q_l^T.
            quad_list: list of c_i (n,).
            symmetric_update: if True, treat K as SPD and use Cholesky on K.
        """
        A = np.asarray(A)
        U = np.asarray(U)
        V = np.asarray(V)
        n, k = U.shape
        assert A.shape == (n, n), "A shape mismatch"
        assert V.shape == (n, k), "V shape mismatch"

        # Cholesky of A (SPD)
        L = _cholesky_spd(A)
        logdet_A = 2.0 * np.sum(np.log(np.diag(L)))

        # Define A^{-1} application
        def T_apply(B: np.ndarray) -> np.ndarray:
            return _solve_spd_from_chol(L, B)

        # Precompute slices
        J = len(rhs_list)
        I = len(quad_list)

        # S = A^{-1} U
        S = T_apply(U)  # (n x k)

        # Dictionary solves
        Y = None
        if J > 0:
            Y = np.column_stack([T_apply(b.reshape(-1, 1)) for b in rhs_list])  # (n x J)

        Yc = None
        if I > 0:
            Yc = np.column_stack([T_apply(c.reshape(-1, 1)) for c in quad_list])  # (n x I)

        # Core K = I + V^T S
        K_core = np.eye(k, dtype=A.dtype) + V.T @ S  # (k x k)

        # Factorization of K if symmetric_update and K ~ SPD
        chol_K = None
        if symmetric_update:
            # Attempt Cholesky of symmetrized K
            K_sym = 0.5 * (K_core + K_core.T)
            try:
                chol_K = _cholesky_spd(K_sym)
                K_core = K_sym  # store the symmetrized core
            except np.linalg.LinAlgError:
                chol_K = None  # fall back to dense solves
        # Precompute M_b = V^T Y and M_c = V^T Yc
        M_b = np.zeros((k, J), dtype=A.dtype) if J > 0 else np.zeros((k, 0), dtype=A.dtype)
        if J > 0:
            M_b = V.T @ Y  # (k x J)

        M_c = np.zeros((k, I), dtype=A.dtype) if I > 0 else np.zeros((k, 0), dtype=A.dtype)
        quad_baseline = np.zeros((I,), dtype=A.dtype) if I > 0 else np.zeros((0,), dtype=A.dtype)
        if I > 0:
            M_c = V.T @ Yc  # (k x I)
            for i in range(I):
                quad_baseline[i] = float(quad_list[i].T @ Yc[:, i])

        # Projections
        PTY: List[np.ndarray] = []
        PT_S: List[np.ndarray] = []
        proj_dims: List[int] = []
        for spec in proj_list:
            P = np.asarray(spec.P)
            assert P.shape[0] == n, "Projection row mismatch"
            p = P.shape[1]
            proj_dims.append(p)
            # Store P^T Y and P^T S
            PTY.append(P.T @ Y if J > 0 else np.zeros((p, 0), dtype=A.dtype))  # (p x J)
            PT_S.append(P.T @ S)  # (p x k)

        # Selected traces
        trace_baseline: List[float] = []
        C1_list: List[np.ndarray] = []
        C2_list: List[np.ndarray] = []
        for spec in trace_list:
            P_l = np.asarray(spec.P)
            Q_l = np.asarray(spec.Q)
            assert P_l.shape[0] == n and Q_l.shape[0] == n, "Low-rank B dims mismatch"
            # AP_l = T P_l
            AP_l = T_apply(P_l)  # (n x r_l)
            # baseline tr(T B_l) = tr(T P_l Q_l^T) = tr(Q_l^T T P_l)
            baseline = float(np.trace(Q_l.T @ AP_l))
            trace_baseline.append(baseline)
            # C1 = V^T AP_l, C2 = Q_l^T S
            C1_list.append(V.T @ AP_l)          # (k x r_l)
            C2_list.append(Q_l.T @ S)           # (r_l x k)

        off = OfflineData(
            logdet_A=float(logdet_A),
            K=K_core,
            chol_K=chol_K,
            PTY=PTY,
            PT_S=PT_S,
            M_b=M_b,
            quad_baseline=quad_baseline,
            M_c=M_c,
            trace_baseline=trace_baseline,
            C1_list=C1_list,
            C2_list=C2_list,
            k=k,
            J=J,
            I=I,
            proj_dims=proj_dims,
        )
        return LowRankO1Kernel(off)

    def _solve_k(self, B: np.ndarray) -> np.ndarray:
        """
        Solve K X = B for small k x k K.
        Uses Cholesky if available; otherwise dense solve.
        """
        K = self.off.K
        if self.off.chol_K is not None:
            L = self.off.chol_K
            # Solve L Y = B
            Y = np.linalg.solve(L, B)
            # Solve L^T X = Y
            X = np.linalg.solve(L.T, Y)
            return X
        else:
            return np.linalg.solve(K, B)

    def _logdet_K(self) -> float:
        """Compute log(det K) using factorization if available."""
        if self.off.chol_K is not None:
            L = self.off.chol_K
            return 2.0 * float(np.sum(np.log(np.diag(L))))
        sign, ld = np.linalg.slogdet(self.off.K)
        if sign <= 0:
            raise np.linalg.LinAlgError("K is not positive det; cannot take logdet safely.")
        return float(ld)

    def logdet(self) -> float:
        """Return logdet(A + U V^T) = logdet(A) + logdet(K)."""
        return self.off.logdet_A + self._logdet_K()

    def solve_projected(self, b_index: int, proj_index: int) -> np.ndarray:
        """
        Return P^T y where y = (A + U V^T)^{-1} b_j, for a fixed projection P.
        Complexity: O(k^2 + p*k).
        """
        if not (0 <= b_index < self.off.J):
            raise IndexError("b_index out of range")
        if not (0 <= proj_index < len(self.off.PT_S)):
            raise IndexError("proj_index out of range")
        PTY = self.off.PTY[proj_index]      # (p x J)
        PT_S = self.off.PT_S[proj_index]    # (p x k)
        m = self.off.M_b[:, b_index]        # (k,)
        z = self._solve_k(m.reshape(-1, 1)) # (k x 1)
        out = PTY[:, b_index].reshape(-1, 1) - PT_S @ z
        return out.ravel()

    def selected_trace(self, trace_index: int) -> float:
        """
        Return tr((A + U V^T)^{-1} B_l) for low-rank B_l = P_l Q_l^T.
        Complexity: O(k^3 + k^2 r_l).
        """
        if not (0 <= trace_index < len(self.off.trace_baseline)):
            raise IndexError("trace_index out of range")
        baseline = self.off.trace_baseline[trace_index]
        C1 = self.off.C1_list[trace_index]  # (k x r)
        C2 = self.off.C2_list[trace_index]  # (r x k)
        H = C1 @ C2                         # (k x k)
        u = self._solve_k(H)                # (k x k) solves column-wise
        corr = float(np.trace(u))
        return float(baseline - corr)

    def quadratic_form(self, c_index: int) -> float:
        """
        Return q = c^T (A + U V^T)^{-1} c for a fixed c from dictionary.
        Complexity: O(k^2).
        """
        if not (0 <= c_index < self.off.I):
            raise IndexError("c_index out of range")
        baseline = float(self.off.quad_baseline[c_index])  # c^T T c
        m = self.off.M_c[:, c_index].reshape(-1, 1)        # V^T T c (k x 1)
        z = self._solve_k(m)                               # K^{-1} m (k x 1)
        return float(baseline - float(m.T @ z))

    # Utility accessors
    @property
    def k(self) -> int:
        return self.off.k

    @property
    def J(self) -> int:
        return self.off.J

    @property
    def I(self) -> int:
        return self.off.I

    @property
    def proj_dims(self) -> List[int]:
        return self.off.proj_dims


if __name__ == "__main__":
    # Minimal self-test (random SPD case, small sizes)
    rng = np.random.default_rng(0)
    n, k = 50, 2
    A0 = rng.standard_normal((n, n))
    A = A0 @ A0.T + 1.0 * np.eye(n)  # SPD
    U = rng.standard_normal((n, k))
    V = U.copy()  # symmetric update for stability

    # Dictionaries
    J = 3
    rhs_list = [rng.standard_normal(n) for _ in range(J)]

    I = 2
    quad_list = [rng.standard_normal(n) for _ in range(I)]

    # Projections
    P1 = rng.standard_normal((n, 4))
    P2 = rng.standard_normal((n, 3))
    proj_list = [ProjectionSpec(P1), ProjectionSpec(P2)]

    # Low-rank B = P Q^T specs
    r1, r2 = 3, 2
    B1 = LowRankTraceSpec(P=rng.standard_normal((n, r1)), Q=rng.standard_normal((n, r1)))
    B2 = LowRankTraceSpec(P=rng.standard_normal((n, r2)), Q=rng.standard_normal((n, r2)))
    trace_list = [B1, B2]

    kernel = LowRankO1Kernel.build_spd(
        A=A, U=U, V=V, rhs_list=rhs_list, proj_list=proj_list,
        trace_list=trace_list, quad_list=quad_list, symmetric_update=True
    )

    # Reference with dense inverses (for sanity, not part of O(1) online path)
    A_u = A + U @ V.T
    A_u_inv = np.linalg.inv(A_u)
    logdet_ref = np.linalg.slogdet(A_u)[1]

    # Check logdet
    assert np.allclose(kernel.logdet(), logdet_ref, rtol=1e-10, atol=1e-10)

    # Check projected solve
    for j in range(J):
        for pidx, Pspec in enumerate(proj_list):
            y_ref = A_u_inv @ rhs_list[j]
            proj_ref = Pspec.P.T @ y_ref
            proj_est = kernel.solve_projected(j, pidx)
            assert np.allclose(proj_est, proj_ref, rtol=1e-10, atol=1e-10)

    # Check selected traces
    for l, Bspec in enumerate(trace_list):
        B = Bspec.P @ Bspec.Q.T
        tr_ref = float(np.trace(A_u_inv @ B))
        tr_est = kernel.selected_trace(l)
        assert np.allclose(tr_est, tr_ref, rtol=1e-10, atol=1e-10)

    # Check quadratic forms
    for i in range(I):
        c = quad_list[i]
        q_ref = float(c.T @ (A_u_inv @ c))
        q_est = kernel.quadratic_form(i)
        assert np.allclose(q_est, q_ref, rtol=1e-10, atol=1e-10)

    print("Self-test passed.")
