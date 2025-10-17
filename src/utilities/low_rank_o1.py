# low_rank_o1.py
# ASCII, review-ready reference implementation
# O(1) Low-Rank Update Kernel for A_u = A + U V^T (rank k)
# - Offline: precompute A^{-1} times needed sticks/dictionaries and the kxk core
# - Online: constant-size solves on K = I + V^T A^{-1} U
#
# Dependencies: numpy only

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

Array = np.ndarray

def _tri_solve_lower(L: Array, B: Array) -> Array:
    """Forward solve L X = B for lower-triangular L. B can be (k,) or (k,m)."""
    L = np.asarray(L)
    B = np.asarray(B).copy()
    k = L.shape[0]
    if B.ndim == 1:
        X = np.empty_like(B, dtype=L.dtype)
        for i in range(k):
            s = B[i] - np.dot(L[i, :i], X[:i])
            X[i] = s / L[i, i]
        return X
    else:
        X = np.empty_like(B, dtype=L.dtype)
        for i in range(k):
            s = B[i] - L[i, :i] @ X[:i]
            X[i] = s / L[i, i]
        return X

def _tri_solve_upper(U: Array, B: Array) -> Array:
    """Backward solve U X = B for upper-triangular U."""
    U = np.asarray(U)
    B = np.asarray(B).copy()
    k = U.shape[0]
    if B.ndim == 1:
        X = np.empty_like(B, dtype=U.dtype)
        for i in range(k-1, -1, -1):
            s = B[i] - np.dot(U[i, i+1:], X[i+1:])
            X[i] = s / U[i, i]
        return X
    else:
        X = np.empty_like(B, dtype=U.dtype)
        for i in range(k-1, -1, -1):
            s = B[i] - U[i, i+1:] @ X[i+1:]
            X[i] = s / U[i, i]
        return X

@dataclass
class LowRankOffline:
    logdet_A: float
    S: Array         # S = A^{-1} U, shape (n,k)
    V: Array         # V, shape (n,k)
    K: Array         # K = I + V^T S, shape (k,k)
    cholK: Optional[Array]  # lower-triangular if SPD update; else None
    invK: Optional[Array]   # small-k inverse if not SPD; else None
    Y_b: Dict[int, Array]   # j -> y0 = A^{-1} b_j, shape (n,)
    Y_c: Dict[int, Array]   # i -> y0 = A^{-1} c_i, shape (n,)
    proj_P: Dict[int, Array]   # ell -> P_ell, shape (n,p)
    AP: Dict[int, Array]       # ell -> A^{-1} P_ell, shape (n,p)
    B_fact: Dict[int, Tuple[Array, Array]]  # ell -> (P_l, Q_l) with shapes (n,r), (n,r)
    AP_l: Dict[int, Array]     # ell -> A^{-1} P_l, shape (n,r_l)
    QS_l: Dict[int, Array]     # ell -> Q_l^T S, shape (r_l,k)
    baseline_tr_TB: Dict[int, float]  # ell -> tr(A^{-1} B_l)

class O1LowRankKernel:
    """
    Constant-time online kernel for rank-k updates A_u = A + U V^T.
    Offline: call build_offline() with dictionaries.
    Online: use methods logdet(), projected_solve(), selected_trace(), quadratic_form(), grad_proj().
    """
    def __init__(self, A: Array, U: Array, V: Optional[Array] = None, assume_spd: bool = True) -> None:
        self.A = np.asarray(A, dtype=np.float64)
        self.U = np.asarray(U, dtype=np.float64)
        self.V = np.asarray(U if V is None else V, dtype=np.float64)
        self.assume_spd = assume_spd

        n = self.A.shape[0]
        assert self.A.shape == (n, n)
        assert self.U.shape[0] == n and self.V.shape[0] == n
        self.k = self.U.shape[1]

        # Baseline factorization of A
        if assume_spd:
            try:
                self.L_A = np.linalg.cholesky(self.A)
            except np.linalg.LinAlgError:
                raise ValueError("A not SPD; set assume_spd=False for general invertible A.")
            self.logdet_A = 2.0 * np.log(np.diag(self.L_A)).sum()
            self.solveA = self._solve_spd
            self.solveAT = self._solve_spd_transpose
        else:
            sign, logdet = np.linalg.slogdet(self.A)
            if sign <= 0:
                raise ValueError("A must be invertible with positive det for logdet; got sign<=0.")
            self.logdet_A = logdet
            self.solveA = lambda rhs: np.linalg.solve(self.A, rhs)
            self.solveAT = lambda rhs: np.linalg.solve(self.A.T, rhs)

        # Sticks
        self.S = self.solveA(self.U)  # n x k
        self.R = self.solveA(self.V)  # n x k

        # Core
        self.K = np.eye(self.k) + self.V.T @ self.S  # k x k

        # Factorization on K
        self._try_factorize_K()

        # Empty dictionaries
        self.Y_b: Dict[int, Array] = {}
        self.Y_c: Dict[int, Array] = {}
        self.proj_P: Dict[int, Array] = {}
        self.AP: Dict[int, Array] = {}
        self.B_fact: Dict[int, Tuple[Array, Array]] = {}
        self.AP_l: Dict[int, Array] = {}
        self.QS_l: Dict[int, Array] = {}
        self.baseline_tr_TB: Dict[int, float] = {}

    # ---------- internal solves on K ----------

    def _try_factorize_K(self) -> None:
        Ksym = (self.K + self.K.T) * 0.5  # symmetrize for SPD test
        try:
            L = np.linalg.cholesky(Ksym)
            self.cholK = L
            self.invK = None
            self.spd_update = True
        except np.linalg.LinAlgError:
            self.cholK = None
            self.invK = np.linalg.inv(self.K)
            self.spd_update = False

        # condition guard (diagnostic)
        svals = np.linalg.svd(self.K, compute_uv=False)
        self.condK = float(svals.max() / svals.min())

    def _solve_K(self, B: Array) -> Array:
        """Solve K X = B using the stored factorization/inverse (constant-size)."""
        if self.cholK is not None:
            Y = _tri_solve_lower(self.cholK, B)
            X = _tri_solve_upper(self.cholK.T, Y)
            return X
        else:
            return self.invK @ B

    # ---------- offline dictionary builds ----------

    def add_b(self, j: int, b_j: Array) -> None:
        y0 = self.solveA(np.asarray(b_j, dtype=np.float64))
        self.Y_b[j] = y0

    def add_c(self, i: int, c_i: Array) -> None:
        y0 = self.solveA(np.asarray(c_i, dtype=np.float64))
        self.Y_c[i] = y0

    def add_projection(self, ell: int, P_ell: Array) -> None:
        P = np.asarray(P_ell, dtype=np.float64)
        self.proj_P[ell] = P
        self.AP[ell] = self.solveA(P)

    def add_selected_trace_factor(self, ell: int, P_l: Array, Q_l: Array) -> None:
        P_l = np.asarray(P_l, dtype=np.float64)
        Q_l = np.asarray(Q_l, dtype=np.float64)
        assert P_l.shape[0] == self.A.shape[0] and Q_l.shape[0] == self.A.shape[0]
        self.B_fact[ell] = (P_l, Q_l)
        self.AP_l[ell] = self.solveA(P_l)               # n x r
        self.QS_l[ell] = Q_l.T @ self.S                 # r x k
        self.baseline_tr_TB[ell] = float(np.sum(self.AP_l[ell] * Q_l))

    # ---------- online constant-size queries ----------

    def logdet(self) -> float:
        """logdet(A + U V^T) = logdet(A) + logdet(K)."""
        if self.cholK is not None:
            return self.logdet_A + 2.0 * np.log(np.diag(self.cholK)).sum()
        else:
            sign, logdetK = np.linalg.slogdet(self.K)
            if sign <= 0:
                raise RuntimeError("K not pos. for slogdet; numerical guard failed.")
            return self.logdet_A + logdetK

    def projected_solve(self, j: int, ell: int) -> Array:
        """Return P_ell^T y for y = (A + U V^T)^{-1} b_j, using stored projection/solve."""
        y0 = self.Y_b[j]                  # n
        P = self.proj_P[ell]              # n x p
        S = self.S                        # n x k
        m = self.V.T @ y0                 # k
        z = self._solve_K(m)              # k
        return P.T @ (y0 - S @ z)         # p

    def selected_trace(self, ell: int) -> float:
        """Compute tr( (A+UV^T)^{-1} B_l ) for B_l = P_l Q_l^T with pre-stored auxiliaries."""
        AP = self.AP_l[ell]               # n x r
        QS = self.QS_l[ell]               # r x k
        H = (self.V.T @ AP) @ QS          # k x k
        KinvH = self._solve_K(H)
        corr = float(np.trace(KinvH))
        return self.baseline_tr_TB[ell] - corr

    def quadratic_form(self, i: int) -> float:
        """Return q = c_i^T (A+UV^T)^{-1} c_i."""
        y0 = self.Y_c[i]
        m = self.V.T @ y0
        z = self._solve_K(m)
        return float(y0 @ y0 - m.T @ z)

    def grad_proj(self, ell: int) -> Tuple[Array, Array]:
        """
        Projected gradients of logdet w.r.t. U and V onto P_ell (columns).
        grad_U = A^{-T} V W^T,  grad_V = S W, with W = K^{-1}.
        Return ( P_ell^T grad_U , P_ell^T grad_V ) of shapes (p,k).
        """
        P = self.proj_P[ell]                  # n x p
        AP = self.AP[ell]                     # n x p  (A^{-1} P)
        W = self._solve_K(np.eye(self.k))     # K^{-1}
        GU_proj = (AP.T @ self.V) @ W.T       # p x k
        GV_proj = (P.T @ self.S) @ W          # p x k
        return GU_proj, GV_proj

    # ---------- helper ----------

    def _solve_spd(self, rhs: Array) -> Array:
        Y = _tri_solve_lower(self.L_A, rhs)
        X = _tri_solve_upper(self.L_A.T, Y)
        return X

    def _solve_spd_transpose(self, rhs: Array) -> Array:
        return self._solve_spd(rhs)
