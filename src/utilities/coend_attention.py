# coend_attention.py
# ASCII reference implementation of Coend-Attention (softmax variant), with:
# - two-pass streaming log-sum-exp (no score matrix kept)
# - Auto-k from an L1 budget epsilon (head/tail bound)
# - Safe-Z using an upper bound UB for non-candidates
#
# Dependencies: numpy only

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence

Array = np.ndarray

def streaming_lse(scores: Array) -> Tuple[float, float]:
    """
    Two-pass stable log-sum-exp in streaming form on a 1D array of scores s_i.
    Returns (m, Z) where m = max_i s_i and Z = sum_i exp(s_i - m).
    """
    m = -np.inf
    Z = 0.0
    for s in scores:
        s = float(s)
        if s > m:
            Z = (Z * np.exp(m - s) + 1.0) if np.isfinite(m) else 1.0
            m = s
        else:
            Z += np.exp(s - m)
    return m, Z

def auto_k_from_epsilon(sorted_scores_desc: Array, epsilon: float) -> int:
    """
    Given scores sorted descending (s_1 >= s_2 >= ...), compute minimal k s.t.
    L1(p, q) = 2T/(H+T) <= epsilon, with H = sum_{i<=k} exp(s_i - s_1), T = sum_{i>k} exp(s_i - s_1).
    """
    assert epsilon > 0 and epsilon < 1.0
    if len(sorted_scores_desc) == 0:
        return 0
    thr = epsilon / (2.0 - epsilon)            # R = T/H <= thr
    m = float(sorted_scores_desc[0])
    a = np.exp(sorted_scores_desc - m)         # a_i in (0,1]
    H = 0.0
    Z_hat = float(np.sum(a))
    for i, ai in enumerate(a, start=1):
        H += ai
        Rhat = (Z_hat - H) / H
        if Rhat <= thr:
            return i
    return len(sorted_scores_desc)

def safe_Z_hat(m: float, Z_candidates: float, UB: float, noncandidate_count: int) -> float:
    """
    Safe upper bound on the partition function using an upper bound UB >= max_{x in P} s(x)
    for the pruned set P (non-candidates).
    """
    Rhat = noncandidate_count * np.exp(UB - m) if noncandidate_count > 0 else 0.0
    return Z_candidates + Rhat

@dataclass
class AttentionOutput:
    out: Array
    alpha: Array
    m: float
    Z: float

def dot_scores(q: Array, K: Array) -> Array:
    """Compute raw scores s_i = q^T k_i (K rows are keys)."""
    return K @ q

def attention_softmax_streaming(q: Array, K: Array, V: Array) -> AttentionOutput:
    """
    Dense path: compute softmax(q^T K^T) @ V using two-pass streaming.
    Shapes: q(d,), K(n,d), V(n,dv). Returns out(dv,), alpha(n,).
    """
    s = dot_scores(q, K)
    m, Z = streaming_lse(s)
    a = np.exp(s - m) / Z
    out = a @ V
    return AttentionOutput(out=out, alpha=a, m=m, Z=Z)

def attention_sublinear(q: Array, K: Array, V: Array,
                        cand_idx: Sequence[int],
                        UB_non_cand: Optional[float],
                        epsilon: float) -> AttentionOutput:
    """
    Sublinear path with candidates and Safe-Z + Auto-k.
    - cand_idx: indices of candidate rows in K,V (already filtered/refined/sorted by score desc).
    - UB_non_cand: upper bound on score for non-candidates (None -> dense fallback).
    - epsilon: L1 budget for Auto-k.
    """
    if UB_non_cand is None or len(cand_idx) == 0:
        return attention_softmax_streaming(q, K, V)

    # compute candidate scores
    s_all = dot_scores(q, K[cand_idx])
    order = np.argsort(-s_all)
    s_sorted = s_all[order]
    idx_sorted = np.array(cand_idx, dtype=int)[order]

    # first pass on candidates
    m, Zc = streaming_lse(s_sorted)
    # Safe-Z
    Zhat = safe_Z_hat(m=m, Z_candidates=Zc, UB=float(UB_non_cand), noncandidate_count=K.shape[0] - len(cand_idx))
    # Auto-k
    k = auto_k_from_epsilon(s_sorted, epsilon=epsilon)

    # second pass on top-k to produce output
    s_top = s_sorted[:k]
    a_top = np.exp(s_top - m) / Zhat
    V_top = V[idx_sorted[:k]]
    out = a_top @ V_top

    alpha = np.zeros(K.shape[0], dtype=float)
    alpha[idx_sorted[:k]] = a_top
    return AttentionOutput(out=out, alpha=alpha, m=m, Z=Zhat)
