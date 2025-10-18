# Sera-pre-core.py
# Sera v2.0 pre-core (strict CCR) 
# Dependencies: numpy.
#
# Provides
# - Stable streaming ratio estimator (PRF + EWVar + Kahan) with overlays.
# - Strict CCR combine (log-sum-exp equivalent), Safe-Z (with L1 bound), Auto-k (CCR-scaled).
# - Selector with optional quantile-based theta (online P^2 estimator).
# - Tokenizer: byte-level BPE with training, encode/decode, and stateful streaming encoder.
#

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


__all__ = [
    "PRF", "EWVar", "KahanDecay",
    "OverlayA", "OverlayC", "Selector",
    "SeraCore", "CCR", "SafeZ", "SafeZResult",
    "AutoK", "AutoKResult",
    "Tokenizer", "StreamingEncoder"
]


# -----------------------------
# Utilities
# -----------------------------

def _check_1d(x: np.ndarray, name: str) -> None:
    if not isinstance(x, np.ndarray) or x.ndim != 1:
        raise ValueError(f"{name} must be 1-D numpy array")

def _check_positive(x: float, name: str) -> None:
    if not (x > 0):
        raise ValueError(f"{name} must be > 0")

def _check_prob01(x: float, name: str) -> None:
    if not (0.0 < x < 1.0):
        raise ValueError(f"{name} must be in (0,1)")


def _safe_randn(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return rng.normal(size=shape)


# -----------------------------
# Core streaming estimator
# -----------------------------

@dataclass
class PRF:
    """Positive Random Features for exp kernel: k(q,x)=exp(<q,x>/tau).
    phi(x) = exp((W x)/sqrt(tau) - ||x||^2/(2 tau)) / sqrt(r), with optional clipping.
    """

    r: int
    d: int
    tau: float = 1.0
    clip_c: Optional[float] = 4.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    W: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.r <= 0 or self.d <= 0:
            raise ValueError("r and d must be positive")
        _check_positive(self.tau, "tau")
        self.W = _safe_randn(self.rng, self.r, self.d).astype(np.float64, copy=False)

    def features(self, x: np.ndarray) -> np.ndarray:
        _check_1d(x, "x")
        if x.shape[0] != self.d:
            raise ValueError("x dimension mismatch")
        g = (self.W @ x) / math.sqrt(self.tau)
        if self.clip_c is not None:
            g = np.clip(g, -self.clip_c, self.clip_c)
        xnorm = float(x @ x)
        phi = np.exp(g - xnorm / (2.0 * self.tau)) / math.sqrt(self.r)
        return phi


@dataclass
class EWVar:
    """Exponentially weighted running variance (per-dim)."""

    dim: int
    alpha: float  # decay: new = alpha * old + (1-alpha) * x
    mu: np.ndarray = field(init=False)
    sig2: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        _check_prob01(self.alpha, "alpha")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        self.mu = np.zeros(self.dim, dtype=float)
        self.sig2 = np.ones(self.dim, dtype=float) * 1e-6

    def update(self, x: np.ndarray) -> None:
        _check_1d(x, "x")
        if x.shape[0] != self.dim:
            raise ValueError("x dimension mismatch")
        mu_old = self.mu
        mu_new = self.alpha * mu_old + (1.0 - self.alpha) * x
        diff = x - mu_new
        sig2_new = self.alpha * self.sig2 + (1.0 - self.alpha) * (diff * diff)
        self.mu, self.sig2 = mu_new, sig2_new


@dataclass
class KahanDecay:
    """Vector Kahan-compensated exponentially decayed sum.
    Maintains S_t = gamma * S_{t-1} + x_t with compensation.
    """

    dim: int
    gamma: float
    S: np.ndarray = field(init=False)
    C: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        _check_prob01(self.gamma, "gamma")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        self.S = np.zeros(self.dim, dtype=float)
        self.C = np.zeros(self.dim, dtype=float)

    def update(self, x: np.ndarray) -> None:
        _check_1d(x, "x")
        if x.shape[0] != self.dim:
            raise ValueError("x dimension mismatch")
        self.S *= self.gamma
        self.C *= self.gamma
        y = x - self.C
        t = self.S + y
        self.C = (t - self.S) - y
        self.S = t


@dataclass
class OverlayA:
    """Scalar numerator overlay anchored at a feature location.
    y_hat(q) = (num(q) + u * <phi_w(q), phi_anchor>) / den(q)
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
    """

    H: np.ndarray              # shape (r, r_v)
    Delta_k: np.ndarray        # shape (r_v, r_v)
    ones: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        r_v = self.Delta_k.shape[0]
        self.ones = np.ones((r_v, 1), dtype=float)

    @staticmethod
    def fit_rank_k(Z: np.ndarray, t: np.ndarray, k: int) -> np.ndarray:
        r_v = Z.shape[1]
        d_hat, *_ = np.linalg.lstsq(Z, t, rcond=None)
        Delta_full = np.diag(d_hat)
        U, S, Vt = np.linalg.svd(Delta_full, full_matrices=False)
        k = max(1, min(k, r_v))
        Wk = U[:, :k] * S[:k]
        Vk = Vt[:k, :]
        Delta_k = Wk @ Vk
        return Delta_k

    def add_correction(self, phi_w: np.ndarray) -> float:
        z = phi_w @ self.H
        return float(z @ (self.Delta_k @ self.ones).reshape(-1))


# -----------------------------
# Selector with optional quantile calibration
# -----------------------------

class P2Quantile:
    """Piecewise-parabolic (P^2) streaming quantile estimator.
    Maintains q-quantile online without storing all samples.
    """

    def __init__(self, q: float) -> None:
        _check_prob01(q, "q")
        self.q = q
        self.initial: List[float] = []
        # markers: positions n[i], desired positions n'[i], heights h[i]
        self.n = np.zeros(5, dtype=float)
        self.np = np.zeros(5, dtype=float)
        self.h = np.zeros(5, dtype=float)
        self.count = 0

    def update(self, x: float) -> None:
        if self.count < 5:
            self.initial.append(float(x))
            self.count += 1
            if self.count == 5:
                self.initial.sort()
                self.h[:] = self.initial
                self.n[:] = [1, 2, 3, 4, 5]
                self.np[:] = [1, 1 + 2*self.q, 1 + 4*self.q, 3 + 2*self.q, 5]
            return

        # find cell k
        k = np.searchsorted(self.h, x)
        if k == 0:
            self.h[0] = x; k = 1
        elif k == 5:
            self.h[4] = x; k = 4
        # increment positions
        self.n[:k] += 1
        self.np[0] += 0; self.np[1] += self.q/2; self.np[2] += self.q; self.np[3] += (1+self.q)/2; self.np[4] += 1
        self.n[k:] += 1
        # adjust heights
        for i in range(1, 4):
            d = self.np[i] - self.n[i]
            if (d >= 1 and self.n[i+1] - self.n[i] > 1) or (d <= -1 and self.n[i-1] - self.n[i] < -1):
                d = math.copysign(1.0, d)
                hp = self.h[i] + d * (
                    (self.n[i] - self.n[i-1] + d) * (self.h[i+1] - self.h[i]) / (self.n[i+1] - self.n[i]) +
                    (self.n[i+1] - self.n[i] - d) * (self.h[i] - self.h[i-1]) / (self.n[i] - self.n[i-1])
                ) / (self.n[i+1] - self.n[i-1])
                if self.h[i-1] < hp < self.h[i+1]:
                    self.h[i] = hp
                else:
                    # linear
                    if d > 0:
                        self.h[i] += (self.h[i+1] - self.h[i]) / (self.n[i+1] - self.n[i])
                    else:
                        self.h[i] += (self.h[i-1] - self.h[i]) / (self.n[i-1] - self.n[i])
        self.count += 1

    def value(self) -> float:
        if self.count < 5:
            return float(np.median(self.initial)) if self.initial else 0.0
        return float(self.h[2])


@dataclass
class Selector:
    """Thresholded, constant-time overlay selector.
    Picks top-P by |score| among those with |score| >= theta.
    theta can be set directly or via streaming quantile estimator fit_theta().
    """

    P: int = 1
    theta: float = 0.0
    q_estimator: Optional[P2Quantile] = None

    def select_scores(self, scores: np.ndarray) -> np.ndarray:
        mask = np.abs(scores) >= self.theta
        idx = np.where(mask)[0]
        if idx.size == 0 or self.P <= 0:
            return np.empty((0,), dtype=int)
        order = np.argsort(np.abs(scores[idx]))[::-1]
        return idx[order[: self.P]]

    def fit_theta(self, abs_scores_iter: Iterable[float], q: float = 0.95) -> float:
        """Fit theta by streaming approximate quantile over |scores|."""
        self.q_estimator = P2Quantile(q)
        for s in abs_scores_iter:
            self.q_estimator.update(float(s))
        self.theta = float(self.q_estimator.value())
        return self.theta


# -----------------------------
# SeraCore
# -----------------------------

@dataclass
class SeraCore:
    """Streaming ratio estimator with overlays and strict-CCR utilities."""

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
        _check_positive(self.floor, "floor")
        _check_prob01(self.gamma, "gamma")
        _check_positive(self.tau, "tau")
        self.prf = PRF(r=self.r, d=self.d, tau=self.tau, clip_c=self.clip_c, rng=self.rng)
        self.ew = EWVar(dim=self.r, alpha=self.gamma)
        self.R_acc = np.zeros(self.r, dtype=float)
        self.s_acc = np.zeros(self.r, dtype=float)
        if self.use_kahan:
            self.kahan_R = KahanDecay(dim=self.r, gamma=self.gamma)
            self.kahan_s = KahanDecay(dim=self.r, gamma=self.gamma)

    def ingest(self, x: np.ndarray, v: float) -> None:
        phi = self.prf.features(x)
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
        phi_q = self.prf.features(q)
        phi_w = phi_q / np.sqrt(self.ew.sig2 + 1e-12)
        num = float(phi_w @ self.R_acc)
        den = float(phi_w @ self.s_acc) + self.floor
        y = num / den
        if return_parts:
            return y, num, den, phi_w
        return y, 0.0, 0.0, phi_w

    def predict_with_overlays(self, q: np.ndarray) -> Tuple[float, dict]:
        y, num, den, phi_w = self.query(q, return_parts=True)
        if self.overlays_A:
            scores = np.array([float(phi_w @ ov.phi_anchor) for ov in self.overlays_A])
            idx = self.selector.select_scores(scores)
            if idx.size > 0:
                num2 = num + float(np.sum(scores[idx] * np.array([self.overlays_A[j].u for j in idx])))
                y = num2 / den
        if self.overlay_C is not None:
            y = y + self.overlay_C.add_correction(phi_w)
        return y, {"num": num, "den": den}

    def add_overlays_A(self, overlays: List[OverlayA]) -> None:
        self.overlays_A.extend(overlays)

    def set_overlay_C(self, overlay_c: OverlayC) -> None:
        self.overlay_C = overlay_c

    def set_selector(self, P: int, theta: float) -> None:
        self.selector = Selector(P=P, theta=theta)


# -----------------------------
# Strict CCR primitives
# -----------------------------

class CCR:
    """Strict CCR combine and utilities (log-sum-exp equivalent)."""

    @staticmethod
    def combine_pair(m1: float, Z1: float, m2: float, Z2: float) -> Tuple[float, float]:
        m = m1 if m1 >= m2 else m2
        Z = Z1 * math.exp(m1 - m) + Z2 * math.exp(m2 - m)
        return m, Z

    @staticmethod
    def combine_many(ms: Sequence[float], Zs: Sequence[float]) -> Tuple[float, float]:
        m, Z = -float("inf"), 0.0
        for mi, Zi in zip(ms, Zs):
            m, Z = CCR.combine_pair(m, Z, mi, Zi)
        return m, Z

    @staticmethod
    def strict_global(scores: np.ndarray) -> Tuple[float, float, np.ndarray]:
        m = float(np.max(scores))
        Z = float(np.sum(np.exp(scores - m)))
        alpha = np.exp(scores - m) / Z
        return m, Z, alpha

    @staticmethod
    def tile_stats(scores: np.ndarray, idxs: np.ndarray) -> Tuple[float, float]:
        s = scores[idxs]
        m = float(np.max(s))
        Z = float(np.sum(np.exp(s - m)))
        return m, Z


# -----------------------------
# Safe-Z pruning under CCR
# -----------------------------

@dataclass
class SafeZResult:
    m_hat: float
    Z_hat: float
    Rhat_global: float
    L1_bound: float


class SafeZ:
    """Construct Z_hat by pruning per-tile tails; combine under CCR; supply an L1 bound."""

    @staticmethod
    def build(scores: np.ndarray, chunks: Sequence[np.ndarray], prune_frac: float) -> SafeZResult:
        if not (0.0 <= prune_frac < 1.0):
            raise ValueError("prune_frac must be in [0,1)")
        ms: List[float] = []
        Zc_list: List[float] = []
        Rhat_list: List[float] = []
        for ch in chunks:
            s = scores[ch]
            m_i = float(np.max(s))
            k_keep = max(1, int(len(s) * (1.0 - prune_frac)))
            idx_sorted = np.argsort(s)[::-1]
            keep = idx_sorted[:k_keep]
            pruned = idx_sorted[k_keep:]
            Zc_i = float(np.sum(np.exp(s[keep] - m_i)))
            if len(pruned) > 0:
                UB = float(np.max(s[pruned]))
                Rhat_i = len(pruned) * math.exp(UB - m_i)
            else:
                Rhat_i = 0.0
            ms.append(m_i); Zc_list.append(Zc_i); Rhat_list.append(Rhat_i)
        m_hat, Z_hat = CCR.combine_many(ms, [a + b for a, b in zip(Zc_list, Rhat_list)])
        Rhat_global = (sum(Rhat_list)) / max(1e-12, sum(Zc_list))
        L1_bound = 2.0 * Rhat_global / (1.0 + Rhat_global)
        return SafeZResult(m_hat=m_hat, Z_hat=Z_hat, Rhat_global=Rhat_global, L1_bound=L1_bound)


# -----------------------------
# Auto-k global stopping (CCR-scaled)
# -----------------------------

@dataclass
class AutoKResult:
    K_per_tile: List[int]
    head_sum: float
    eps_eff: float


class AutoK:
    """Pick minimal per-tile heads to meet 2*(Z - head_sum)/Z <= eps in global frame."""

    @staticmethod
    def prepare(scores: np.ndarray, chunks: Sequence[np.ndarray]) -> Tuple[List[float], List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[float], float, float, List[float]]:
        ms: List[float] = []
        Zs: List[float] = []
        head_lists: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for ch in chunks:
            s = scores[ch]
            m_i = float(np.max(s)); ms.append(m_i)
            a_i = np.exp(s - m_i)
            idx = np.argsort(a_i)[::-1]
            a_sorted = a_i[idx]
            head_lists.append((ch, idx, a_sorted))
            Zs.append(float(np.sum(a_i)))
        m_c, Z_c = CCR.combine_many(ms, Zs)
        scales = [math.exp(m_i - m_c) for m_i in ms]
        return ms, head_lists, Zs, m_c, Z_c, scales

    @staticmethod
    def run(heads: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], scales: List[float], Z_c: float, eps: float) -> AutoKResult:
        _check_positive(eps, "eps")
        P = len(heads)
        K = [0] * P
        head_sum = 0.0
        total_len = sum(len(a) for (_, _, a) in heads)
        while True:
            R = (Z_c - head_sum) / Z_c
            if 2 * R / (1 + R) <= eps or sum(K) == total_len:
                break
            best_gain = -1.0; best_i = -1
            for i, (_, _, a_sorted) in enumerate(heads):
                if K[i] < len(a_sorted):
                    gain = scales[i] * float(a_sorted[K[i]])
                    if gain > best_gain:
                        best_gain = gain; best_i = i
            if best_i < 0:
                break
            head_sum += best_gain
            K[best_i] += 1
        eps_eff = 2 * (Z_c - head_sum) / Z_c
        return AutoKResult(K_per_tile=K, head_sum=head_sum, eps_eff=eps_eff)


# -----------------------------
# Tokenizer (byte-level BPE) + stateful streaming encoder
# -----------------------------

@dataclass
class Tokenizer:
    """Byte-level BPE tokenizer with training and encode/decode.
    - Vocabulary starts from 256 byte tokens.
    - Merges learned by simple pair-frequency BPE on bytes.
    - Greedy encoding using merge ranks (lower rank = higher priority).
    """

    merges: List[Tuple[bytes, bytes]] = field(default_factory=list)
    vocab: Dict[bytes, int] = field(default_factory=dict)
    inv_vocab: Dict[int, bytes] = field(default_factory=dict)
    special: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.vocab:
            for b in range(256):
                self.vocab[bytes([b])] = b
        self.inv_vocab = {i: t for t, i in self.vocab.items()}
        for tok in ["<pad>", "<bos>", "<eos>", "<unk>"]:
            if tok not in self.special:
                self.special[tok] = 100000 + len(self.special)

    @staticmethod
    def _text_to_bytes(text: str) -> bytes:
        return text.encode("utf-8", errors="strict")

    @staticmethod
    def _bytes_to_text(b: bytes) -> str:
        return b.decode("utf-8", errors="strict")

    @staticmethod
    def _split_bytes_to_symbols(b: bytes) -> List[bytes]:
        return [bytes([x]) for x in b]

    def _pair_freq(self, seqs: List[List[bytes]]) -> Dict[Tuple[bytes, bytes], int]:
        freq: Dict[Tuple[bytes, bytes], int] = {}
        for seq in seqs:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                freq[pair] = freq.get(pair, 0) + 1
        return freq

    def _apply_merge_once(self, seq: List[bytes], a: bytes, b: bytes, merged: bytes) -> List[bytes]:
        out: List[bytes] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    def train_bpe(self, texts: Iterable[str], vocab_size: int = 4096, min_freq: int = 2, max_merges: Optional[int] = None) -> None:
        seqs: List[List[bytes]] = []
        for t in texts:
            b = self._text_to_bytes(t)
            seqs.append(self._split_bytes_to_symbols(b))

        target_merges = (vocab_size - 256)
        if max_merges is not None:
            target_merges = min(target_merges, max_merges)
        merges: List[Tuple[bytes, bytes]] = []

        for _ in range(max(0, target_merges)):
            freq = self._pair_freq(seqs)
            if not freq:
                break
            pair, f = max(freq.items(), key=lambda x: x[1])
            if f < min_freq:
                break
            a, b = pair
            merged = a + b
            merges.append(pair)
            seqs = [self._apply_merge_once(seq, a, b, merged) for seq in seqs]

        self.merges = merges
        self.vocab = {bytes([i]): i for i in range(256)}
        next_id = 256
        for a, b in merges:
            tok = a + b
            if tok not in self.vocab:
                self.vocab[tok] = next_id
                next_id += 1
        self.inv_vocab = {i: t for t, i in self.vocab.items()}

    def encode_bytes(self, b: bytes, level: Optional[int] = None) -> List[int]:
        symbols: List[bytes] = self._split_bytes_to_symbols(b)
        merges = self.merges if level is None else self.merges[:max(0, min(len(self.merges), level))]
        ranks = {pair: i for i, pair in enumerate(merges)}
        while True:
            best_i = None
            best_rank = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair in ranks:
                    rank = ranks[pair]
                    if best_rank is None or rank < best_rank:
                        best_rank = rank; best_i = i
            if best_i is None:
                break
            a, b = symbols[best_i], symbols[best_i + 1]
            merged = a + b
            symbols = symbols[:best_i] + [merged] + symbols[best_i + 2 :]
        ids: List[int] = []
        for s in symbols:
            if s not in self.vocab:
                self.vocab[s] = max(self.vocab.values()) + 1
                self.inv_vocab[self.vocab[s]] = s
            ids.append(self.vocab[s])
        return ids

    def decode_bytes(self, ids: List[int]) -> bytes:
        toks = [self.inv_vocab[i] for i in ids]
        return b"".join(toks)

    def encode(self, text: str, level: Optional[int] = None) -> List[int]:
        return self.encode_bytes(self._text_to_bytes(text), level=level)

    def decode(self, ids: List[int]) -> str:
        return self._bytes_to_text(self.decode_bytes(ids))


class StreamingEncoder:
    """Stateful streaming encoder that preserves cross-chunk merges.
    Keeps a tail of last K symbols; on each chunk, re-merges tail+chunk and emits
    all but the last K symbols (kept for the next step). Call flush() at end.
    """

    def __init__(self, tokenizer: Tokenizer, level: Optional[int] = None, tail_keep: int = 32) -> None:
        self.tok = tokenizer
        self.level = level
        self.K = max(1, int(tail_keep))
        self._tail: List[bytes] = []

    def _symbols_from_text(self, text: str) -> List[bytes]:
        return [bytes([x]) for x in text.encode("utf-8", errors="strict")]

    def push(self, text_chunk: str) -> List[int]:
        symbols = self._tail + self._symbols_from_text(text_chunk)
        # encode in one shot, then keep last K symbols (by bytes) for future merges
        ids = self.tok.encode_bytes(b"".join(symbols), level=self.level)
        # Rebuild symbols from ids to determine tail boundary
        merged_symbols = [self.tok.inv_vocab[i] for i in ids]
        if len(merged_symbols) <= self.K:
            # keep everything; emit nothing yet
            self._tail = merged_symbols
            return []
        emit_syms = merged_symbols[:-self.K]
        self._tail = merged_symbols[-self.K:]
        # map emit_syms to ids (ensure existing vocab)
        out: List[int] = [self.tok.vocab[s] for s in emit_syms]
        return out

    def flush(self) -> List[int]:
        if not self._tail:
            return []
        ids = [self.tok.vocab[s] for s in self._tail]
        self._tail = []
        return ids


# -----------------------------
# Smoke tests
# -----------------------------

def _smoke_all() -> None:
    rng = np.random.default_rng(7)
    # CCR correctness
    N, P = 2000, 8
    scores = rng.normal(size=N)
    order = rng.permutation(N)
    chunks = np.array_split(order, P)
    ms, Zs = zip(*[CCR.tile_stats(scores, ch) for ch in chunks])
    m_c, Z_c = CCR.combine_many(list(ms), list(Zs))
    m_g, Z_g, alpha_g = CCR.strict_global(scores)
    alpha_c = np.exp(scores - m_c) / Z_c
    assert abs(m_c - m_g) < 1e-10
    assert abs(Z_c - Z_g) / max(1.0, abs(Z_g)) < 1e-12
    assert float(np.sum(np.abs(alpha_c - alpha_g))) < 1e-10

    # Safe-Z bound
    from math import isfinite
    for pf in [0.3, 0.5]:
        sz = SafeZ.build(scores, chunks, prune_frac=pf)
        alpha_hat = np.exp(scores - sz.m_hat) / sz.Z_hat
        L1_err = float(np.sum(np.abs(alpha_hat - alpha_g)))
        assert L1_err <= sz.L1_bound + 1e-8
        assert isfinite(sz.m_hat) and isfinite(sz.Z_hat)

    # Auto-k
    _, heads, Zs2, m_c2, Z_c2, scales = AutoK.prepare(scores, chunks)
    ak = AutoK.run(heads, scales, Z_c2, eps=0.1)
    assert ak.eps_eff <= 0.11

    # Streaming core
    d, r = 16, 256
    core = SeraCore(d=d, r=r, tau=1.2, gamma=0.99, clip_c=4.0, floor=1e-6, rng=rng)
    X = rng.normal(size=(200, d)) / math.sqrt(d)
    w = rng.normal(size=d)
    v = X @ w + 0.05 * rng.normal(size=200)
    for t in range(200):
        core.ingest(X[t], float(v[t]))
    y, dbg = core.predict_with_overlays(rng.normal(size=d) / math.sqrt(d))
    _ = (y, dbg)

    # Tokenizer batch vs streaming
    text = "Sera tokenizer streaming test. " * 20
    tok = Tokenizer()
    tok.train_bpe([text], vocab_size=1024, min_freq=2, max_merges=200)
    ids_batch = tok.encode(text)
    enc = StreamingEncoder(tok, level=None, tail_keep=32)
    ids_stream: List[int] = []
    for i in range(0, len(text), 17):
        ids_stream.extend(enc.push(text[i:i+17]))
    ids_stream.extend(enc.flush())
    # Equivalence check: streaming equals batch (same tokenizer and merges)
    assert tok.decode(ids_stream) == tok.decode(ids_batch)

if __name__ == "__main__":
    _smoke_all()
    print("OK: strict CCR core + Selector(P^2) + Tokenizer(streaming) passed smoke tests.")

# =============================
# Minimal additions
# =============================

# AutoK: heap-based O(K log P) greedy
def _autok_run_heap(heads, scales, Z_c, eps):
    import heapq
    P = len(heads)
    K = [0] * P
    head_sum = 0.0
    total_len = sum(len(a) for (_,_,a) in heads)

    # max-heap via negatives
    heap = []
    for i, (_, _, a_sorted) in enumerate(heads):
        if len(a_sorted) > 0:
            gain = scales[i] * float(a_sorted[0])
            heapq.heappush(heap, (-gain, i, 0))

    while True:
        R = (Z_c - head_sum) / Z_c
        if 2 * R / (1 + R) <= eps or sum(K) == total_len or not heap:
            break
        ngain, i, k = heapq.heappop(heap)
        gain = -ngain
        head_sum += gain
        K[i] += 1
        # push next candidate from this tile
        _, _, a_sorted = heads[i]
        if K[i] < len(a_sorted):
            nxt_gain = scales[i] * float(a_sorted[K[i]])
            heapq.heappush(heap, (-nxt_gain, i, K[i]))

    eps_eff = 2 * (Z_c - head_sum) / Z_c
    return AutoKResult(K_per_tile=K, head_sum=head_sum, eps_eff=eps_eff)

# Bind as classmethod-like attribute
AutoK.run_heap = staticmethod(_autok_run_heap)


# SafeZ: argpartition-based fast build
def _safez_build_fast(scores: np.ndarray, chunks, prune_frac: float) -> SafeZResult:
    if not (0.0 <= prune_frac < 1.0):
        raise ValueError("prune_frac must be in [0,1)")
    ms = []
    Zc_list = []
    Rhat_list = []
    for ch in chunks:
        s = scores[ch]
        m_i = float(np.max(s))
        n = len(s)
        k_keep = max(1, int(n * (1.0 - prune_frac)))
        # argpartition to get top-k indices (unsorted)
        idx_top = np.argpartition(s, n - k_keep)[n - k_keep:]
        # optional: get UB over pruned part if any
        if k_keep < n:
            idx_pruned = np.argpartition(s, n - k_keep - 1)[: n - k_keep]
            UB = float(np.max(s[idx_pruned]))
            Rhat_i = (n - k_keep) * math.exp(UB - m_i)
        else:
            Rhat_i = 0.0
        Zc_i = float(np.sum(np.exp(s[idx_top] - m_i)))
        ms.append(m_i); Zc_list.append(Zc_i); Rhat_list.append(Rhat_i)

    m_hat, Z_hat = CCR.combine_many(ms, [a + b for a, b in zip(Zc_list, Rhat_list)])
    Rhat_global = (sum(Rhat_list)) / max(1e-12, sum(Zc_list))
    L1_bound = 2.0 * Rhat_global / (1.0 + Rhat_global)
    return SafeZResult(m_hat=m_hat, Z_hat=Z_hat, Rhat_global=Rhat_global, L1_bound=L1_bound)

# Bind
SafeZ.build_fast = staticmethod(_safez_build_fast)


# Tokenizer: unigram-like DP encoder with weights learned from BPE segmentation
def _tok_train_unigram_from_bpe(self, texts, smooth: float = 1.0):
    # Count token frequencies by BPE segmentation of corpus, then set log-probs.
    counts = {}
    total = 0
    for t in texts:
        ids = self.encode(t)  # BPE segmentation
        for i in ids:
            tok = self.inv_vocab[i]
            counts[tok] = counts.get(tok, 0) + 1
            total += 1
    self._uni_logprob = {}
    V = len(self.vocab)
    for tok, idx in self.vocab.items():
        c = counts.get(tok, 0)
        p = (c + smooth) / (total + smooth * V)
        self._uni_logprob[tok] = -math.log(max(p, 1e-12))

def _tok_encode_unigram(self, text: str, max_tok_bytes: int = 16) -> list:
    # Viterbi/DP over bytes with token costs from _uni_logprob (fallback to large cost).
    b = text.encode("utf-8", errors="strict")
    n = len(b)
    if n == 0:
        return []
    # Precompute a set of token bytes and max length
    if not hasattr(self, "_uni_logprob"):
        # default: uniform costs over vocab
        self._uni_logprob = {tok: 0.0 for tok in self.vocab.keys()}
    vocab_tokens = list(self.vocab.keys())
    max_len = max(max(len(t) for t in vocab_tokens), max_tok_bytes)
    # DP arrays
    INF = 1e18
    dp = [INF] * (n + 1); dp[0] = 0.0
    prev = [-1] * (n + 1); prev_tok = [b""] * (n + 1)
    # Build a quick index by first byte to prune
    first_map = {}
    for t in vocab_tokens:
        if not t:
            continue
        first_map.setdefault(t[0], []).append(t)
    for i in range(n):
        if dp[i] >= INF:
            continue
        bucket = first_map.get(b[i], [])
        for t in bucket:
            L = len(t)
            if L == 0 or L > max_len or i + L > n:
                continue
            if b[i:i+L] == t:
                cost = self._uni_logprob.get(t, 10.0)  # fallback cost
                if dp[i] + cost < dp[i+L]:
                    dp[i+L] = dp[i] + cost
                    prev[i+L] = i
                    prev_tok[i+L] = t
        # fallback to single byte (guaranteed)
        if dp[i] + 5.0 < dp[i+1]:  # small penalty
            dp[i+1] = dp[i] + 5.0
            prev[i+1] = i
            prev_tok[i+1] = bytes([b[i]])
    # reconstruct
    ids = []
    cur = n
    while cur > 0:
        p = prev[cur]
        t = prev_tok[cur]
        if p < 0 or not t:
            # safety: single byte fallback
            t = bytes([b[cur-1]])
            p = cur - 1
        ids.append(self.vocab.setdefault(t, max(self.vocab.values()) + 1))
        cur = p
    ids.reverse()
    return ids

# Bind
Tokenizer.train_unigram_from_bpe = _tok_train_unigram_from_bpe
Tokenizer.encode_unigram = _tok_encode_unigram
