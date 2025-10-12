"""Self-contained O(1) encoder/generator demo script.

This module coalesces the original multi-file implementation that lived under
the ``o1stack`` package into a single importable / executable script.  The
contents here are effectively the union of ``xp.py``, ``backends.py``,
``numerics.py``, ``model.py``, ``bench.py``, ``selftest.py`` and ``cli.py``.

The goal of providing the functionality in one file is to make distribution or
experimentation easier in environments where packaging a full module tree is
undesirable.  All public behaviours of the original modules are preserved:

* ``xp`` helpers that pick between NumPy and CuPy.
* Backend abstractions (with a TensorRT probe shim).
* Numerical helpers (Kahan summation, SERAS stabiliser).
* Encoder / generator toy models.
* Benchmark and self-test entry points.
* CLI with ``bench`` and ``selftest`` sub-commands.

The script can be executed directly::

    python -m src.machine_learning.o1stack_single bench --out-dir results

or imported as a module to access the helpers programmatically.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import importlib.util
import pandas as pd

try:  # pragma: no cover - prefer CuPy when available.
    import cupy as _cp  # type: ignore

    xp = _cp
    IS_CUPY = True
except Exception:  # pragma: no cover - executed on CPU-only environments.
    import numpy as _np  # type: ignore

    xp = _np
    IS_CUPY = False


# ---------------------------------------------------------------------------
# xp utilities
# ---------------------------------------------------------------------------

def seed(s: int) -> None:
    """Seed the random generator of the active xp backend."""

    xp.random.seed(s)


def to_numpy(a: Any):
    """Ensure a host-side NumPy array (for pandas/matplotlib)."""

    if IS_CUPY:
        return _cp.asnumpy(a)  # type: ignore[attr-defined]
    return a


def as_xp(a: Any):
    """Convert a Python/NumPy array to the xp backend array."""

    if IS_CUPY:
        return _cp.asarray(a)  # type: ignore[attr-defined]
    import numpy as _np

    return _np.asarray(a)


def scalar(x: Any) -> float:
    """Convert a 0-d xp array or Python scalar to ``float``."""

    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:  # pragma: no cover - defensive fallback.
            pass
    return float(x)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


@dataclass
class Timer:
    """Simple wall-clock timer context manager."""

    start: float = 0.0
    end: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end = time.perf_counter()

    @property
    def seconds(self) -> float:
        return max(0.0, self.end - self.start)


def pct(a: Iterable[float], q: float) -> float:
    """Tiny percentile helper (for pure Python lists)."""

    arr = list(a)
    if not arr:
        return float("nan")
    arr.sort()
    if q <= 0:
        return float(arr[0])
    if q >= 100:
        return float(arr[-1])
    k = (len(arr) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(arr) - 1)
    return float(arr[f] + (arr[c] - arr[f]) * (k - f))


def bench_repeat(fn: Callable[[], None], repeats: int) -> List[float]:
    """Benchmark helper returning a list of execution times in seconds."""

    times: List[float] = []
    for _ in range(repeats):
        with Timer() as t:
            fn()
        times.append(t.seconds)
    return times


@dataclass(frozen=True)
class PerTokenStats:
    """Percentile statistics for a per-token micro-benchmark."""

    p50_us: float
    p95_us: float
    p99_us: float

    @classmethod
    def from_times(cls, times: Iterable[float], count: int) -> "PerTokenStats":
        per_tok = xp.asarray(list(times), dtype=xp.float64) / max(float(count), 1.0)
        scaled = per_tok * 1e6
        q = xp.asarray([50.0, 95.0, 99.0], dtype=xp.float64)
        p50, p95, p99 = xp.percentile(scaled, q)
        return cls(float(p50), float(p95), float(p99))

    def as_dict(self) -> Dict[str, float]:
        return {
            "per_tok_us_p50": self.p50_us,
            "per_tok_us_p95": self.p95_us,
            "per_tok_us_p99": self.p99_us,
        }


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


Array = Any  # xp.ndarray (NumPy or CuPy)


@dataclass
class BackendInfo:
    name: str
    ok: bool
    note: str


class BackendBase:
    """Abstract math backend interface (xp-agnostic)."""

    name: str = "base"

    def matvec(self, A: Array, x: Array) -> Array:
        return A @ x

    def exp_clip(self, z: Array, clip: float) -> Array:
        return xp.exp(xp.clip(z, -clip, clip))

    def sigmoid(self, z: float) -> float:
        return 1.0 / (1.0 + float(xp.exp(-z)))

    def logits(
        self,
        f_self: Array,
        f_cross: Array,
        W_self: Array,
        W_cross: Array,
        b: Array,
        g: float,
    ) -> Array:
        return g * (f_self @ W_self) + (1.0 - g) * (f_cross @ W_cross) + b


class NumpyBackend(BackendBase):
    name: str = "numpy"


class TensorRTBackend(BackendBase):
    """Probe-only TensorRT backend."""

    name: str = "tensorrt"

    def __init__(self) -> None:
        self.available: bool = False
        self.reason: str = ""
        try:
            if importlib.util.find_spec("tensorrt") is None:
                self.reason = "tensorrt module not found"
            else:  # pragma: no cover - TensorRT unavailable in CI.
                import tensorrt as trt  # type: ignore

                _ = trt.__version__
                self.available = True
                self.reason = (
                    "tensorrt {0} detected (using xp path until engines bound)".format(
                        trt.__version__
                    )
                )
        except Exception as e:  # pragma: no cover - defensive fallback.
            self.reason = f"tensorrt probe failed: {e}"

    def is_ready(self) -> bool:
        return self.available


def pick_backend() -> Tuple[BackendBase, BackendInfo]:
    trt = TensorRTBackend()
    if getattr(trt, "available", False):
        return trt, BackendInfo(name="tensorrt", ok=True, note=trt.reason)
    return (
        NumpyBackend(),
        BackendInfo(name="numpy", ok=False, note=getattr(trt, "reason", "TensorRT not available")),
    )


# ---------------------------------------------------------------------------
# Numerics helpers
# ---------------------------------------------------------------------------


@dataclass
class KahanVec:
    """Kahan compensated sum for vector accumulation (xp)."""

    n: int

    def __post_init__(self) -> None:
        self.s: Array = xp.zeros((self.n,), dtype=xp.float64)
        self.c: Array = xp.zeros((self.n,), dtype=xp.float64)

    def add(self, x: Array) -> None:
        y = x - self.c
        t = self.s + y
        self.c = (t - self.s) - y
        self.s = t


@dataclass
class SERAStab:
    """Self-normalising exponential features with Kahan stabilisation."""

    backend: BackendBase
    d: int = 64
    r: int = 128
    clip: float = 8.0

    def __post_init__(self) -> None:
        self.W: Array = xp.random.normal(size=(self.r, self.d)).astype(xp.float64)
        self.num = KahanVec(self.r)
        self.den = KahanVec(self.r)

    def update(self, x: Array) -> None:
        z = self.backend.matvec(self.W, x) / math.sqrt(self.d)
        f = self.backend.exp_clip(z, self.clip)
        self.num.add(f)
        self.den.add(xp.ones_like(f))

    def digest(self) -> Array:
        den = xp.maximum(self.den.s, 1e-12)
        return self.num.s / den


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


def build_embeddings(V: int = 4096, d: int = 64, seed: int | None = None) -> Array:
    if seed is not None:
        xp.random.seed(seed)
    return xp.random.normal(scale=0.5, size=(V, d)).astype(xp.float64)


@dataclass
class HLMGate:
    backend: BackendBase
    r_self: int
    r_cross: int

    def __post_init__(self) -> None:
        self.w: Array = xp.random.normal(size=(self.r_self + self.r_cross,)).astype(xp.float64)
        self.b: float = float(xp.random.normal())

    def gate(self, f_self: Array, f_cross: Array) -> float:
        z = float(self.w @ xp.concatenate([f_self, f_cross]) + self.b)
        return self.backend.sigmoid(z)


@dataclass
class EncoderSim:
    backend: BackendBase
    E: Array
    d: int = 64
    r: int = 128

    def __post_init__(self) -> None:
        self.sera = SERAStab(self.backend, d=self.d, r=self.r)

    def run_once(self, tokens: Array) -> Array:
        for t in tokens:
            self.sera.update(self.E[int(t)])
        return self.sera.digest()


@dataclass
class GeneratorSim:
    backend: BackendBase
    E: Array
    enc_digest: Array
    V: int = 256
    d: int = 64
    r: int = 128
    r_cross: int = 128
    topk: int = 32
    temp: float = 0.8

    def __post_init__(self) -> None:
        self.sera_self = SERAStab(self.backend, d=self.d, r=self.r)
        self.W_self = xp.random.normal(size=(self.r, self.V)).astype(xp.float64)
        self.W_cross = xp.random.normal(size=(self.r_cross, self.V)).astype(xp.float64)
        self.b = xp.zeros((self.V,), dtype=xp.float64)
        self.gate_mod = HLMGate(self.backend, r_self=self.r, r_cross=self.r_cross)

    def step(self, prev_token: int) -> int:
        x = self.E[int(prev_token)]
        self.sera_self.update(x)
        f_self = self.sera_self.digest()
        f_cross = self.enc_digest
        g = self.gate_mod.gate(f_self, f_cross)
        logits = self.backend.logits(f_self, f_cross, self.W_self, self.W_cross, self.b, g)
        idx = xp.argpartition(-logits, self.topk)[: self.topk]
        z = (logits[idx] - logits[idx].max()) / max(self.temp, 1e-6)
        p = xp.exp(z)
        p /= p.sum()
        return int(xp.random.choice(idx, p=p))

    def run(self, start_token: int, L: int) -> int:
        tok = int(start_token)
        for _ in range(L):
            tok = self.step(tok)
        return tok


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


@dataclass
class BenchConfig:
    enc_V: int = 4096
    enc_d: int = 48
    enc_r: int = 48
    gen_V: int = 128
    gen_d: int = 48
    gen_r: int = 48
    gen_r_cross: int = 32
    topk: int = 16
    temp: float = 0.8
    enc_lengths: Iterable[int] = (512, 2048, 4096)
    gen_lengths: Iterable[int] = (64, 128, 256)
    seed: int = 7


@dataclass(frozen=True)
class BenchRow:
    component: str
    size_key: str
    size_value: int
    backend: str
    stats: PerTokenStats

    def as_dict(self) -> Dict[str, Any]:
        row = {
            "component": self.component,
            "backend": self.backend,
            self.size_key: self.size_value,
        }
        row.update(self.stats.as_dict())
        return row


class BenchRunner:
    """Helper that materialises the micro-benchmark rows."""

    def __init__(self, backend: BackendBase, cfg: BenchConfig) -> None:
        self.backend = backend
        self.cfg = cfg
        seed(cfg.seed)
        self._enc_embeddings = build_embeddings(V=cfg.enc_V, d=cfg.enc_d, seed=cfg.seed)
        self._gen_embeddings = build_embeddings(V=cfg.gen_V, d=cfg.gen_d, seed=cfg.seed + 1)
        self._enc_digest = xp.random.normal(size=(cfg.gen_r_cross,)).astype(xp.float64)

    def run(self) -> pd.DataFrame:
        rows = [*self._encoder_rows(), *self._generator_rows()]
        return pd.DataFrame([row.as_dict() for row in rows])

    def _encoder_rows(self) -> Iterable[BenchRow]:
        for N in self.cfg.enc_lengths:
            tokens = xp.random.randint(0, self.cfg.enc_V, size=(int(N),), dtype=xp.int32)
            enc = EncoderSim(self.backend, self._enc_embeddings, d=self.cfg.enc_d, r=self.cfg.enc_r)

            def work() -> None:
                enc.run_once(tokens)

            stats = PerTokenStats.from_times(bench_repeat(work, 3), int(N))
            yield BenchRow(
                component="Encoder",
                size_key="N_input",
                size_value=int(N),
                backend=self.backend.name,
                stats=stats,
            )

    def _generator_rows(self) -> Iterable[BenchRow]:
        for L in self.cfg.gen_lengths:
            gen = GeneratorSim(
                self.backend,
                self._gen_embeddings,
                self._enc_digest,
                V=self.cfg.gen_V,
                d=self.cfg.gen_d,
                r=self.cfg.gen_r,
                r_cross=self.cfg.gen_r_cross,
                topk=self.cfg.topk,
                temp=self.cfg.temp,
            )
            start = int(xp.random.randint(0, self.cfg.gen_V))

            def work() -> None:
                gen.run(start, int(L))

            stats = PerTokenStats.from_times(bench_repeat(work, 5), int(L))
            yield BenchRow(
                component="Generator",
                size_key="L_output",
                size_value=int(L),
                backend=self.backend.name,
                stats=stats,
            )


def run_bench(backend: BackendBase, cfg: BenchConfig) -> pd.DataFrame:
    return BenchRunner(backend, cfg).run()


def auto_bench(cfg: BenchConfig):
    backend, info = pick_backend()
    df = run_bench(backend, cfg)
    return df, {"backend": info.name, "ok": info.ok, "note": info.note}


# ---------------------------------------------------------------------------
# Self-test helpers
# ---------------------------------------------------------------------------


def _shape_nan_test(bk: BackendBase) -> Dict[str, Any]:
    seed(123)
    E = build_embeddings(256, 32, seed=123)
    enc = EncoderSim(bk, E, d=32, r=24)
    toks = xp.random.randint(0, 256, size=(128,), dtype=xp.int32)
    dig = enc.run_once(toks)
    ok = (dig.shape == (24,)) and bool(xp.isfinite(dig).all())
    return {"name": "encoder_digest_shape_finite", "ok": bool(ok)}


def _generator_step_test(bk: BackendBase) -> Dict[str, Any]:
    seed(124)
    E = build_embeddings(64, 32, seed=124)
    gen = GeneratorSim(
        bk,
        E,
        xp.random.normal(size=(16,)),
        V=64,
        d=32,
        r=24,
        r_cross=16,
        topk=8,
        temp=0.9,
    )
    tok = int(xp.random.randint(0, 64))
    nxt = gen.step(tok)
    ok = 0 <= nxt < 64
    return {"name": "generator_step_range", "ok": bool(ok)}


def _o1_flatness_test(bk: BackendBase) -> Dict[str, Any]:
    df = run_bench(bk, BenchConfig(enc_lengths=(256, 1024, 4096), gen_lengths=(64, 256)))
    enc = df[df["component"] == "Encoder"].sort_values("N_input")
    x0, x1 = enc["N_input"].iloc[0], enc["N_input"].iloc[-1]
    y0, y1 = enc["per_tok_us_p50"].iloc[0], enc["per_tok_us_p50"].iloc[-1]
    slope = (y1 - y0) / max((x1 - x0), 1)
    return {"name": "encoder_o1_flatness", "ok": bool(abs(slope) < 1e-3), "slope": float(slope)}


def self_test() -> Dict[str, Any]:
    bk, info = pick_backend()
    report = {"backend": info.name, "ok": True, "note": info.note, "tests": []}
    for test in (_shape_nan_test, _generator_step_test, _o1_flatness_test):
        out = test(bk)
        report["tests"].append(out)
        report["ok"] = report["ok"] and out["ok"]
    return report


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _save_figs(df, out_dir: Path, backend_name: str) -> dict:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    enc = df[df["component"] == "Encoder"].sort_values("N_input")
    plt.figure()
    plt.plot(enc["N_input"], enc["per_tok_us_p50"], marker="o", label="p50")
    plt.plot(enc["N_input"], enc["per_tok_us_p95"], marker="o", label="p95")
    plt.plot(enc["N_input"], enc["per_tok_us_p99"], marker="o", label="p99")
    plt.xlabel("Input length N")
    plt.ylabel("Time per token (µs)")
    plt.title(f"Encoder per-token vs N ({backend_name})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    enc_fig = out_dir / "encoder_vs_N.png"
    plt.savefig(enc_fig, dpi=160, bbox_inches="tight")
    plt.close()

    gen = df[df["component"] == "Generator"].sort_values("L_output")
    plt.figure()
    plt.plot(gen["L_output"], gen["per_tok_us_p50"], marker="o", label="p50")
    plt.plot(gen["L_output"], gen["per_tok_us_p95"], marker="o", label="p95")
    plt.plot(gen["L_output"], gen["per_tok_us_p99"], marker="o", label="p99")
    plt.xlabel("Output length L")
    plt.ylabel("Time per token (µs)")
    plt.title(f"Generator per-token vs L ({backend_name})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    gen_fig = out_dir / "generator_vs_L.png"
    plt.savefig(gen_fig, dpi=160, bbox_inches="tight")
    plt.close()
    return {"encoder": str(enc_fig), "generator": str(gen_fig)}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="o1stack_single",
        description="O(1) Enc/Gen analysis with CuPy/NumPy xp backend (and TRT-probe).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    ap_bench = sub.add_parser("bench", help="Run microbench and output CSV/figures.")
    ap_bench.add_argument("--enc-lengths", default="512,2048,4096")
    ap_bench.add_argument("--gen-lengths", default="64,128,256")
    ap_bench.add_argument("--out-csv", default="o1_bench.csv")
    ap_bench.add_argument("--out-dir", default=".")

    ap_test = sub.add_parser("selftest", help="Run self-tests and output JSON.")
    ap_test.add_argument("--out-json", default="o1_selftest.json")

    args = parser.parse_args()
    if args.cmd == "bench":
        enc_lengths = [int(x) for x in args.enc_lengths.split(",") if x]
        gen_lengths = [int(x) for x in args.gen_lengths.split(",") if x]
        cfg = BenchConfig(enc_lengths=enc_lengths, gen_lengths=gen_lengths)
        df, info = auto_bench(cfg)
        p = Path(args.out_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        figs = _save_figs(df, Path(args.out_dir), info["backend"])
        readiness = {"backend": info["backend"], "ok": info["ok"], "note": info["note"]}
        print(
            json.dumps(
                {"csv": str(p), "figs": figs, "readiness": readiness},
                indent=2,
            )
        )
    elif args.cmd == "selftest":
        rep = self_test()
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        print(json.dumps({"selftest_json": str(p), "ok": rep.get("ok", False)}, indent=2))


if __name__ == "__main__":
    main()

