#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for the LARCH-δRAG harness.

The original repository carried several near-identical copies of the grid harness
spread across ``src/algorithms``.  The authoritative implementation now lives in
``utilities.larch_deltarag_refactored``; this module simply re-exports the public
API so external scripts that ``import LARCH-δRAG`` continue to work without
duplicated code.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

# When executed as a script (``python src/algorithms/LARCH-δRAG.py``), ensure the
# repository ``src`` directory is importable so we can reach ``utilities``.
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utilities.larch_deltarag_refactored import (  # type: ignore E402
    Config,
    Evaluator,
    QueryEngine,
    QueryResult,
    Stats,
    fallback,
    lb_ub,
    preprocess,
    query,
    main as _core_main,
)

Coord = Tuple[int, int]

__all__ = [
    "Config",
    "Evaluator",
    "QueryEngine",
    "QueryResult",
    "Stats",
    "preprocess",
    "query",
    "fallback",
    "lb_ub",
    "main",
]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Delegate to the refactored harness CLI."""
    if argv is None:
        argv = sys.argv[1:]
    return _core_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
