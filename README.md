# Lab

Lab is a public research sandbox for Python prototypes across finance, machine learning, attention systems, cryptography, quantum computing, rendering, and general utilities.

## What This Repository Is

- A broad, exploratory codebase rather than a packaged library.
- `src/` contains standalone modules and subpackages.
- `docs/` contains design notes, working papers, and archived research writeups.
- `tests/` covers the parts of the repository that currently have regression protection.

## Repository Map

- `src/finance`: market simulation, analytics, econometrics, and corporate finance utilities.
- `src/machine_learning`: classical ML, deep learning, NLP, state-space, time-series, and O(1) stack experiments.
- `src/attention`: custom attention, fast attention, and Sera variants.
- `src/compiler_and_systems`: LLVM/PTX, GPU, native, and simulation experiments.
- `src/cryptography`: SymPLONK-style notes and utilities.
- `src/quantum`: minimal gate simulation utilities.
- `src/rendering`: rendering experiments and checked-in visual artifacts.
- `src/utilities`: constant-time algorithms, fractals, graph/network utilities, and misc helpers.
- `docs`: indexed research notes. Start with `docs/README.md`.
- `tests`: pytest coverage for finance, utilities, compiler/system simulation, and cryptography.

## Quickstart

1. Create and activate a Python 3.9+ virtual environment.
2. Install development dependencies with `python -m pip install -r requirements-dev.txt`.
3. Add `src` to `PYTHONPATH`.
   - PowerShell: `$env:PYTHONPATH = "$PWD/src"`
   - Bash: `export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"`
4. Run `python -m pytest` from the repository root.

## Documentation

Use `docs/README.md` as the entry point for the research notes.
Useful starting points:

- `docs/unit-08-market-simulation.md`
- `docs/TCM-Core.md`
- `docs/SDPR.md`
- `docs/VEEN001.md`

## Testing

Current tests exercise:

- finance market simulation and statistics helpers
- constant-time differentiation and periodic convolution utilities
- hypercube CPU and assembler simulation
- SymPLONK serialization and validation behavior

Some modules remain exploratory and do not yet have dedicated tests.

## Notes

- Some filenames are historical and have not been normalized yet.
- Several directories mix polished modules with active experiments; use the docs index when orienting yourself for the first time.
