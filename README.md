# Repository Overview

[![Pytest](https://img.shields.io/badge/tests-pytest-blue.svg)](#testing)
[![Coverage](https://img.shields.io/badge/coverage-report-brightgreen.svg)](#testing)
[![Ruff](https://img.shields.io/badge/lint-ruff-5B4FE9.svg)](#testing)

This repository is a grab bag of research prototypes and utilities spanning finance, machine learning, systems, cryptography, quantum computing, and general-purpose tooling.  Each subpackage under `src/` is self-contained and can be imported once the project root is added to `PYTHONPATH`.

## Directory Structure

```
src/
├── finance/                 # Pricing analytics, econometrics pipelines, corporate finance models, and market simulations
├── machine_learning/        # Classical ML experiments, state-space models, NLP pipelines, and time-series demos
├── attention/               # Implementations of custom and fast attention mechanisms
├── compiler_and_systems/    # LLVM→PTX tooling, GPU acceleration utilities, native stubs, and virtual CPU experiments
├── cryptography/            # PLONK-inspired zero-knowledge proof simulations and helpers
├── quantum/                 # Minimal quantum gate simulation utilities
└── utilities/               # Miscellaneous helpers including fractal generators and network/game theory tools
```

Every directory contains an `__init__.py` so modules can be imported directly (for example, `from finance.market_simulation import market_model`).  The repository is intentionally broad; there is no single entry point, so consult individual modules for details and example usage.

## Getting Started

1. Ensure you are using Python 3.9+.
2. Add the project to your module search path, e.g. `export PYTHONPATH="$(pwd)/src:$PYTHONPATH"`.
3. Import the modules you need inside your own scripts or interactive sessions.

## Documentation

Detailed usage notes for the market simulation utilities are available in
`docs/market_simulation.md`.

## Testing

The test suite currently covers the market simulation utilities, including the
analytics helpers introduced in the latest update.

```
pytest
```

Run tests from the repository root after setting `PYTHONPATH` as shown above.
