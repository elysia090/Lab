# Market Simulation Toolkit

This document provides a product-focused overview of the technical trading
simulation utilities located in `finance.market_simulation`. The toolkit is
designed to support rapid experimentation with stochastic price paths and
indicator-driven strategies, while offering reproducible reporting for
portfolio outcomes.

## Quick Start

1. Install the project dependencies and add `src/` to your `PYTHONPATH`.
2. Run a batch simulation via the CLI:

   ```bash
   python -m finance.market_simulation.market_model 390 --no-plot
   ```

   This command simulates 390 steps (roughly one trading day at one-minute
   resolution) using the default configuration. The summary table printed to
   stdout provides key performance metrics for each simulation run.

3. Persist the equity curves for further analysis:

   ```bash
   python -m finance.market_simulation.market_model 780 --output results.npy
   ```

   Saved arrays can be reloaded with `numpy.load("results.npy")` for custom
   analysis pipelines or plotting routines.

## Core Components

### `MarketModel`

Generates synthetic price paths via geometric Brownian motion. The model
exposes parameters for volatility, drift, and seeding to ensure deterministic
replays during testing.

Mathematical form:

* Price dynamics follow the discrete-time GBM update
  \(S_{t+1} = S_t \cdot \exp\big((\mu - \tfrac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}\,\varepsilon_t\big)\),
  where \(\varepsilon_t \sim \mathcal{N}(0,1)\) and \(\Delta t\) is the step size supplied in `SimulationConfig`.
* Log returns are therefore \(r_{t+1} = \log(S_{t+1}/S_t) = (\mu - \tfrac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}\,\varepsilon_t\).

Indicator calculations exposed via the default strategy are fully specified inside the document to avoid cross-references:

* Exponential moving average (EMA) with decay \(\alpha\):
  \(\text{EMA}_{t} = \alpha p_t + (1-\alpha) \text{EMA}_{t-1}\) with \(\text{EMA}_0 = p_0\).
* Relative Strength Index (RSI) over window \(n\):
  average gains \(G_t = (1-1/n) G_{t-1} + \max(r_t,0)/n\),
  average losses \(L_t = (1-1/n) L_{t-1} + \max(-r_t,0)/n\),
  \(\text{RSI}_t = 100 - 100/(1 + G_t / (L_t + \epsilon))\) with \(\epsilon\) preventing division by zero.
* Moving-average filter on signals uses the simple average
  \(\text{MA}_t = \tfrac{1}{m} \sum_{i=0}^{m-1} s_{t-i}\) over window length \(m\).

### `TradeStrategy`

Implements an indicator-driven entry strategy that combines RSI, exponential
moving averages, and a moving-average filter. Signals can be replaced with
custom logic by injecting a subclass into `run_single_simulation`.

### `Portfolio`

Applies simple risk budgeting, position management, and equity tracking. The
class handles both long and short entries and maintains an equity curve for the
backtest horizon.

## Performance Analytics

The module now provides first-class analytics via
`compute_equity_statistics`, `summarise_equity_curves`, and
`format_statistics_table`.

* `compute_equity_statistics` returns an `EquityCurveStatistics` dataclass with
  total return, CAGR, volatility, max drawdown, and Sharpe ratio fields.
* `summarise_equity_curves` applies the calculation to an arbitrary collection
  of simulated curves.
* `format_statistics_table` presents the results in a CLI-friendly table.

Example usage:

```python
from finance.market_simulation.market_model import (
    SimulationConfig,
    simulate_equity_curves,
    summarise_equity_curves,
    format_statistics_table,
)

config = SimulationConfig(num_steps=252, time_interval=1/252)
curves = simulate_equity_curves(config)
stats = summarise_equity_curves(curves, periods_per_year=252, risk_free_rate=0.02)
print(format_statistics_table(stats))
```

## Testing Strategy

Automated tests validate the new analytics against deterministic equity curves
and ensure percentage formatting remains stable for CLI consumers. Run the
suite with `pytest` from the repository root.
