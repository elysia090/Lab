"""Entry point for the reusable market simulation helpers."""

from typing import List

from market_simulation import (
    MarketModelConfig,
    PortfolioConfig,
    SimulationConfig,
    run_multiple_simulations,
    plot_equity_curves,
)


def main() -> None:
    simulation_config = SimulationConfig(
        num_steps=1464,
        time_interval=4 / (24 * 60),
        market=MarketModelConfig(initial_price=140),
        portfolio=PortfolioConfig(initial_balance=10_000),
        num_simulations=10,
    )

    results = run_multiple_simulations(simulation_config)
    equity_curves: List[List[float]] = [result.equity_curve for result in results]

    if not plot_equity_curves(equity_curves):
        print("matplotlib is not available; skipped plotting equity curves.")


if __name__ == "__main__":
    main()
