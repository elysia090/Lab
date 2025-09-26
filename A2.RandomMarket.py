"""Scenario focused wrapper around the shared market simulation helpers."""

from typing import Iterable, Tuple

from market_simulation import (
    MarketModelConfig,
    PortfolioConfig,
    SimulationConfig,
    StrategyConfig,
    StrategySignals,
    run_multiple_simulations,
    plot_equity_curves,
)


def summarise_signals(signals: Iterable[StrategySignals]) -> Tuple[int, int]:
    long_triggers = 0
    short_triggers = 0
    for signal in signals:
        if signal.enter_long or signal.enter_long_mean_reversion:
            long_triggers += 1
        if signal.enter_short or signal.enter_short_mean_reversion:
            short_triggers += 1
    return long_triggers, short_triggers


def main() -> None:
    simulation_config = SimulationConfig(
        num_steps=1464,
        time_interval=4 / (24 * 60),
        market=MarketModelConfig(
            initial_price=140,
            volatility_range=(0.5, 0.8),
            drift_range=(0.01, 0.02),
        ),
        strategy=StrategyConfig(
            rsi_oversold=25,
            rsi_overbought=75,
            moving_average_period=60,
        ),
        portfolio=PortfolioConfig(
            initial_balance=10_000,
            risk_percentage=0.001,
            stop_loss_multiplier=2.5,
            volatility_window=20,
            compound_interest_rate=0.002,
        ),
        num_simulations=4,
    )

    results = run_multiple_simulations(simulation_config)
    equity_curves = [result.equity_curve for result in results]

    if not plot_equity_curves(equity_curves):
        print("matplotlib is not available; skipped plotting equity curves.")

    print("Simulation summary:")
    for index, result in enumerate(results, start=1):
        long_triggers, short_triggers = summarise_signals(result.signals)
        final_equity = result.equity_curve[-1]
        print(
            f"  Simulation {index}: final equity {final_equity:,.2f}"
            f" (long signals: {long_triggers}, short signals: {short_triggers})"
        )


if __name__ == "__main__":
    main()
