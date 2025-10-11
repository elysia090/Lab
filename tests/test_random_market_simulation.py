from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

pytest.importorskip("numpy")

from finance.market_simulation.random_market import (
    MarketModel,
    Portfolio,
    TradeStrategy,
    simulate_random_market,
)


def test_trade_strategy_returns_two_entry_signals():
    market_model = MarketModel(initial_price=100)
    strategy = TradeStrategy()
    Portfolio(initial_balance=1000)  # ensure instantiation works alongside the strategy

    time_interval = 4 / (24 * 60)

    for _ in range(5):
        prices = market_model.simulate_price(1, time_interval)
        signals = strategy.evaluate_entry_signal(prices)
        assert isinstance(signals, tuple)
        assert len(signals) == 2
        long_signal, short_signal = signals
        assert isinstance(long_signal, bool)
        assert isinstance(short_signal, bool)


def test_random_market_loop_runs_with_boolean_signals():
    num_simulations = 3
    num_steps = 10

    equity_curves = simulate_random_market(
        num_simulations=num_simulations,
        num_steps=num_steps,
        initial_price=100,
        initial_balance=1000,
    )

    assert len(equity_curves) == num_simulations
    for curve in equity_curves:
        # The portfolio starts with the initial balance and appends one value per step
        assert len(curve) == num_steps + 1
