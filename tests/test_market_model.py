from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

# Ensure the src directory is available on the Python path for package imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from finance.market_simulation.market_model import (  # type: ignore  # pylint: disable=import-error
    MarketModel,
    Portfolio,
    SimulationConfig,
    compute_ema,
    compute_rsi,
    plot_equity_curves,
    simulate_equity_curves,
)


def test_market_model_generates_positive_prices():
    model = MarketModel(initial_price=100, seed=123)
    prices = model.generate_path(num_steps=100, time_interval=1 / 252)
    assert prices.shape == (101,)
    assert np.all(prices > 0)


def test_compute_rsi_known_values():
    prices = np.linspace(1, 10, num=20)
    rsi = compute_rsi(prices, period=14)
    assert rsi is not None
    # Strictly increasing sequence should yield maximum RSI.
    assert np.isclose(rsi, 100.0)


def test_compute_ema_matches_numpy():
    prices = np.array([1, 2, 3, 4, 5], dtype=float)
    period = 3
    ema = compute_ema(prices, period=period)
    assert ema is not None
    alpha = 2.0 / (period + 1.0)
    expected = prices[0]
    for price in prices[1:]:
        expected = (price - expected) * alpha + expected
    assert np.isclose(ema, expected)


def test_portfolio_tracks_pnl():
    portfolio = Portfolio(initial_balance=1_000, risk_percentage=0.1)
    size = portfolio.calculate_position_size(price=10)
    portfolio.open_position(price=10, size=size)
    portfolio.update_equity(price=10)
    portfolio.close_position(price=12)
    equity = portfolio.update_equity(price=12)
    assert pytest.approx(equity, rel=1e-9) == portfolio.cash
    assert portfolio.realised_pnl > 0
