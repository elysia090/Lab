import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_NAME = "A1.MarketModel"
MODULE_PATH = Path(__file__).resolve().parents[1] / "A1.MarketModel.py"
_spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert _spec is not None and _spec.loader is not None
market_model = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = market_model
_spec.loader.exec_module(market_model)

MarketModel = market_model.MarketModel
Portfolio = market_model.Portfolio
SimulationConfig = market_model.SimulationConfig
compute_ema = market_model.compute_ema
compute_rsi = market_model.compute_rsi
plot_equity_curves = market_model.plot_equity_curves
simulate_equity_curves = market_model.simulate_equity_curves


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


def test_simulation_pipeline_deterministic_ranges():
    config = SimulationConfig(
        num_steps=10,
        time_interval=1 / 252,
        initial_price=50,
        initial_balance=5_000,
        num_simulations=3,
        volatility_range=(0.0, 0.0),
        drift_range=(0.0, 0.0),
    )
    curves = simulate_equity_curves(config)
    assert len(curves) == config.num_simulations
    for curve in curves:
        assert curve.ndim == 1
        assert np.all(curve == curve[0])


def test_plot_equity_curves_handles_show_false():
    curves = [np.array([1.0, 1.1, 1.2])]
    # Should not raise when show is False even without a display.
    plot_equity_curves(curves, show=False)


def test_plot_equity_curves_requires_data():
    with pytest.raises(ValueError):
        plot_equity_curves([], show=False)

