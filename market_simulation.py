"""Core helpers for reusable market simulations.

This module centralises the shared logic that powers the assorted market
simulation scripts that live at the root of the repository.  The original
implementations in :mod:`A1.MarketModel` and :mod:`A2.RandomMarket` drifted apart
over time and even developed incompatible interfaces.  By extracting the
classes, configuration dataclasses, and indicator utilities into a common
module we resolve those behavioural conflicts and make future refactors much
easier to stage incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import importlib.util
import math
import random
import statistics
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "MarketModelConfig",
    "MarketModel",
    "StrategyConfig",
    "StrategySignals",
    "TradeStrategy",
    "PortfolioConfig",
    "Portfolio",
    "SimulationConfig",
    "SimulationResult",
    "run_single_simulation",
    "run_multiple_simulations",
    "plot_equity_curves",
]


# ---------------------------------------------------------------------------
# Utility resolvers


def _resolve_numpy():
    """Return the ``numpy`` module when available, otherwise ``None``."""

    if importlib.util.find_spec("numpy") is None:
        return None
    return importlib.import_module("numpy")


def _resolve_pyplot():
    """Return ``matplotlib.pyplot`` when available."""

    if importlib.util.find_spec("matplotlib") is None:
        return None
    return importlib.import_module("matplotlib.pyplot")


# ---------------------------------------------------------------------------
# Market model


@dataclass(frozen=True)
class MarketModelConfig:
    """Configuration for synthetic price generation."""

    initial_price: float
    volatility_range: Tuple[float, float] = (0.2, 0.8)
    drift_range: Tuple[float, float] = (0.03, 0.08)


class MarketModel:
    """Generate synthetic price paths using a Gaussian random walk."""

    def __init__(self, config: MarketModelConfig, seed: Optional[int] = None) -> None:
        self.config = config
        self._random = random.Random(seed)
        self._np = _resolve_numpy()
        self._np_rng = self._np.random.default_rng(seed) if self._np is not None else None

    def generate_path(self, num_steps: int, time_interval: float) -> List[float]:
        """Simulate a single price path."""

        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")

        volatility = self._random.uniform(*self.config.volatility_range)
        drift = self._random.uniform(*self.config.drift_range)

        prices = [float(self.config.initial_price)]
        if num_steps == 0:
            return prices

        dt = float(time_interval)
        sigma = volatility * math.sqrt(dt)
        mu = drift * dt

        if self._np_rng is not None:
            shocks = self._np_rng.normal(loc=mu, scale=sigma, size=num_steps)
            cumulative = shocks.cumsum()
            prices.extend((self.config.initial_price + cumulative).tolist())
            return prices

        price = float(self.config.initial_price)
        for _ in range(num_steps):
            price += self._random.gauss(mu, sigma)
            prices.append(price)
        return prices


# ---------------------------------------------------------------------------
# Indicator helpers


def compute_rsi(prices: Iterable[float], period: int = 14) -> Optional[float]:
    """Return the Relative Strength Index for the provided price history."""

    price_list = [float(price) for price in prices]
    if len(price_list) <= period:
        return None

    recent_prices = price_list[-(period + 1) :]
    deltas = [recent_prices[i + 1] - recent_prices[i] for i in range(period)]
    gains = [max(delta, 0.0) for delta in deltas]
    losses = [max(-delta, 0.0) for delta in deltas]

    avg_gain = statistics.fmean(gains) if gains else 0.0
    avg_loss = statistics.fmean(losses) if losses else 0.0

    if math.isclose(avg_gain + avg_loss, 0.0, abs_tol=1e-12):
        return 50.0
    if math.isclose(avg_loss, 0.0, abs_tol=1e-12):
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_ema_series(prices: Sequence[float], period: int) -> List[float]:
    if not prices:
        return []
    ema_values = [float(prices[0])]
    alpha = 2.0 / (period + 1.0)
    for price in prices[1:]:
        ema_values.append((price - ema_values[-1]) * alpha + ema_values[-1])
    return ema_values


def compute_ema(prices: Iterable[float], period: int) -> Optional[float]:
    """Return the most recent exponential moving average."""

    price_list = [float(price) for price in prices]
    if len(price_list) < period:
        return None
    ema_values = _compute_ema_series(price_list, period)
    return float(ema_values[-1])


def compute_average_true_range(prices: Sequence[float], window: int) -> Optional[float]:
    """Approximate the Average True Range from closing prices."""

    if window <= 0:
        return None

    price_list = [float(price) for price in prices]
    if len(price_list) < 2:
        return None

    true_ranges = [abs(price_list[i] - price_list[i - 1]) for i in range(1, len(price_list))]
    if not true_ranges:
        return None

    if len(true_ranges) >= window:
        windowed = true_ranges[-window:]
    else:
        windowed = true_ranges
    return float(statistics.fmean(windowed))


# ---------------------------------------------------------------------------
# Strategy


@dataclass(frozen=True)
class StrategyConfig:
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    moving_average_period: int = 50
    rsi_period: int = 14
    ema_short_period: int = 12
    ema_long_period: int = 26


@dataclass(frozen=True)
class StrategySignals:
    enter_long: bool = False
    enter_short: bool = False
    enter_long_mean_reversion: bool = False
    enter_short_mean_reversion: bool = False

    def any_signal(self) -> bool:
        return any((
            self.enter_long,
            self.enter_short,
            self.enter_long_mean_reversion,
            self.enter_short_mean_reversion,
        ))


class TradeStrategy:
    """Simple technical indicator based entry strategy."""

    def __init__(self, config: Optional[StrategyConfig] = None) -> None:
        self.config = config or StrategyConfig()

    def evaluate_signals(self, prices: Iterable[float]) -> StrategySignals:
        price_list = [float(price) for price in prices]

        required_history = max(
            self.config.moving_average_period,
            self.config.rsi_period + 1,
            self.config.ema_long_period,
        )
        if len(price_list) < required_history:
            return StrategySignals()

        rsi = compute_rsi(price_list, self.config.rsi_period)
        ema_short_series = _compute_ema_series(price_list, self.config.ema_short_period)
        ema_long_series = _compute_ema_series(price_list, self.config.ema_long_period)

        if rsi is None or len(ema_short_series) < len(price_list) or len(ema_long_series) < len(price_list):
            return StrategySignals()

        ema_short = float(ema_short_series[-1])
        ema_long = float(ema_long_series[-1])
        macd = ema_short - ema_long

        recent_prices = price_list[-self.config.moving_average_period :]
        moving_average = statistics.fmean(recent_prices)
        latest_price = price_list[-1]

        enter_long = (
            rsi <= self.config.rsi_oversold
            and latest_price < moving_average
            and macd > 0
        )
        enter_short = (
            rsi >= self.config.rsi_overbought
            and latest_price > moving_average
            and macd < 0
        )
        enter_long_mean_reversion = (
            rsi >= self.config.rsi_overbought
            and latest_price < moving_average
            and macd > 0
        )
        enter_short_mean_reversion = (
            rsi <= self.config.rsi_oversold
            and latest_price > moving_average
            and macd < 0
        )

        return StrategySignals(
            enter_long=enter_long,
            enter_short=enter_short,
            enter_long_mean_reversion=enter_long_mean_reversion,
            enter_short_mean_reversion=enter_short_mean_reversion,
        )


# ---------------------------------------------------------------------------
# Portfolio and simulation


@dataclass(frozen=True)
class PortfolioConfig:
    initial_balance: float
    risk_percentage: float = 0.01
    stop_loss_multiplier: float = 2.0
    volatility_window: int = 14
    compound_interest_rate: float = 0.0


@dataclass(frozen=True)
class Position:
    size: float
    entry_price: float


class Portfolio:
    """Track cash, open positions, and the equity curve."""

    def __init__(self, config: PortfolioConfig) -> None:
        self.config = config
        self.cash = float(config.initial_balance)
        self.positions: List[Position] = []
        self.equity_curve: List[float] = [float(config.initial_balance)]

    def calculate_position_size(self, price: float, volatility: Optional[float] = None) -> float:
        risk_capital = self.cash * self.config.risk_percentage
        if risk_capital <= 0:
            return 0.0

        if volatility is None or math.isclose(volatility, 0.0, abs_tol=1e-12):
            denominator = max(price, 1e-6)
            return risk_capital / denominator

        stop_distance = max(
            volatility * self.config.stop_loss_multiplier,
            price * 0.01,
            1e-6,
        )
        return risk_capital / stop_distance

    def open_position(self, price: float, size: float) -> None:
        if math.isclose(size, 0.0, abs_tol=1e-12):
            return
        if size > 0:
            max_affordable_size = self.cash / max(price, 1e-6)
            size = min(size, max_affordable_size)
            if math.isclose(size, 0.0, abs_tol=1e-12):
                return
        cost = price * size
        self.cash -= cost
        self.positions.append(Position(size=size, entry_price=price))

    def close_all(self, price: float) -> None:
        if not self.positions:
            return
        for position in self.positions:
            self.cash += price * position.size
        self.positions.clear()

    def manage_risk(self, current_price: float, volatility: Optional[float]) -> None:
        if volatility is None or not self.positions:
            return

        updated_positions: List[Position] = []
        threshold = volatility * self.config.stop_loss_multiplier
        for position in self.positions:
            if position.size > 0:
                stop_loss = position.entry_price - threshold
                if current_price <= stop_loss:
                    self.cash += current_price * position.size
                    continue
            else:
                stop_loss = position.entry_price + threshold
                if current_price >= stop_loss:
                    self.cash += current_price * position.size
                    continue
            updated_positions.append(position)
        self.positions = updated_positions

    def apply_compound_interest(self, volatility: Optional[float]) -> None:
        rate = self.config.compound_interest_rate
        if rate <= 0 or not self.equity_curve:
            return

        if volatility is None:
            dynamic_rate = rate
        else:
            dynamic_rate = rate * max(0.0, 1.0 - min(volatility, 1.0))

        interest = self.equity_curve[-1] * dynamic_rate
        self.cash += interest
        self.equity_curve[-1] += interest

    def update_equity(self, price: float) -> float:
        equity = self.cash
        if self.positions:
            equity += price * sum(position.size for position in self.positions)
        self.equity_curve.append(equity)
        return equity


@dataclass(frozen=True)
class SimulationConfig:
    num_steps: int
    time_interval: float
    market: MarketModelConfig
    portfolio: PortfolioConfig
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    num_simulations: int = 1


@dataclass
class SimulationResult:
    prices: List[float]
    equity_curve: List[float]
    signals: List[StrategySignals]


def run_single_simulation(
    config: SimulationConfig,
    rng_seed: Optional[int] = None,
) -> SimulationResult:
    market_model = MarketModel(config.market, seed=rng_seed)
    strategy = TradeStrategy(config.strategy)
    portfolio = Portfolio(config.portfolio)

    prices = market_model.generate_path(config.num_steps, config.time_interval)
    price_history = [prices[0]]
    signals_log: List[StrategySignals] = []

    for price in prices[1:]:
        price_history.append(price)
        signals = strategy.evaluate_signals(price_history)
        signals_log.append(signals)

        atr = compute_average_true_range(
            price_history,
            config.portfolio.volatility_window,
        )

        portfolio.manage_risk(price, atr)

        direction = 0.0
        if signals.enter_long or signals.enter_long_mean_reversion:
            direction = 1.0
        elif signals.enter_short or signals.enter_short_mean_reversion:
            direction = -1.0

        if direction:
            position_size = portfolio.calculate_position_size(price, atr) * direction
            portfolio.open_position(price, position_size)

        portfolio.update_equity(price)
        portfolio.apply_compound_interest(atr)

    portfolio.close_all(prices[-1])
    portfolio.update_equity(prices[-1])

    return SimulationResult(
        prices=prices,
        equity_curve=list(portfolio.equity_curve),
        signals=signals_log,
    )


def run_multiple_simulations(
    config: SimulationConfig,
    seeds: Optional[Sequence[Optional[int]]] = None,
) -> List[SimulationResult]:
    if seeds is None:
        seeds_iter: List[Optional[int]] = list(range(config.num_simulations))
    else:
        seeds_iter = list(seeds)

    if not seeds_iter:
        seeds_iter = [None]

    results: List[SimulationResult] = []
    for seed in seeds_iter:
        results.append(run_single_simulation(config, rng_seed=seed))
    return results


def plot_equity_curves(equity_curves: Sequence[Sequence[float]]) -> bool:
    if not equity_curves:
        return False

    plt = _resolve_pyplot()
    if plt is None:
        return False

    plt.figure(figsize=(10, 6))
    for i, equity_curve in enumerate(equity_curves, start=1):
        plt.plot(list(equity_curve), label=f"Simulation {i}")

    plt.xlabel("Time Steps")
    plt.ylabel("Equity")
    plt.title("Equity Curves of Multiple Simulations")
    plt.legend()
    plt.grid(True)
    plt.show()
    return True

