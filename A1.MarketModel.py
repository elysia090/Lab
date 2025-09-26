from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import importlib
import importlib.util
import math
import random
from types import ModuleType


@dataclass(frozen=True)
class MarketModelConfig:
    initial_price: float
    volatility_range: Tuple[float, float] = (0.2, 0.8)
    drift_range: Tuple[float, float] = (0.03, 0.08)


class MarketModel:
    """Generate synthetic price paths using a simple random walk."""

    def __init__(self, config: MarketModelConfig, seed: Optional[int] = None) -> None:
        self.config = config
        self.rng = random.Random(seed)

    def generate_path(self, num_steps: int, time_interval: float) -> List[float]:
        """Simulate a single price path."""

        volatility = self.rng.uniform(*self.config.volatility_range)
        drift = self.rng.uniform(*self.config.drift_range)

        prices = [self.config.initial_price]
        for _ in range(num_steps):
            shock = self.rng.gauss(
                mu=drift * time_interval,
                sigma=volatility * math.sqrt(time_interval),
            )
            prices.append(prices[-1] + shock)
        return prices


def compute_rsi(prices: Iterable[float], period: int = 14) -> Optional[float]:
    """Return the Relative Strength Index for the provided price history."""

    price_list = list(float(price) for price in prices)
    if len(price_list) <= period:
        return None

    recent_prices = price_list[-(period + 1) :]
    deltas = [recent_prices[i + 1] - recent_prices[i] for i in range(period)]
    gains = [max(delta, 0.0) for delta in deltas]
    losses = [max(-delta, 0.0) for delta in deltas]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if math.isclose(avg_gain + avg_loss, 0.0, abs_tol=1e-12):
        return 50.0
    if math.isclose(avg_loss, 0.0, abs_tol=1e-12):
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_ema(prices: Iterable[float], period: int) -> Optional[float]:
    """Return the most recent exponential moving average."""

    price_list = list(float(price) for price in prices)
    if len(price_list) < period:
        return None

    ema_value = price_list[0]
    alpha = 2.0 / (period + 1.0)
    for price in price_list[1:]:
        ema_value = (price - ema_value) * alpha + ema_value
    return float(ema_value)


@dataclass(frozen=True)
class StrategyConfig:
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    moving_average_period: int = 50
    rsi_period: int = 14
    ema_short_period: int = 12
    ema_long_period: int = 26


class TradeStrategy:
    """Simple technical indicator based entry strategy."""

    def __init__(self, config: Optional[StrategyConfig] = None) -> None:
        self.config = config or StrategyConfig()

    def evaluate_entry_signal(self, prices: Iterable[float]) -> Tuple[bool, bool]:
        prices = [float(price) for price in prices]
        required_history = max(
            self.config.moving_average_period,
            self.config.rsi_period + 1,
            self.config.ema_long_period,
        )
        if len(prices) < required_history:
            return False, False

        rsi = compute_rsi(prices, self.config.rsi_period)
        ema_short = compute_ema(prices, self.config.ema_short_period)
        ema_long = compute_ema(prices, self.config.ema_long_period)

        if rsi is None or ema_short is None or ema_long is None:
            return False, False

        recent_prices = prices[-self.config.moving_average_period :]
        moving_average = sum(recent_prices) / len(recent_prices)
        macd = ema_short - ema_long
        latest_price = prices[-1]

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
        return enter_long, enter_short


@dataclass
class Position:
    size: float
    entry_price: float


@dataclass(frozen=True)
class PortfolioConfig:
    initial_balance: float
    risk_percentage: float = 0.01


class Portfolio:
    """Track cash, open position, and equity curve."""

    def __init__(self, config: PortfolioConfig) -> None:
        self.cash = config.initial_balance
        self.config = config
        self.position: Optional[Position] = None
        self.equity_curve = [config.initial_balance]

    def calculate_position_size(self, price: float) -> float:
        risk_capital = self.cash * self.config.risk_percentage
        if risk_capital <= 0 or price <= 0:
            return 0.0
        return risk_capital / price

    def open_position(self, price: float, size: float) -> None:
        if math.isclose(size, 0.0, abs_tol=1e-12):
            return
        self.close_position(price)
        self.cash -= price * size
        self.position = Position(size=size, entry_price=price)

    def close_position(self, price: float) -> None:
        if self.position is None:
            return
        self.cash += price * self.position.size
        self.position = None

    def update_equity(self, price: float) -> float:
        equity = self.cash
        if self.position is not None:
            equity += price * self.position.size
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


def run_single_simulation(
    config: SimulationConfig,
    rng_seed: Optional[int] = None,
) -> SimulationResult:
    market_model = MarketModel(config.market, seed=rng_seed)
    strategy = TradeStrategy(config.strategy)
    portfolio = Portfolio(config.portfolio)

    prices = market_model.generate_path(config.num_steps, config.time_interval)
    price_history = [prices[0]]

    for price in prices[1:]:
        price_history.append(price)
        long_signal, short_signal = strategy.evaluate_entry_signal(price_history)

        if long_signal:
            position_size = portfolio.calculate_position_size(price)
            portfolio.open_position(price, position_size)
        elif short_signal:
            position_size = -portfolio.calculate_position_size(price)
            portfolio.open_position(price, position_size)

        portfolio.update_equity(price)

    portfolio.close_position(prices[-1])
    portfolio.update_equity(prices[-1])
    return SimulationResult(prices=prices, equity_curve=list(portfolio.equity_curve))


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


def _resolve_pyplot() -> Optional[ModuleType]:
    if importlib.util.find_spec("matplotlib") is None:
        return None
    return importlib.import_module("matplotlib.pyplot")


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


def main() -> None:
    simulation_config = SimulationConfig(
        num_steps=1464,
        time_interval=4 / (24 * 60),
        market=MarketModelConfig(initial_price=140),
        portfolio=PortfolioConfig(initial_balance=10_000),
        num_simulations=10,
    )

    results = run_multiple_simulations(simulation_config)
    equity_curves = [result.equity_curve for result in results]

    plotted = plot_equity_curves(equity_curves)
    if not plotted:
        print("matplotlib is not available; skipped plotting equity curves.")


if __name__ == "__main__":
    main()
