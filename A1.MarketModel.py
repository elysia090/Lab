from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class MarketModel:
    """Generate synthetic price paths using a simple random walk."""

    def __init__(
        self,
        initial_price: float,
        volatility_range: Tuple[float, float] = (0.2, 0.8),
        drift_range: Tuple[float, float] = (0.03, 0.08),
        seed: Optional[int] = None,
    ) -> None:
        self.initial_price = initial_price
        self.volatility_range = volatility_range
        self.drift_range = drift_range
        self.rng = np.random.default_rng(seed)

    def generate_path(self, num_steps: int, time_interval: float) -> np.ndarray:
        """Simulate a single price path."""

        volatility = self.rng.uniform(*self.volatility_range)
        drift = self.rng.uniform(*self.drift_range)

        prices = [self.initial_price]
        for _ in range(num_steps):
            shock = self.rng.normal(
                loc=drift * time_interval,
                scale=volatility * np.sqrt(time_interval),
            )
            prices.append(prices[-1] + shock)
        return np.asarray(prices)


def compute_rsi(prices: Iterable[float], period: int = 14) -> Optional[float]:
    """Return the Relative Strength Index for the provided price history."""

    prices = np.asarray(prices, dtype=float)
    if prices.size <= period:
        return None

    deltas = np.diff(prices[-(period + 1) :])
    gains = np.clip(deltas, a_min=0, a_max=None)
    losses = np.clip(-deltas, a_min=0, a_max=None)

    avg_gain = gains.mean()
    avg_loss = losses.mean()

    if np.isclose(avg_gain + avg_loss, 0):
        return 50.0
    if np.isclose(avg_loss, 0):
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_ema(prices: Iterable[float], period: int) -> Optional[float]:
    """Return the most recent exponential moving average."""

    prices = np.asarray(prices, dtype=float)
    if prices.size < period:
        return None

    ema_value = prices[0]
    alpha = 2.0 / (period + 1.0)
    for price in prices[1:]:
        ema_value = (price - ema_value) * alpha + ema_value
    return float(ema_value)


class TradeStrategy:
    """Simple technical indicator based entry strategy."""

    def __init__(
        self,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        moving_average_period: int = 50,
        rsi_period: int = 14,
        ema_short_period: int = 12,
        ema_long_period: int = 26,
    ) -> None:
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.moving_average_period = moving_average_period
        self.rsi_period = rsi_period
        self.ema_short_period = ema_short_period
        self.ema_long_period = ema_long_period

    def evaluate_entry_signal(self, prices: Iterable[float]) -> Tuple[bool, bool]:
        prices = np.asarray(prices, dtype=float)
        required_history = max(
            self.moving_average_period,
            self.rsi_period + 1,
            self.ema_long_period,
        )
        if prices.size < required_history:
            return False, False

        rsi = compute_rsi(prices, self.rsi_period)
        ema_short = compute_ema(prices, self.ema_short_period)
        ema_long = compute_ema(prices, self.ema_long_period)

        if rsi is None or ema_short is None or ema_long is None:
            return False, False

        moving_average = prices[-self.moving_average_period :].mean()
        macd = ema_short - ema_long
        latest_price = prices[-1]

        enter_long = (
            rsi <= self.rsi_oversold
            and latest_price < moving_average
            and macd > 0
        )
        enter_short = (
            rsi >= self.rsi_overbought
            and latest_price > moving_average
            and macd < 0
        )
        return enter_long, enter_short


@dataclass
class Position:
    size: float
    entry_price: float


class Portfolio:
    """Track cash, open position, and equity curve."""

    def __init__(self, initial_balance: float, risk_percentage: float = 0.01) -> None:
        self.cash = initial_balance
        self.risk_percentage = risk_percentage
        self.position: Optional[Position] = None
        self.equity_curve = [initial_balance]

    def calculate_position_size(self, price: float) -> float:
        risk_capital = self.cash * self.risk_percentage
        if risk_capital <= 0 or price <= 0:
            return 0.0
        return risk_capital / price

    def open_position(self, price: float, size: float) -> None:
        if np.isclose(size, 0.0):
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


def run_single_simulation(
    num_steps: int,
    time_interval: float,
    initial_price: float,
    initial_balance: float,
    rng_seed: Optional[int] = None,
) -> np.ndarray:
    market_model = MarketModel(initial_price, seed=rng_seed)
    strategy = TradeStrategy()
    portfolio = Portfolio(initial_balance)

    prices = market_model.generate_path(num_steps, time_interval)
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
    return np.asarray(portfolio.equity_curve)


def main() -> None:
    initial_price = 140
    num_steps = 1464
    time_interval = 4 / (24 * 60)
    initial_balance = 10_000
    num_simulations = 10

    equity_curves = [
        run_single_simulation(
            num_steps,
            time_interval,
            initial_price,
            initial_balance,
            rng_seed=seed,
        )
        for seed in range(num_simulations)
    ]

    plt.figure(figsize=(10, 6))
    for i, equity_curve in enumerate(equity_curves, start=1):
        plt.plot(equity_curve, label=f"Simulation {i}")

    plt.xlabel("Time Steps")
    plt.ylabel("Equity")
    plt.title("Equity Curves of Multiple Simulations")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
