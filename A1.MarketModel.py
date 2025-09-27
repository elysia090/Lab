from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration parameters controlling a market simulation run."""

    num_steps: int
    time_interval: float
    initial_price: float = 140.0
    initial_balance: float = 10_000.0
    num_simulations: int = 10
    volatility_range: Tuple[float, float] = (0.2, 0.8)
    drift_range: Tuple[float, float] = (0.03, 0.08)


@dataclass(frozen=True)
class CLIOptions:
    """Container for CLI parsed options."""

    config: SimulationConfig
    output_path: Optional[Path]
    show_plot: bool


class MarketModel:
    """Generate synthetic price paths using a geometric Brownian motion process."""

    def __init__(
        self,
        initial_price: float,
        volatility_range: Tuple[float, float] = (0.2, 0.8),
        drift_range: Tuple[float, float] = (0.03, 0.08),
        seed: Optional[int] = None,
    ) -> None:
        if initial_price <= 0:
            raise ValueError("initial_price must be positive")

        self.initial_price = float(initial_price)
        self.volatility_range = volatility_range
        self.drift_range = drift_range
        self.rng = np.random.default_rng(seed)

    def generate_path(self, num_steps: int, time_interval: float) -> np.ndarray:
        """Simulate a single price path using a discretised GBM process.

        Args:
            num_steps: Number of simulation steps to evolve.
            time_interval: Size of each time increment in fractions of a day.

        Returns:
            A ``numpy.ndarray`` containing strictly positive prices.
        """

        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if time_interval <= 0:
            raise ValueError("time_interval must be positive")

        volatility = self.rng.uniform(*self.volatility_range)
        drift = self.rng.uniform(*self.drift_range)

        prices = np.empty(num_steps + 1, dtype=float)
        prices[0] = self.initial_price

        # Use the exact solution of GBM so prices remain positive.
        for step in range(1, num_steps + 1):
            normal_shock = self.rng.normal()
            drift_term = (drift - 0.5 * volatility**2) * time_interval
            diffusion_term = volatility * np.sqrt(time_interval) * normal_shock
            prices[step] = prices[step - 1] * np.exp(drift_term + diffusion_term)
        return prices


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
    """Simple technical-indicator based entry strategy.

    The strategy combines RSI, a simple moving average, and a MACD-like
    signal to decide whether to enter a long or short position.
    """

    def __init__(
        self,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        moving_average_period: int = 50,
        rsi_period: int = 14,
        ema_short_period: int = 12,
        ema_long_period: int = 26,
    ) -> None:
        if not 0 <= rsi_oversold <= rsi_overbought <= 100:
            raise ValueError("RSI thresholds must satisfy 0 ≤ oversold ≤ overbought ≤ 100")
        if moving_average_period <= 0:
            raise ValueError("moving_average_period must be positive")
        if rsi_period <= 1:
            raise ValueError("rsi_period must be greater than 1 to compute differences")
        if ema_short_period <= 1 or ema_long_period <= 1:
            raise ValueError("EMA periods must be greater than 1")

        self.rsi_oversold = float(rsi_oversold)
        self.rsi_overbought = float(rsi_overbought)
        self.moving_average_period = int(moving_average_period)
        self.rsi_period = int(rsi_period)
        self.ema_short_period = int(ema_short_period)
        self.ema_long_period = int(ema_long_period)

    def evaluate_entry_signal(self, prices: Iterable[float]) -> Tuple[bool, bool]:
        """Return long/short entry signals for the latest price history."""

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

        moving_average = float(prices[-self.moving_average_period :].mean())
        macd = ema_short - ema_long
        latest_price = float(prices[-1])

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


@dataclass(frozen=True)
class Position:
    """Represents an open position in the simulated portfolio."""

    size: float
    entry_price: float

    @property
    def direction(self) -> str:
        return "long" if self.size >= 0 else "short"


class Portfolio:
    """Track cash, open position, realised PnL and equity curve."""

    def __init__(
        self,
        initial_balance: float,
        risk_percentage: float = 0.01,
    ) -> None:
        if initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        if not 0 < risk_percentage <= 1:
            raise ValueError("risk_percentage must be in the interval (0, 1]")

        self.cash = float(initial_balance)
        self.risk_percentage = float(risk_percentage)
        self.position: Optional[Position] = None
        self.equity_curve: List[float] = [float(initial_balance)]
        self.realised_pnl = 0.0

    def calculate_position_size(self, price: float) -> float:
        """Determine position size based on risk settings."""

        risk_capital = self.cash * self.risk_percentage
        if risk_capital <= 0 or price <= 0:
            return 0.0
        return risk_capital / price

    def open_position(self, price: float, size: float) -> None:
        """Open a new position, closing any existing one first."""

        if np.isclose(size, 0.0):
            return
        self.close_position(price)
        self.cash -= price * size
        self.position = Position(size=float(size), entry_price=float(price))

    def close_position(self, price: float) -> None:
        """Close the currently open position and realise PnL."""

        if self.position is None:
            return
        pnl = (price - self.position.entry_price) * self.position.size
        self.cash += price * self.position.size
        self.realised_pnl += pnl
        self.position = None

    def update_equity(self, price: float) -> float:
        """Update and return the latest portfolio equity."""

        equity = self.cash
        if self.position is not None:
            equity += price * self.position.size
        self.equity_curve.append(float(equity))
        return float(equity)


def run_single_simulation(
    config: SimulationConfig,
    rng_seed: Optional[int] = None,
    strategy: Optional[TradeStrategy] = None,
) -> np.ndarray:
    """Run a single equity curve simulation.

    Args:
        config: Parameters controlling the simulation horizon and balances.
        rng_seed: Seed for deterministic reproducibility.
        strategy: Optional trading strategy. A default strategy is used when
            omitted.

    Returns:
        Array containing the simulated equity curve.
    """

    market_model = MarketModel(
        config.initial_price,
        volatility_range=config.volatility_range,
        drift_range=config.drift_range,
        seed=rng_seed,
    )
    strategy = strategy or TradeStrategy()
    portfolio = Portfolio(config.initial_balance)

    prices = market_model.generate_path(config.num_steps, config.time_interval)
    price_history: List[float] = [float(prices[0])]

    for price in prices[1:]:
        price_history.append(float(price))
        long_signal, short_signal = strategy.evaluate_entry_signal(price_history)

        if long_signal:
            position_size = portfolio.calculate_position_size(price)
            portfolio.open_position(price, position_size)
        elif short_signal:
            position_size = -portfolio.calculate_position_size(price)
            portfolio.open_position(price, position_size)

        portfolio.update_equity(price)

    portfolio.close_position(float(prices[-1]))
    portfolio.update_equity(float(prices[-1]))
    return np.asarray(portfolio.equity_curve)


def simulate_equity_curves(
    config: SimulationConfig,
    strategy: Optional[TradeStrategy] = None,
) -> List[np.ndarray]:
    """Run multiple simulations and collect their equity curves."""

    return [
        run_single_simulation(config, rng_seed=seed, strategy=strategy)
        for seed in range(config.num_simulations)
    ]


def plot_equity_curves(equity_curves: Sequence[np.ndarray], *, show: bool = True) -> None:
    """Plot one or more equity curves."""

    if not equity_curves:
        raise ValueError("No equity curves supplied for plotting")

    figure = plt.figure(figsize=(10, 6))
    for i, equity_curve in enumerate(equity_curves, start=1):
        plt.plot(equity_curve, label=f"Simulation {i}")

    plt.xlabel("Time Steps")
    plt.ylabel("Equity")
    plt.title("Equity Curves of Multiple Simulations")
    plt.legend()
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.close(figure)


def save_equity_curves(equity_curves: Sequence[np.ndarray], output_path: Path) -> None:
    """Persist the simulated equity curves to disk as a NumPy binary file."""

    stacked = np.vstack(equity_curves)
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, stacked)


def parse_arguments() -> CLIOptions:
    """Create CLI configuration from command-line arguments."""

    parser = argparse.ArgumentParser(description="Simulate technical trading strategies")
    parser.add_argument("num_steps", type=int, help="Number of time steps per simulation")
    parser.add_argument(
        "--time-interval",
        type=float,
        default=4 / (24 * 60),
        help="Length of each simulation step (fraction of a day)",
    )
    parser.add_argument("--initial-price", type=float, default=140.0)
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--num-simulations", type=int, default=10)
    parser.add_argument(
        "--volatility-range",
        type=float,
        nargs=2,
        default=(0.2, 0.8),
        metavar=("LOW", "HIGH"),
        help="Range for sampling annualised volatility",
    )
    parser.add_argument(
        "--drift-range",
        type=float,
        nargs=2,
        default=(0.03, 0.08),
        metavar=("LOW", "HIGH"),
        help="Range for sampling annualised drift",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional .npy file to save the equity curves",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting the equity curves (useful for headless environments)",
    )
    args = parser.parse_args()

    config = SimulationConfig(
        num_steps=args.num_steps,
        time_interval=args.time_interval,
        initial_price=args.initial_price,
        initial_balance=args.initial_balance,
        num_simulations=args.num_simulations,
        volatility_range=tuple(args.volatility_range),
        drift_range=tuple(args.drift_range),
    )

    return CLIOptions(config=config, output_path=args.output, show_plot=not args.no_plot)


def main() -> None:
    options = parse_arguments()
    equity_curves = simulate_equity_curves(options.config)

    if options.output_path is not None:
        save_equity_curves(equity_curves, options.output_path)

    if options.show_plot:
        plot_equity_curves(equity_curves)



if __name__ == "__main__":
    main()
