#完全な市場ランダムver
from __future__ import annotations

from typing import Iterable

import numpy as np

class MarketModel:
    def __init__(self, initial_price, volatility_range=(0.5, 0.8), drift_range=(0.01, 0.02)):
        self.initial_price = initial_price
        self.volatility_range = volatility_range
        self.drift_range = drift_range
        self.prices = None

    def simulate_price(self, num_steps, time_interval):
        prices = [self.initial_price]
        for _ in range(num_steps):
            volatility = np.random.uniform(*self.volatility_range)
            drift = np.random.uniform(*self.drift_range)
            shock = np.random.normal(loc=drift * time_interval, scale=volatility * np.sqrt(time_interval))
            price = prices[-1] + shock
            prices.append(price)
        self.prices = prices
        return prices


class TradeStrategy:
    def __init__(self, rsi_oversold=30, rsi_overbought=70, moving_average_period=50):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.moving_average_period = moving_average_period

    def evaluate_entry_signal(self, prices):
        if len(prices) < self.moving_average_period:
            return False, False  # データが不足している場合はエントリーシグナルなし

        # Calculate RSI
        changes = np.diff(prices)
        gains = changes[changes >= 0]
        losses = -changes[changes < 0]
        avg_gain = np.mean(gains) if gains.size else 0
        avg_loss = np.mean(losses) if losses.size else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
        rsi = 100 - (100 / (1 + rs))

        # Calculate moving average
        ma = np.mean(prices[-self.moving_average_period:])

        # Calculate MACD
        ema_short = self.calculate_ema(prices, 12)
        ema_long = self.calculate_ema(prices, 26)
        macd_line = ema_short - ema_long

        # Evaluate entry signals
        entry_long_momentum = rsi < self.rsi_oversold and prices[-1] < ma and macd_line[-1] > 0
        entry_short_momentum = rsi > self.rsi_overbought and prices[-1] > ma and macd_line[-1] < 0
        entry_long_mean_reversion = rsi > self.rsi_overbought and prices[-1] < ma and macd_line[-1] > 0
        entry_short_mean_reversion = rsi < self.rsi_oversold and prices[-1] > ma and macd_line[-1] < 0

        entry_long = entry_long_momentum or entry_long_mean_reversion
        entry_short = entry_short_momentum or entry_short_mean_reversion

        return entry_long, entry_short

    def calculate_ema(self, prices, period):
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        alpha = 2 / (period + 1)
        for i in range(1, len(prices)):
            ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
        return ema

class Portfolio:
    def __init__(self, initial_balance, risk_percentage=0.01, stop_loss_multiplier=2.0, volatility_window=14, compound_interest_rate=0.0):
        self.balance = initial_balance
        self.equity_curve = [initial_balance]
        self.trades = []
        self.risk_percentage = risk_percentage
        self.stop_loss_multiplier = stop_loss_multiplier
        self.volatility_window = volatility_window
        self.compound_interest_rate = compound_interest_rate

    def execute_trade(self, price, atr):
        position_size = self.calculate_position_size(price, atr)
        trade_cost = price * position_size
        self.balance -= trade_cost
        self.trades.append((price, position_size))

    def calculate_equity(self, prices):
        equity = self.balance
        for trade in self.trades:
            price, position_size = trade
            equity += position_size * prices[-1]  # 最後の価格を使用する
        self.equity_curve.append(equity)
        return equity

    def update_trades(self, current_price, atr):
        if self.stop_loss_multiplier is not None:
            stop_loss_level = current_price - (self.stop_loss_multiplier * atr)
            updated_trades = []
            for trade in self.trades:
                price, position_size = trade
                if price <= stop_loss_level:
                    # ストップロスレベルに達した場合はトレードをクローズ
                    self.balance += position_size * price
                else:
                    updated_trades.append(trade)
            self.trades = updated_trades

    def calculate_position_size(self, price, atr):
        risk_amount = self.balance * self.risk_percentage
        stop_loss_distance = price - (price - atr)
        position_size = risk_amount / stop_loss_distance
        return position_size

    def apply_compound_interest(self, volatility):
        if self.compound_interest_rate > 0:
            # ボラティリティに応じて再投資の割合を調整する
            dynamic_compound_interest_rate = self.compound_interest_rate * (1 - volatility)
            interest = self.equity_curve[-1] * dynamic_compound_interest_rate
            self.balance += interest
            self.equity_curve[-1] += interest

def simulate_random_market(
    *,
    initial_price: float = 140,
    num_steps: int = 1464,
    time_interval: float | None = None,
    initial_balance: float = 10000,
    num_simulations: int = 4,
    strategy_factory: type[TradeStrategy] | None = None,
    model_factory: type[MarketModel] | None = None,
    portfolio_factory: type[Portfolio] | None = None,
) -> list[list[float]]:
    """Run the random market simulation and return the equity curves."""

    if time_interval is None:
        time_interval = 4 / (24 * 60)

    strategy_factory = strategy_factory or TradeStrategy
    model_factory = model_factory or MarketModel
    portfolio_factory = portfolio_factory or Portfolio

    equity_curves: list[list[float]] = []

    for _ in range(num_simulations):
        market_model = model_factory(initial_price)
        strategy = strategy_factory()
        portfolio = portfolio_factory(initial_balance)

        for _ in range(num_steps):
            prices = market_model.simulate_price(1, time_interval)
            entry_long, entry_short = strategy.evaluate_entry_signal(prices)

            if entry_long:
                position_size = 1  # 仮のポジションサイズ
                portfolio.execute_trade(prices[-1], position_size)
            elif entry_short:
                position_size = -1  # 仮のポジションサイズ
                portfolio.execute_trade(prices[-1], position_size)

            portfolio.calculate_equity(prices)  # 価格のリストを渡す

        equity_curves.append(list(portfolio.equity_curve))

    return equity_curves


def plot_equity_curves(equity_curves: Iterable[Iterable[float]]) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for i, equity_curve in enumerate(equity_curves):
        plt.plot(equity_curve, label=f'Simulation {i+1}')

    plt.xlabel('Time Steps')
    plt.ylabel('Equity')
    plt.title('Equity Curves of Multiple Simulations')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    curves = simulate_random_market()
    plot_equity_curves(curves)
