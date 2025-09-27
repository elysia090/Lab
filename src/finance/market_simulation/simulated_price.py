import numpy as np
import matplotlib.pyplot as plt

class MarketModel:
    def __init__(self, initial_price, volatility_range=(0.2, 0.5), drift_range=(0.01, 0.02)):
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

# シミュレーションパラメータ
initial_price = 100
num_steps = 1252
time_interval = 1/252

# MarketModelのインスタンス化と価格データの生成
market_model = MarketModel(initial_price)
prices = market_model.simulate_price(num_steps, time_interval)

# 価格データのプロット
plt.plot(prices)
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.title('Simulated Price Data')
plt.grid(True)
plt.show()
