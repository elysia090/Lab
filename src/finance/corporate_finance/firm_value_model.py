import numpy as np

# モンテカルロシミュレーションの回数
N = 1000

# 未来のフリーキャッシュフローの平均と標準偏差
mean_future_cash_flow = 1000
std_future_cash_flow = 100

# オプションの閾値
option_threshold = 900

# 割引率
discount_rate = 0.05

# 予測期間の最終年
n = 5

# シミュレーションの実行
option_values = []
for _ in range(N):
    future_cash_flows = np.random.normal(mean_future_cash_flow, std_future_cash_flow, n)
    option_value = max(future_cash_flows[-1] - option_threshold, 0) / ((1 + discount_rate) ** n)
    option_values.append(option_value)

# オプションの価値の平均を計算
mean_option_value = np.mean(option_values)

# 企業価値を計算
company_value = sum([future_cash_flows[t] / ((1 + discount_rate) ** (t+1)) for t in range(n)]) \
                + mean_option_value

print("企業価値:", company_value)
