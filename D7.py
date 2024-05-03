import math

def calculate_theoretical_price_increase_ratio(PER, ROIC, earnings_growth_rate, net_profit_margin, equity_ratio, annual_stock_growth_rate, PBR):
    theoretical_price_increase_ratio = PER * (ROIC + earnings_growth_rate * net_profit_margin) * (equity_ratio ** 2) * math.sqrt(annual_stock_growth_rate) * (1 / math.sqrt(PBR))
    return theoretical_price_increase_ratio

# それぞれの要素を設定
PER = 10
ROIC = 0.15
earnings_growth_rate = 0.05
net_profit_margin = 0.1
equity_ratio = 0.5
annual_stock_growth_rate = 0.1
PBR = 2

# 理論的な株価上昇倍率を計算
theoretical_ratio = calculate_theoretical_price_increase_ratio(PER, ROIC, earnings_growth_rate, net_profit_margin, equity_ratio, annual_stock_growth_rate, PBR)
print("理論的な株価上昇倍率:", theoretical_ratio)
