import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np

# データの読み込み
data = pd.read_csv("C:\\Users\\renta\\testcode\\sample.csv", skiprows=160, encoding='shift_jis')

# 不要な列を削除（NaNで始まる列を除外）
data = data.loc[:, ~data.columns.str.startswith('Unnamed')]

# 不要な行を削除
data = data.dropna()

# 年月のフォーマットを修正
data[data.columns[0]] = pd.to_datetime(data[data.columns[0]], format='%b-%y')

# 日付列をインデックスに設定
data.set_index(data.columns[0], inplace=True)

# 列名を置き換え
new_columns = ['spot_month_end', 'spot_monthly_average', 'spot_center_end', 'spot_center_average', 'spot_monthly_max', 'spot_monthly_min']
data.columns = new_columns

def plot_data(data):
    plt.figure(figsize=(10, 6))
    for column in data.columns:
        plt.plot(data.index, data[column], label=column)
    plt.title('Tokyo Market USD/JPY Spot Exchange Rates')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.xticks(rotation=45)  
    plt.legend()
    plt.show()

def build_arima_model(data, order=(5,1,0), forecast_steps=12):
    model = ARIMA(data['spot_month_end'], order=order, freq='MS')
    fit_model = model.fit()
    predictions = fit_model.forecast(steps=forecast_steps)
    return fit_model, predictions

def plot_predictions(data, fit_model, predictions, zoom=True):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['spot_month_end'], label='Actual')
    plt.plot(predictions.index, predictions, label='Predicted', linestyle='dashed')
    plt.title('Tokyo Market USD/JPY Spot Exchange Rates Prediction')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.xticks(rotation=45)
    plt.legend()
    if zoom:
        plt.figure(figsize=(10, 6))
        plt.plot(data.index[-24:], data['spot_month_end'][-24:], label='Actual')
        plt.plot(predictions.index, predictions, label='Predicted', linestyle='dashed')
        plt.title('Tokyo Market USD/JPY Spot Exchange Rates Prediction (Zoomed)')
        plt.xlabel('Date')
        plt.ylabel('Exchange Rate')
        plt.xticks(rotation=45)
        plt.legend()
    plt.show()

def evaluate_model(actual_values, predictions):
    mse = mean_squared_error(actual_values, predictions)
    print("平均二乗誤差 (MSE) =", mse)

if __name__ == "__main__":
    plot_data(data)
    fit_model, predictions = build_arima_model(data)
    plot_predictions(data, fit_model, predictions)
    actual_values = data['spot_month_end'][-12:]
    evaluate_model(actual_values, predictions)
