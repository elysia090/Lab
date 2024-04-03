import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, initial_state_estimate, initial_error_covariance):
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.state_estimate = initial_state_estimate
        self.error_covariance = initial_error_covariance
        self.state_estimates = []

    def predict(self):
        self.state_estimate = np.dot(self.A, self.state_estimate)
        self.error_covariance = np.dot(np.dot(self.A, self.error_covariance), self.A.T) + self.Q

    def update(self, measurement):
        innovation = measurement - np.dot(self.H, self.state_estimate)
        innovation_covariance = np.dot(np.dot(self.H, self.error_covariance), self.H.T) + self.R
        kalman_gain = np.dot(np.dot(self.error_covariance, self.H.T), np.linalg.inv(innovation_covariance))
        self.state_estimate = self.state_estimate + np.dot(kalman_gain, innovation)
        self.error_covariance = np.dot((np.eye(self.A.shape[0]) - np.dot(kalman_gain, self.H)), self.error_covariance)

    def run_filter(self, measurements):
        for measurement in measurements:
            self.predict()
            self.update(measurement)
            self.state_estimates.append(self.state_estimate.copy())
        return self.state_estimates

    def predict_horizon(self, num_steps):
        predicted_states = []
        state_estimate = self.state_estimate.copy()
        for _ in range(num_steps):
            state_estimate = np.dot(self.A, state_estimate) + self.B
            predicted_states.append(state_estimate)
        return predicted_states

class SARIMAKalmanIntegration:
    def __init__(self, file_path):
        self.file_path = file_path
        self.arima_model = None

    def process_data(self):
        data = pd.read_csv(self.file_path, skiprows=160, encoding='shift_jis')
        data = data.loc[:, ~data.columns.str.startswith('Unnamed')]
        data = data.dropna()
        data[data.columns[0]] = pd.to_datetime(data[data.columns[0]], format='%Y/%m')
        data.set_index(data.columns[0], inplace=True)
        new_columns = ['spot_month_end', 'spot_monthly_average', 'spot_center_end', 'spot_center_average', 'spot_monthly_max', 'spot_monthly_min']
        data.columns = new_columns
        measurements = data[['spot_monthly_average', 'spot_center_average']].values
        return data

    def fit_arima_model(self, data):
        # ARIMAモデルの適合
        self.arima_model = ARIMA(data['spot_monthly_average'], order=(2, 1, 3))
        self.arima_model_fit = self.arima_model.fit()

    def evaluate_forecast(self, data, predicted_states):
        # 実際の値
        actual_values = data['spot_monthly_average'].values[-len(predicted_states):]

        # MAEとMSEの計算
        mae = mean_absolute_error(actual_values, predicted_states)
        mse = mean_squared_error(actual_values, predicted_states)

        # 結果の表示
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")

    
    def run(self):
        # データの読み込みと前処理
        data = self.process_data()
        
        # ARIMAモデルの適合
        self.fit_arima_model(data)

        # 予測ホライゾンの時間ステップ数
        num_prediction_steps = 12

         # ARIMAモデルを使用して予測
        predicted_states = self.arima_model_fit.forecast(steps=num_prediction_steps)

        # 結果の可視化
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data['spot_monthly_average'], label='Spot Monthly Average')
        ax.plot(data.index[-1] + pd.to_timedelta(np.arange(1, num_prediction_steps + 1), unit='D'), predicted_states, label='Predicted State', color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Spot Price')
        ax.set_title('Spot Monthly Average and Predicted State')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        # 予測の評価
        self.evaluate_forecast(data, predicted_states)

def main():
    sarima_kalman_integration = SARIMAKalmanIntegration("C:\\Users\\renta\\testcode\\fm08_m_1.csv")
    sarima_kalman_integration.run()

if __name__ == "__main__":
    main()
