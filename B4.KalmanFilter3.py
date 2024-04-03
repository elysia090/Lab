import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, initial_state_estimate, initial_error_covariance):
        self.A = A  # State transition matrix
        self.B = B  # Input matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Covariance matrix of process noise
        self.R = R  # Covariance matrix of observation noise
        self.state_estimate = initial_state_estimate  # Initial state estimate
        self.error_covariance = initial_error_covariance  # Initial error covariance matrix
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
            self.update(np.array([[measurement]]))
            self.state_estimates.append(self.state_estimate.copy())
        return self.state_estimates

    def predict_horizon(self, num_steps):
        predicted_states = []
        state_estimate = self.state_estimate.copy()
        for _ in range(num_steps):
            state_estimate = np.dot(self.A, state_estimate) + self.B
            predicted_states.append(state_estimate)
        return predicted_states

class ExtendedKalmanFilter(KalmanFilter):
    def __init__(self, file_path, A, B, H, Q, R, initial_state_estimate, initial_error_covariance, trend_coefficient):
        super().__init__(A, B, H, Q, R, initial_state_estimate, initial_error_covariance)
        self.file_path = file_path
        self.trend_coefficient = trend_coefficient

    def process_data(self):
        data = pd.read_csv(self.file_path, skiprows=160, encoding='shift_jis')
        data = data.loc[:, ~data.columns.str.startswith('Unnamed')]
        data = data.dropna()
        data[data.columns[0]] = pd.to_datetime(data[data.columns[0]], format='%Y/%m')
        data.set_index(data.columns[0], inplace=True)
        data.index.freq = 'MS'
        new_columns = ['spot_month_end', 'spot_monthly_average', 'spot_center_end', 'spot_center_average', 'spot_monthly_max', 'spot_monthly_min']
        data.columns = new_columns
        return data

    def fit_arima_model(self, data):
        self.arima_model = ARIMA(data['spot_monthly_average'], order=(2, 1, 3))
        self.arima_model_fit = self.arima_model.fit()

    def kalman_filter(self, measurements):
        state_estimates = []
        for measurement in measurements:
            self.state_estimate = np.dot(self.A, self.state_estimate) + self.B
            self.error_covariance = np.dot(np.dot(self.A, self.error_covariance), self.A.T) + self.Q

            innovation = measurement - np.dot(self.H, self.state_estimate)
            innovation_covariance = np.dot(np.dot(self.H, self.error_covariance), self.H.T) + self.R
            kalman_gain = np.dot(np.dot(self.error_covariance, self.H.T), np.linalg.inv(innovation_covariance))
            self.state_estimate = self.state_estimate + np.dot(kalman_gain, innovation)
            self.error_covariance = np.dot((np.eye(self.A.shape[0]) - np.dot(kalman_gain, self.H)), self.error_covariance)

            state_estimates.append(self.state_estimate.copy())
        return state_estimates

    def run(self):
        data = self.process_data()
        self.fit_arima_model(data)
        num_prediction_steps = 12
        predicted_states_arima = self.arima_model_fit.forecast(steps=num_prediction_steps).to_numpy().reshape(-1, 1)
        
        measurements = data['spot_monthly_average'].values[-len(predicted_states_arima):]
        self.state_estimate = np.array([[predicted_states_arima[0][0]], [0]])
        self.error_covariance = np.eye(self.A.shape[0])
        predicted_states_kalman = self.kalman_filter(measurements)
   
        actual_values = data['spot_monthly_average'].values[-len(predicted_states_arima):]

        mae_arima = mean_absolute_error(actual_values, predicted_states_arima)
        mse_arima = mean_squared_error(actual_values, predicted_states_arima)
        mae_kalman = mean_absolute_error(actual_values, [x[0] for x in predicted_states_kalman])
        mse_kalman = mean_squared_error(actual_values, [x[0] for x in predicted_states_kalman])

        print("ARIMA Forecast Evaluation:")
        print(f"Mean Absolute Error (MAE): {mae_arima}")
        print(f"Mean Squared Error (MSE): {mse_arima}")

        print("\nKalman Filter Forecast Evaluation:")
        print(f"Mean Absolute Error (MAE): {mae_kalman}")
        print(f"Mean Squared Error (MSE): {mse_kalman}")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data['spot_monthly_average'], label='Spot Monthly Average')
        ax.plot(data.index[-1] + pd.to_timedelta(np.arange(1, num_prediction_steps + 1), unit='D'), predicted_states_arima, label='ARIMA Predicted State', color='red', linestyle='--')
        ax.plot(data.index[-1] + pd.to_timedelta(np.arange(1, num_prediction_steps + 1), unit='D'), [x[0] for x in predicted_states_kalman], label='Kalman Filter Predicted State', color='blue', linestyle='--')
        ax.set_xlabel('Date')
        ax.set_ylabel('Spot Price')
        ax.set_title('Spot Monthly Average and Predicted States')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

def main():
    file_path = "C:\\Users\\renta\\testcode\\fm08_m_1.csv"
    A = np.eye(2)
    B = np.zeros((2, 1))
    H = np.eye(2)
    Q = np.eye(2)
    R = np.eye(2)

    initial_state_estimate = np.zeros((2, 1))
    initial_error_covariance = np.eye(2)
    trend_coefficient = 0.1

    sarima_kalman_integration = ExtendedKalmanFilter(file_path, A, B, H, Q, R, initial_state_estimate, initial_error_covariance, trend_coefficient)
    sarima_kalman_integration.run()

if __name__ == "__main__":
    main()
