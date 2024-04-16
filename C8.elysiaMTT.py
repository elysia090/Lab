
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class DataLoader:
    @staticmethod
    def load_data(file_path, columns):
        try:
            data = pd.read_csv(file_path, usecols=columns)
            return data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except pd.errors.EmptyDataError:
            print(f"File is empty: {file_path}")
            return None
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

class CWLEM:
    def __init__(self, beta_length):
        self.beta = np.zeros(beta_length, dtype=float)

    def set_beta(self, beta):
        self.beta[:len(beta)] = beta

    def predict(self, external_factors):
        return np.dot(external_factors, self.beta)

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

def plot_results(data, actual, predicted, title):
    plt.plot(data['date'], actual, label='Actual')
    plt.plot(data['date'], predicted, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title(title)
    plt.legend()
    plt.show()

def save_predictions(predictions, output_file, start_id=3000888, end_id=3029399):
    with open(output_file, "w") as f:
        f.write("id,sales\n")
        for i, prediction in zip(range(start_id, end_id + 1), predictions):
            f.write(f"{i},{prediction}\n")

def main():
    # Load data
    base_path = '/kaggle/input/store-sales-time-series-forecasting/'
    train_file = 'train.csv'
    obs_files = ['holidays_events.csv', 'oil.csv', 'transactions.csv']
    obs_columns = [['date'], ['date', 'dcoilwtico'], ['date', 'transactions']]
    
    train_data = DataLoader.load_data(base_path + train_file, ['date', 'sales'])
    obs_data = []
    for file, cols in zip(obs_files, obs_columns):
        df = DataLoader.load_data(base_path + file, cols)
        obs_data.append(df)

    if train_data is not None and all(obs is not None for obs in obs_data):
        try:
            # CWLEMとEKFの初期化
            beta_length = 1  # Length of beta vector
            state_dim = 1  # State dimension for Kalman filter
            process_noise_cov = np.eye(state_dim) * 0.1  # Process noise covariance matrix for Kalman filter
            cwlem = CWLEM(beta_length)
            ekf = KalmanFilter(A=np.array([[1]]), B=np.array([[0]]), H=np.array([[1]]), Q=np.eye(1), R=np.eye(1),
                               initial_state_estimate=np.zeros((state_dim, 1)), initial_error_covariance=np.eye(state_dim))

            # Kalman filterのトレーニング
            state_estimates = ekf.run_filter(train_data['sales'])

            # CWLEMによる予測
            cwlem.set_beta(state_estimates[-1])  # Set beta vector to the last state estimate
            cwlem_predicted_sales = cwlem.predict(np.array(state_estimates))[:, 0]

            # 最終的な予測を統合
            combined_sales = np.concatenate((train_data['sales'], cwlem_predicted_sales))

            # 結果のプロット
            ranges = [(0, len(train_data) - 1), (len(train_data), len(train_data) + len(cwlem_predicted_sales) - 1),
                      (0, len(train_data) + len(cwlem_predicted_sales) - 1)]
            titles = ['Actual vs Predicted Sales (Training Data)', 'Actual vs Predicted Sales (Prediction Data)', 'Combined Actual vs Predicted Sales']
            for range_, title in zip(ranges, titles):
                start, end = range_
                plot_results(train_data.iloc[start:end + 1], train_data['sales'].iloc[start:end + 1], combined_sales[start:end + 1], title)

            # 予測をCSVファイルに保存
            output_file = "submission.csv"
            save_predictions(cwlem_predicted_sales, output_file, start_id=len(train_data) + 1, end_id=len(train_data) + len(cwlem_predicted_sales))

            # 予測の評価
            actual_sales = train_data['sales'].iloc[len(train_data):]  # Predictions start from the end of training data
            predicted_sales_trimmed = combined_sales[len(train_data):]  # Trim combined sales to match prediction data
            rmsle = np.sqrt(mean_squared_error(np.log1p(actual_sales), np.log1p(predicted_sales_trimmed)))
            print("RMSLE for actual vs predicted sales (Prediction Data):", rmsle)

        except Exception as e:
            print("An error occurred:", e)
    else:
        print("Train data or observation data is missing or empty.")

if __name__ == "__main__":
    main()





