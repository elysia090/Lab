import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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
    file_path = "/kaggle/input/store-sales-time-series-forecasting/train.csv"
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])

    # Initialize Kalman filter model
    initial_state_estimate = np.array([[data['sales'].iloc[0]]])
    initial_error_covariance = np.eye(1) * 0.1
    A = np.array([[1]])
    B = np.array([[0]])
    H = np.array([[1]])
    Q = np.eye(1)
    R = np.eye(1)
    kalman_filter = KalmanFilter(A, B, H, Q, R, initial_state_estimate, initial_error_covariance)

    # Train the model using data up to index 1048574
    train_data = data.iloc[:1048575]
    state_estimates = kalman_filter.run_filter(train_data['sales'])

    # Predict sales for the remaining data points
    predicted_sales = []
    for measurement in data['sales'].iloc[1048575:]:
        kalman_filter.predict()
        kalman_filter.update(np.array([[measurement]]))
        predicted_sales.append(kalman_filter.state_estimate[0][0])

    # Combine actual sales and predicted sales
    combined_sales = np.concatenate((train_data['sales'], predicted_sales))

    # Plot results for different ranges
    ranges = [(0, 1048574), (3000888, 3029399), (0, 3029399)]
    titles = ['Actual vs Predicted Sales (0 - 1048574)', 'Actual vs Predicted Sales (3000888 - 3029399)', 'Actual vs Predicted Sales (0 - 3029399)']
    for range_, title in zip(ranges, titles):
        start, end = range_
        plot_results(data.iloc[start:end + 1], data['sales'].iloc[start:end + 1], combined_sales[start:end + 1], title)

    # Save predictions to CSV file
    output_file = "submission.csv"
    save_predictions(predicted_sales, output_file)

    # Calculate RMSLE for actual and predicted sales from index 0 to 1048574
    actual_sales = data['sales'].iloc[:1048575]
    predicted_sales_trimmed = combined_sales[:1048575]
    rmsle = np.sqrt(mean_squared_error(np.log1p(actual_sales), np.log1p(predicted_sales_trimmed)))
    print("RMSLE for actual vs predicted sales (0 - 1048574):", rmsle)

if __name__ == "__main__":
    main()
