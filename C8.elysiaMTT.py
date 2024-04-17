import numpy as np
import pandas as pd

class StateSpaceModel:
    def __init__(self, ar_params):
        self.ar_params = ar_params

    def state_transition_function(self, state, external_factors):
        return np.dot(self.ar_params, state) + external_factors

class ExtendedKalmanFilter:
    def __init__(self, state_space_model, process_noise_cov):
        self.state_space_model = state_space_model
        self.process_noise_cov = process_noise_cov
        self.state_estimation = None
        self.state_covariance = None

    def initialize(self, initial_state, initial_state_cov):
        self.state_estimation = initial_state
        self.state_covariance = initial_state_cov

    def predict(self, external_factors):
        predicted_state = self.state_space_model.state_transition_function(self.state_estimation, external_factors)
        self.state_estimation = predicted_state

    def update(self, observation):
        state_transition_jacobian = np.eye(len(self.state_estimation))
        state_transition_jacobian[-1, -1] = 1
        kalman_gain = np.dot(np.dot(self.state_covariance, state_transition_jacobian.T),
                             np.linalg.inv(np.dot(np.dot(state_transition_jacobian, self.state_covariance), state_transition_jacobian.T) + self.process_noise_cov))
        predicted_state = self.state_space_model.state_transition_function(self.state_estimation, 0)  # External factors are not used here
        self.state_estimation += np.dot(kalman_gain, (observation - predicted_state))
        self.state_covariance = np.dot((np.eye(len(self.state_covariance)) - np.dot(kalman_gain, state_transition_jacobian)), self.state_covariance)

class CliffordLinearEstimationMethod:
    def __init__(self, h):
        self.h = h

    def predict(self, observation):
        return np.dot(self.h.conjugate().T, observation)

class PerformanceEvaluator:
    @staticmethod
    def calculate_rmsle(predictions, true_values):
        return np.sqrt(np.mean(np.square(np.log(predictions + 1) - np.log(true_values + 1))))

class MeanSquareErrorOptimizer:
    def __init__(self):
        pass

    def optimize(self, ekf, cl_method, observations):
        predictions = []
        true_values = []
        for i, observation in enumerate(observations):
            external_factors = np.array([obs_data[j].iloc[i].values[1] for j in range(len(obs_data))])
            ekf.predict(external_factors)
            ekf.update(observation)
            prediction = cl_method.predict(ekf.state_estimation[:-1])
            predictions.append(prediction)
            true_values.append(observation)

        predictions = np.array(predictions)
        true_values = np.array(true_values)

        # 最小二乗法でパラメータを最適化
        h_optimal = np.linalg.lstsq(predictions, true_values, rcond=None)[0]
        return h_optimal

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

def initialize_models(train_data, obs_data):
    state_space_model = StateSpaceModel([])  # No ARIMA parameters
    initial_state = np.zeros(len(train_data.columns) + len(obs_data))
    initial_state_cov = np.eye(len(initial_state))
    process_noise_cov = np.eye(len(initial_state))
    ekf = ExtendedKalmanFilter(state_space_model, process_noise_cov)
    ekf.initialize(initial_state, initial_state_cov)
    h = np.zeros(len(train_data.columns))  # Adjust based on the dimensionality of the state
    cl_method = CliffordLinearEstimationMethod(h)
    return ekf, cl_method

def main():
    base_path = '/kaggle/input/store-sales-time-series-forecasting/'
    train_file = base_path + 'train.csv'
    obs_files = [base_path + 'holidays_events.csv',
                 base_path + 'oil.csv',
                 base_path + 'transactions.csv']
    obs_columns = [['date', 'type'], ['date', 'dcoilwtico'], ['date', 'transactions']]

    train_data = load_data(train_file, ['date', 'sales'])
    obs_data = [load_data(file, cols) for file, cols in zip(obs_files, obs_columns)]

    if train_data is not None and all(obs is not None for obs in obs_data):
        ekf, cl_method = initialize_models(train_data, obs_data)

        mse_optimizer = MeanSquareErrorOptimizer()
        h_optimal = mse_optimizer.optimize(ekf, cl_method, train_data['sales'].values)

        # 最適化されたパラメータを適用
        cl_method.h = h_optimal

        # 予測を生成
        predictions = []
        for i, observation in enumerate(train_data['sales'].values):
            external_factors = np.array([obs_data[j].iloc[i].values[1] for j in range(len(obs_data))])
            ekf.predict(external_factors)
            ekf.update(observation)
            prediction = cl_method.predict(ekf.state_estimation[:-1])
            predictions.append(prediction)

        # RMSLEを計算
        true_sales = train_data['sales'].values
        rmsle = PerformanceEvaluator.calculate_rmsle(np.array(predictions), true_sales)
        print("RMSLE:", rmsle)

    else:
        print("Train data or observation data is missing or empty.")

if __name__ == "__main__":
    main()
