import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ModelUpdater:
    @staticmethod
    def preprocess_data(file_path, columns):
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

class StateSpaceModel:
    def __init__(self, ar_params, ma_params):
        self.ar_params = ar_params
        self.ma_params = ma_params

    def state_transition_function(self, state):
        return np.dot(self.ar_params, state)

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
        predicted_state = self.state_space_model.state_transition_function(self.state_estimation)
        self.state_estimation = predicted_state + np.dot(external_factors, self.state_space_model.ar_params)

    def update(self, observation):
        state_transition_jacobian = np.eye(len(self.state_estimation))
        state_transition_jacobian[-1, -1] = 1
        kalman_gain = np.dot(np.dot(self.state_covariance, state_transition_jacobian.T),
                             np.linalg.inv(np.dot(np.dot(state_transition_jacobian, self.state_covariance), state_transition_jacobian.T) + self.process_noise_cov))
        predicted_state = self.state_space_model.state_transition_function(self.state_estimation)
        self.state_estimation += np.dot(kalman_gain, (observation - predicted_state))
        self.state_covariance = np.dot((np.eye(len(self.state_covariance)) - np.dot(kalman_gain, state_transition_jacobian)), self.state_covariance)

class CWLEM:
    def __init__(self, beta):
        self.beta = beta

    def predict(self, external_factors):
        return np.dot(external_factors, self.beta)

class PerformanceEvaluator:
    @staticmethod
    def calculate_rmsle(predictions, true_values):
        return np.sqrt(np.mean(np.square(np.log(predictions + 1) - np.log(true_values + 1))))

def preprocess_data(file_path, columns):
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

def calculate_feature_importance(obs_data):
    # 外部要因同士の相関行列を計算
    correlation_matrix = np.corrcoef([obs_data[i].iloc[:, 1] for i in range(len(obs_data))])
    # 各外部要因の重要度を計算（相関係数の絶対値の平均）
    feature_importance = np.mean(np.abs(correlation_matrix), axis=1)
    return feature_importance

def auto_adjust_weights(obs_data):
    # 外部要因の重要度を計算
    feature_importance = calculate_feature_importance(obs_data)
    # 重要度を基に、各外部要因に対する重みを決定
    total_importance = np.sum(feature_importance)
    weights = feature_importance / total_importance
    return weights

def initialize_models(train_data, obs_data):
    state_space_model = StateSpaceModel([], [])  # ARIMA パラメータ不要
    initial_state = np.zeros(len(train_data.columns) + len(obs_data))
    initial_state_cov = np.eye(len(initial_state))
    process_noise_cov = np.eye(len(initial_state))
    ekf = ExtendedKalmanFilter(state_space_model, process_noise_cov)
    ekf.initialize(initial_state, initial_state_cov)
    # 自動で重み付けを調整
    weights = auto_adjust_weights(obs_data)
    cwlem = CWLEM(weights)
    return ekf, cwlem

def main():
    base_path = '/kaggle/input/store-sales-time-series-forecasting/'
    train_file = base_path + 'train.csv'
    obs_files = [base_path + 'holidays_events.csv',
                 base_path + 'oil.csv',
                 base_path + 'transactions.csv']
    obs_columns = [['date', 'type'], ['date', 'dcoilwtico'], ['date', 'transactions']]

    train_data = preprocess_data(train_file, ['date', 'sales'])
    obs_data = [preprocess_data(file, cols) for file, cols in zip(obs_files, obs_columns)]

    if train_data is not None and all(obs is not None for obs in obs_data):
        ekf, cwlem = initialize_models(train_data, obs_data)

        predicted_sales = []
        true_sales = train_data['sales'].values
        for i, observation in enumerate(true_sales):
            external_factors = np.array([obs_data[j].iloc[i].values[1] for j in range(len(obs_data))])
            cwlem_prediction = cwlem.predict(external_factors)
            ekf.predict(cwlem_prediction)
            ekf.update(observation)
            predicted_sales.append(ekf.state_estimation[-1])

        rmsle = PerformanceEvaluator.calculate_rmsle(predicted_sales, true_sales)
        print("RMSLE:", rmsle)

        plt.plot(true_sales, label='True Sales')
        plt.plot(predicted_sales, label='Predicted Sales')
        plt.xlabel('Time')
        plt.ylabel('Sales')
        plt.title('True vs Predicted Sales')
        plt.legend()
        plt.show()

    else:
        print("Train data or observation data is missing or empty.")

if __name__ == "__main__":
    main()


# Particularly useful for forecasting complex systems that are time-dependent and influenced by external factors
