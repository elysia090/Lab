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

    @staticmethod
    def update_elysia(X, y):
        try:
            # エリシアメゾットの更新
            return np.linalg.pinv(X), y
        except Exception as e:
            print(f"Elysia method update error: {e}")
            return None, None

class StateSpaceModel:
    def __init__(self, ar_params, ma_params):
        self.ar_params = ar_params
        self.ma_params = ma_params

    def state_transition_function(self, state):
        return np.dot(self.ar_params, state)

class ExtendedKalmanFilterWithElysia:
    def __init__(self, state_space_model, process_noise_cov):
        self.state_space_model = state_space_model
        self.process_noise_cov = process_noise_cov
        self.state_estimation = None
        self.state_covariance = None

    def initialize(self, initial_state, initial_state_cov):
        self.state_estimation = initial_state
        self.state_covariance = initial_state_cov

    def predict(self, X, external_factors):
        try:
            predicted_state = self.state_space_model.state_transition_function(self.state_estimation)
            # エリシアメゾットを使用して外部要因を予測
            X_inv, y = ModelUpdater.update_elysia(X, external_factors)
            if X_inv is not None and y is not None:
                beta = np.dot(X_inv, y)
                self.state_estimation = predicted_state + np.dot(X, beta)
        except Exception as e:
            print(f"Prediction error: {e}")

    def update(self, observation):
        try:
            state_transition_jacobian = np.eye(len(self.state_estimation))
            state_transition_jacobian[-1, -1] = 1
            kalman_gain = np.dot(np.dot(self.state_covariance, state_transition_jacobian.T),
                                 np.linalg.inv(np.dot(np.dot(state_transition_jacobian, self.state_covariance), state_transition_jacobian.T) + self.process_noise_cov))
            predicted_state = self.state_space_model.state_transition_function(self.state_estimation)
            self.state_estimation += np.dot(kalman_gain, (observation - predicted_state))
            self.state_covariance = np.dot((np.eye(len(self.state_covariance)) - np.dot(kalman_gain, state_transition_jacobian)), self.state_covariance)
        except Exception as e:
            print(f"Update error: {e}")

class CWLEM:
    def __init__(self, beta):
        self.beta = beta

    def predict(self, external_factors):
        return np.dot(external_factors, self.beta)

class PerformanceEvaluator:
    @staticmethod
    def calculate_rmsle(predictions, true_values):
        return np.sqrt(np.mean(np.square(np.log(predictions + 1) - np.log(true_values + 1))))

def calculate_feature_importance(obs_data):
    try:
        # 外部要因同士の相関行列を計算
        correlation_matrix = np.corrcoef([obs_data[i].iloc[:, 1] for i in range(len(obs_data))])
        # 各外部要因の重要度を計算（相関係数の絶対値の平均）
        feature_importance = np.mean(np.abs(correlation_matrix), axis=1)
        return feature_importance
    except Exception as e:
        print(f"Feature importance calculation error: {e}")
        return None

def auto_adjust_weights(obs_data):
    try:
        # 外部要因の重要度を計算
        feature_importance = calculate_feature_importance(obs_data)
        if feature_importance is not None:
            # 重要度を基に、各外部要因に対する重みを決定
            total_importance = np.sum(feature_importance)
            weights = feature_importance / total_importance
            return weights
        else:
            return None
    except Exception as e:
        print(f"Weight adjustment error: {e}")
        return None

def initialize_models(train_data, obs_data):
    try:
        state_space_model = StateSpaceModel([], [])  # ARIMA パラメータ不要
        initial_state = np.zeros(len(train_data.columns) + len(obs_data))
        initial_state_cov = np.eye(len(initial_state))
        process_noise_cov = np.eye(len(initial_state))
        ekf = ExtendedKalmanFilterWithElysia(state_space_model, process_noise_cov)
        ekf.initialize(initial_state, initial_state_cov)
        # 自動で重み付けを調整
        weights = auto_adjust_weights(obs_data)
        if weights is not None:
            cwlem = CWLEM(weights)
            return ekf, cwlem
        else:
            return None, None
    except Exception as e:
        print(f"Model initialization error: {e}")
        return None, None

def main():
    try:
        base_path = '/kaggle/input/store-sales-time-series-forecasting/'
        train_file = base_path + 'train.csv'
        obs_files = [base_path + 'holidays_events.csv',
                     base_path + 'oil.csv',
                     base_path + 'transactions.csv']
        obs_columns = [['date', 'type'], ['date', 'dcoilwtico'], ['date', 'transactions']]

        train_data = ModelUpdater.preprocess_data(train_file, ['date', 'sales'])
        obs_data = [ModelUpdater.preprocess_data(file, cols) for file, cols in zip(obs_files, obs_columns)]

        if train_data is not None and all(obs is not None for obs in obs_data):
            ekf, cwlem = initialize_models(train_data, obs_data)
            if ekf is not None and cwlem is not None:
                predicted_sales = []
                true_sales = train_data['sales'].values
                for i, observation in enumerate(true_sales):
                    external_factors = np.array([obs_data[j].iloc[i].values[1] for j in range(len(obs_data))])
                    cwlem_prediction = cwlem.predict(external_factors)
                    ekf.predict(external_factors)
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
                print("Model initialization failed.")
        else:
            print("Train data or observation data is missing or empty.")
    except Exception as e:
        print(f"Main error: {e}")

if __name__ == "__main__":
    main()
