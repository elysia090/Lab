
import numpy as np
import pandas as pd

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

class ExtendedKalmanFilter:
    def __init__(self, state_dim, process_noise_cov):
        self.state_dim = state_dim
        self.process_noise_cov = process_noise_cov
        self.state_estimation = np.zeros(state_dim)
        self.state_covariance = np.eye(state_dim)

    def initialize(self, initial_state, initial_state_cov):
        self.state_estimation = initial_state.astype(float)
        self.state_covariance = initial_state_cov.astype(float)

    def predict(self, external_factors):
        # 外部要因を用いて状態の予測を行う
        predicted_state = self.state_estimation + external_factors  # 仮の予測
        predicted_covariance = self.state_covariance + self.process_noise_cov  # 仮の共分散

        return predicted_state, predicted_covariance

    def update(self, observation):
        # カルマンゲインを計算
        kalman_gain = np.dot(self.state_covariance, np.linalg.inv(self.state_covariance + self.process_noise_cov))

        # 観測と予測の誤差を計算
        observation_error = observation - self.state_estimation

        # 状態の更新
        self.state_estimation += np.dot(kalman_gain, observation_error)

        # 共分散の更新
        self.state_covariance = np.dot((np.eye(self.state_dim) - kalman_gain), self.state_covariance)

def integrate_cwlem_ekf(train_data, obs_data, cwlem, ekf, obs_columns):
    for i, observation in enumerate(train_data['sales']):
        # CWLEMによる予測
        external_factors = []  # 外部要因の抽出
        for obs_df, cols in zip(obs_data, obs_columns):
            obs_row = obs_df.loc[obs_df['date'] == train_data.iloc[i]['date']]
            if not obs_row.empty:
                external_factors.extend(obs_row[cols[1:]].values.flatten())
            else:
                external_factors.extend([np.nan] * (len(cols) - 1))
        cwlem_prediction = cwlem.predict(np.array(external_factors)[:len(cwlem.beta)])

        # EKFによる予測と更新
        predicted_state, predicted_covariance = ekf.predict(cwlem_prediction)
        ekf.update(observation)

    # 最終的な予測を取得
    final_prediction = ekf.state_estimation[-1]

    return final_prediction

def evaluate_prediction(predictions, true_values):
    rmsle = np.sqrt(np.mean(np.square(np.log(predictions + 1) - np.log(true_values + 1))))
    return rmsle

def main():
    base_path = '/kaggle/input/store-sales-time-series-forecasting/'
    train_file = 'train.csv'
    obs_files = ['holidays_events.csv', 'oil.csv', 'transactions.csv']
    obs_columns = [['date'], ['date', 'dcoilwtico'], ['date', 'transactions']]

    # トレーニングデータの読み込み
    train_data = DataLoader.load_data(base_path + train_file, ['date', 'sales'])

    # 観測データの読み込み
    obs_data = []
    for file, cols in zip(obs_files, obs_columns):
        df = DataLoader.load_data(base_path + file, cols)
        obs_data.append(df)

    if train_data is not None and all(obs is not None for obs in obs_data):
        try:
            # CWLEMとEKFの初期化
            beta_length = len(obs_columns[0]) - 1  # CWLEMのベータの長さ
            state_dim = len(train_data.columns) + sum(len(cols) - 1 for cols in obs_columns)  # 状態の次元
            process_noise_cov = np.eye(state_dim)  # プロセスノイズの共分散行列
            cwlem = CWLEM(beta_length)
            ekf = ExtendedKalmanFilter(state_dim, process_noise_cov)

            # EKFの初期化
            initial_state = np.zeros(state_dim)  # 初期状態ベクトル
            initial_state_cov = np.eye(state_dim) * 1.0  # 初期共分散行列
            ekf.initialize(initial_state, initial_state_cov)

            # CWLEMとEKFの統合
            final_prediction = integrate_cwlem_ekf(train_data, obs_data, cwlem, ekf, obs_columns)

            if final_prediction is not None:
                # 予測の評価
                true_sales = train_data['sales'].values
                rmsle = evaluate_prediction(final_prediction, true_sales)
                print("RMSLE:", rmsle)
            else:
                print("Integration failed.")

        except Exception as e:
            print("An error occurred:", e)
    else:
        print("Train data or observation data is missing or empty.")

if __name__ == "__main__":
    main()






