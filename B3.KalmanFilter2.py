import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# カルマンフィルタの定義
class KalmanFilter:
    def __init__(self, A, B, H, Q, R):
        self.A = A  # 状態遷移行列
        self.B = B  # 入力行列
        self.H = H  # 観測行列
        self.Q = Q  # プロセスノイズの共分散行列
        self.R = R  # 観測ノイズの共分散行列
        self.state_estimate = None
        self.error_covariance = None

    def predict(self, state_estimate, error_covariance):
        # 予測ステップ
        predicted_state_estimate = np.dot(self.A, state_estimate)
        predicted_error_covariance = np.dot(np.dot(self.A, error_covariance), self.A.T) + self.Q
        return predicted_state_estimate, predicted_error_covariance

    def update(self, predicted_state_estimate, predicted_error_covariance, measurement):
        # 更新ステップ
        innovation = measurement - np.dot(self.H, predicted_state_estimate)
        innovation_covariance = np.dot(np.dot(self.H, predicted_error_covariance), self.H.T) + self.R
        kalman_gain = np.dot(np.dot(predicted_error_covariance, self.H.T), np.linalg.inv(innovation_covariance))
        updated_state_estimate = predicted_state_estimate + np.dot(kalman_gain, innovation)
        updated_error_covariance = np.dot((np.eye(self.A.shape[0]) - np.dot(kalman_gain, self.H)), predicted_error_covariance)
        return updated_state_estimate, updated_error_covariance

# カルマンフィルタを用いて状態推定を実行
def run_kalman_filter(A, B, H, Q, R, initial_state_estimate, initial_error_covariance, measurements):
    kf = KalmanFilter(A, B, H, Q, R)
    state_estimates = [initial_state_estimate]
    error_covariances = [initial_error_covariance]

    for measurement in measurements:
        predicted_state_estimate, predicted_error_covariance = kf.predict(state_estimates[-1], error_covariances[-1])
        updated_state_estimate, updated_error_covariance = kf.update(predicted_state_estimate, predicted_error_covariance, measurement)
        state_estimates.append(updated_state_estimate)
        error_covariances.append(updated_error_covariance)

    return state_estimates[1:], error_covariances[1:]

# カルマンフィルタを用いて予測ホライゾンを進める
def predict_horizon(A, B, initial_state_estimate, num_steps):
    state_estimates = [initial_state_estimate]
    state_estimate = initial_state_estimate

    for i in range(num_steps):
        # 予測ステップの実行
        state_estimate = np.dot(A, state_estimate) + B
        state_estimates.append(state_estimate)

    return state_estimates[1:]

# データの読み込みや前処理
data = pd.read_csv("C:\\Users\\renta\\testcode\\sample.CSV", skiprows=160, encoding='shift_jis')

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

# 実際のデータを使用してカルマンフィルタを実行
# 各パラメータは適切な値に置き換える必要があります
A = np.eye(2)  # 状態遷移行列
B = np.zeros((2, 1))  # 入力行列
H = np.eye(2)  # 観測行列
Q = np.eye(2)  # プロセスノイズの共分散行列
R = np.eye(2)  # 観測ノイズの共分散行列
initial_state_estimate = np.zeros((2, 1))  # 初期状態推定値
initial_error_covariance = np.eye(2)  # 初期誤差共分散行列

measurements = data[['spot_month_end', 'spot_monthly_average']].values

state_estimates, _ = run_kalman_filter(A, B, H, Q, R, initial_state_estimate, initial_error_covariance, measurements)

# 予測ホライゾンの時間ステップ数
num_prediction_steps = 500

# 予測ホライゾンを進める
predicted_states = predict_horizon(A, B, state_estimates[-1], num_prediction_steps)

def calculate_mse(state_estimates, measurements):
    mse_sum = 0
    for i in range(len(measurements)):
        mse_sum += np.mean(np.square(measurements[i] - state_estimates[i]))
    mse = mse_sum / len(measurements)
    return mse

# MSEを計算
mse = calculate_mse(state_estimates, measurements)
print("MSE:", mse)


# 結果の可視化
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# 全体のグラフ
ax1.plot(data.index, [state[0, 0] for state in state_estimates], label='Kalman Filter Estimate')
ax1.plot(data.index, data['spot_monthly_average'], label='Spot Monthly Average')
ax1.set_xlabel('Date')
ax1.set_ylabel('Spot Price')
ax1.set_title('Kalman Filter Estimate and Spot Monthly Average')
ax1.legend()
ax1.grid(True)

# 拡大図
ax2.plot(data.index, [state[0, 0] for state in state_estimates], label='Kalman Filter Estimate')
ax2.plot(data.index[-1] + pd.to_timedelta(np.arange(1, num_prediction_steps + 1), unit='D'), [state[0, 0] for state in predicted_states], label='Predicted State', color='red')
ax2.set_xlabel('Date')
ax2.set_ylabel('Spot Price')
ax2.set_title('Kalman Filter Estimate and Predicted State (Zoomed In)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
