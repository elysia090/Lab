import numpy as np
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

    def predict(self, state_estimate, error_covariance, control_input):
        # 予測ステップ
        predicted_state_estimate = np.dot(self.A, state_estimate) + np.dot(self.B, control_input)
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
def run_kalman_filter(A, B, H, Q, R, initial_state_estimate, initial_error_covariance, control_inputs, measurements):
    kf = KalmanFilter(A, B, H, Q, R)
    state_estimates = [initial_state_estimate]
    error_covariances = [initial_error_covariance]

    for control_input, measurement in zip(control_inputs, measurements):
        predicted_state_estimate, predicted_error_covariance = kf.predict(state_estimates[-1], error_covariances[-1], control_input)
        updated_state_estimate, updated_error_covariance = kf.update(predicted_state_estimate, predicted_error_covariance, measurement)
        state_estimates.append(updated_state_estimate)
        error_covariances.append(updated_error_covariance)

    return state_estimates[1:], error_covariances[1:]

# 人工的なデータの生成
def generate_data(num_steps):
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0], [1]])
    H = np.array([[1, 0]])
    Q = np.array([[0.1, 0], [0, 0.1]])
    R = np.array([[1]])
    initial_state_estimate = np.array([[0], [0]])
    initial_error_covariance = np.eye(2)

    control_inputs = [np.array([[i]]) for i in range(num_steps)]
    true_states = [np.dot(A, initial_state_estimate)]
    measurements = [np.dot(H, true_states[-1]) + np.random.normal(0, np.sqrt(R[0, 0]))]

    for control_input in control_inputs[1:]:
        true_state = np.dot(A, true_states[-1]) + np.dot(B, control_input)
        true_states.append(true_state)
        measurement = np.dot(H, true_state) + np.random.normal(0, np.sqrt(R[0, 0]))
        measurements.append(measurement)

    return A, B, H, Q, R, initial_state_estimate, initial_error_covariance, control_inputs, measurements

# データの生成
num_steps = 100
data = generate_data(num_steps)

# カルマンフィルタを実行
state_estimates, _ = run_kalman_filter(*data)

# 状態推定値と真の状態の比較
true_states = [state for state in data[8]]
plt.figure(figsize=(10, 6))

# カルマンフィルタによる推定結果のプロット
plt.subplot(2, 1, 1)
plt.plot([state[0, 0] for state in state_estimates], label='Kalman Filter Estimate')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('Kalman Filter Estimate')
plt.legend()
plt.grid(True)

# 実際の状態のプロット
plt.subplot(2, 1, 2)
plt.plot([state[0, 0] for state in true_states], label='True State')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('True State')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# カルマンフィルタを用いて予測ホライゾンを進める
def predict_horizon(A, B, initial_state_estimate, control_inputs, num_steps):
    state_estimates = [initial_state_estimate]
    state_estimate = initial_state_estimate

    for i in range(num_steps):
        # 予測ステップの実行
        state_estimate = np.dot(A, state_estimate) + np.dot(B, control_inputs[i])
        state_estimates.append(state_estimate)

    return state_estimates[1:]

# 予測ホライゾンの時間ステップ数
num_prediction_steps = 10

# 予測ホライゾンを進める
predicted_states = predict_horizon(data[0], data[1], state_estimates[-1], data[8], num_prediction_steps)

# 予測結果のプロット
plt.figure(figsize=(10, 6))
plt.plot([state[0, 0] for state in state_estimates], label='Kalman Filter Estimate')
plt.plot(range(len(state_estimates), len(state_estimates) + len(predicted_states)), [state[0, 0] for state in predicted_states], label='Predicted State')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('Comparison of Kalman Filter Estimate and Predicted State')
plt.legend()
plt.grid(True)
plt.show()
