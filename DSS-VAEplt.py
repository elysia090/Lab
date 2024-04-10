import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model

# データの読み込みと前処理
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url)
data = df['Passengers'].values.astype(float)
data_mean, data_std = np.mean(data), np.std(data)
data_normalized = (data - data_mean) / data_std

# 時系列データのシーケンス生成
def generate_sequence(data, input_seq_len, output_seq_len):
    sequences_in = []
    sequences_out = []
    for i in range(len(data) - input_seq_len - output_seq_len + 1):
        sequence_in = data[i:i+input_seq_len]
        sequence_out = data[i+input_seq_len:i+input_seq_len+output_seq_len]
        sequences_in.append(sequence_in)
        sequences_out.append(sequence_out)
    return np.array(sequences_in), np.array(sequences_out)

# データセットの分割
input_seq_len = 12
output_seq_len = 1
sequences_in, sequences_out = generate_sequence(data_normalized, input_seq_len, output_seq_len)
train_size = int(len(sequences_in) * 0.7)
train_input = sequences_in[:train_size]
train_output = sequences_out[:train_size]
test_input = sequences_in[train_size:]
test_output = sequences_out[train_size:]

# DSS-VAEモデルの定義
class DSSVAE(Model):
    def __init__(self, input_dim, latent_dim, state_dim, lambda_ssm):
        super(DSSVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.lambda_ssm = lambda_ssm

        # VAEのエンコーダー
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(latent_dim + latent_dim)  # mean and log_var
        ])

        # VAEのデコーダー
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(input_dim)
        ])

        # 状態空間モデルの推定器
        self.ssm_estimator = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(latent_dim)  # 状態空間の次元数に合わせる
        ])

    def encode(self, x):
        mean, log_var = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(log_var * .5) + mean

    def decode(self, z):
        return self.decoder(z)

    def ssm_estimate(self, z):
        return self.ssm_estimator(z)

    def call(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        x_reconstructed = self.decode(z)
        state_prediction = self.ssm_estimate(z)
        return x_reconstructed, mean, log_var, state_prediction

def dssvae_loss(x, x_reconstructed, mean, log_var, state_prediction, lambda_ssm):
    reconstruction_loss = tf.reduce_mean(tf.square(x - x_reconstructed))
    kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
    state_prediction_resized = tf.keras.layers.Dense(latent_dim)(state_prediction)  # 状態予測をlatent_dimにリサイズ
    ssm_loss = lambda_ssm * tf.reduce_mean(tf.square(state_prediction_resized - mean))
    return reconstruction_loss + kl_loss + ssm_loss

# データのバッチ生成関数
def generate_batch(data, batch_size):
    indices = np.random.choice(len(data), size=batch_size, replace=False)
    return data[indices]

# モデルのインスタンス化
input_dim = input_seq_len  # 入力次元数
latent_dim = 32  # 潜在空間の次元数
state_dim = 16   # 状態空間の次元数
lambda_ssm = 0.1  # 状態空間モデルの損失重み
dss_vae = DSSVAE(input_dim, latent_dim, state_dim, lambda_ssm)

# オプティマイザの設定
optimizer = tf.keras.optimizers.Adam()

# 訓練ループ
epochs = 50
batch_size = 64

for epoch in range(epochs):
    for step in range(len(train_input) // batch_size):
        x_batch = generate_batch(train_input, batch_size)
        x_batch = x_batch.reshape(batch_size, -1)
        with tf.GradientTape() as tape:
            x_reconstructed, mean, log_var, state_prediction = dss_vae(x_batch)
            loss = dssvae_loss(x_batch, x_reconstructed, mean, log_var, state_prediction, lambda_ssm)
        gradients = tape.gradient(loss, dss_vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, dss_vae.trainable_variables))

    # エポックごとに損失を表示
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")

# テストデータでモデルの性能を評価
test_x = test_input
test_x = test_x.reshape(-1, input_seq_len)
reconstructed_x, _, _, _ = dss_vae(test_x)

