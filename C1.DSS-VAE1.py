import tensorflow as tf
import numpy as np

class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(latent_dim + latent_dim) # mean and log_var
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(input_dim)
        ])

    def encode(self, x):
        mean, log_var = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(log_var * .5) + mean

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mean, log_var

def vae_loss(x, x_reconstructed, mean, log_var):
    reconstruction_loss = tf.reduce_mean(tf.square(x - x_reconstructed))
    kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
    return reconstruction_loss + kl_loss

class StateSpaceModel(tf.keras.Model):
    def __init__(self, latent_dim, state_dim):
        super(StateSpaceModel, self).__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.transition = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(state_dim)
        ])
        self.observation = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(state_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(input_dim)
        ])

    def call(self, z):
        state = self.transition(z)
        observation = self.observation(state)
        return state, observation

def train_step(x, vae, ssm, vae_optimizer, ssm_optimizer):
    with tf.GradientTape() as tape:
        x_reconstructed, mean, log_var = vae(x)
        reconstruction_loss = vae_loss(x, x_reconstructed, mean, log_var)
    gradients = tape.gradient(reconstruction_loss, vae.trainable_variables)
    vae_optimizer.apply_gradients(zip(gradients, vae.trainable_variables))

    with tf.GradientTape() as tape:
        _, _, log_var = vae(x)
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
    gradients = tape.gradient(kl_loss, vae.trainable_variables)
    vae_optimizer.apply_gradients(zip(gradients, vae.trainable_variables))

    with tf.GradientTape() as tape:
        _, _, log_var = vae(x)
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
        state, observation = ssm(mean)
        observation_loss = tf.reduce_mean(tf.square(x - observation))
        total_loss = reconstruction_loss + kl_loss + observation_loss
    gradients = tape.gradient(total_loss, vae.trainable_variables + ssm.trainable_variables)
    vae_optimizer.apply_gradients(zip(gradients[:len(vae.trainable_variables)], vae.trainable_variables))
    ssm_optimizer.apply_gradients(zip(gradients[len(vae.trainable_variables):], ssm.trainable_variables))
