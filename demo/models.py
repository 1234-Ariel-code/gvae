from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, Model

def _build_encoder(input_dim: int, latent_dim: int, depth: int = 2):
    inp = layers.Input(shape=(input_dim,))
    x = inp
    width = max(64, latent_dim * (2 ** max(depth - 1, 1)))
    for _ in range(depth):
        x = layers.Dense(width, activation="relu")(x)
        width = max(latent_dim * 2, width // 2)
    z_params = layers.Dense(latent_dim * 2)(x)
    return Model(inp, z_params, name="encoder")

def _build_decoder(input_dim: int, latent_dim: int, depth: int = 2, input_latent_dim: int | None = None):
    zdim = input_latent_dim if input_latent_dim is not None else latent_dim
    inp = layers.Input(shape=(zdim,))
    x = inp
    width = max(64, latent_dim)
    for _ in range(depth):
        x = layers.Dense(width, activation="relu")(x)
        width *= 2
    out = layers.Dense(input_dim)(x)
    return Model(inp, out, name="decoder")

class BaseVAE(Model):
    def __init__(self, input_dim: int, latent_dim: int, depth: int = 2, beta: float = 1.0):
        super().__init__()
        self.encoder = _build_encoder(input_dim, latent_dim, depth)
        self.decoder = _build_decoder(input_dim, latent_dim, depth)
        self.latent_dim = latent_dim
        self.beta = beta
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def encode(self, x):
        z_params = self.encoder(x, training=False)
        mu, log_var = tf.split(z_params, 2, axis=1)
        return mu, log_var

    def sample(self, mu, log_var):
        eps = tf.random.normal(tf.shape(mu), dtype=tf.float32)
        return mu + eps * tf.exp(0.5 * log_var)

    def call(self, x, training=False):
        mu, log_var = self.encode(x)
        z = self.sample(mu, log_var)
        xhat = self.decoder(z, training=training)
        return xhat, mu, log_var, z

    def train_step(self, data):
        x = tf.cast(data[0] if isinstance(data, (tuple, list)) else data, tf.float32)
        with tf.GradientTape() as tape:
            xhat, mu, log_var, _ = self(x, training=True)
            recon = tf.reduce_mean(tf.reduce_sum(tf.square(x - xhat), axis=1))
            kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))
            loss = recon + self.beta * kl
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

class GVAE(Model):
    def __init__(self, input_dim: int, latent_dim: int, num_samples: int = 10, depth: int = 2):
        super().__init__()
        self.encoder = _build_encoder(input_dim, latent_dim, depth)
        self.decoder = _build_decoder(input_dim, latent_dim, depth, input_latent_dim=2 * latent_dim)
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def encode(self, x):
        z_params = self.encoder(x, training=False)
        mu, log_var = tf.split(z_params, 2, axis=1)
        return mu, log_var

    def reparameterize_many(self, mu, log_var):
        eps = tf.random.normal((self.num_samples, tf.shape(mu)[0], self.latent_dim), dtype=tf.float32)
        mu_exp = tf.expand_dims(mu, 0)
        lv_exp = tf.expand_dims(log_var, 0)
        return mu_exp + eps * tf.exp(0.5 * lv_exp)

    def quantile_features(self, zstack):
        zsorted = tf.sort(zstack, axis=0)
        k = tf.shape(zsorted)[0]
        q25 = zsorted[tf.cast(0.25 * tf.cast(k, tf.float32), tf.int32)]
        q75 = zsorted[tf.cast(0.75 * tf.cast(k, tf.float32), tf.int32)]
        return tf.concat([q25, q75], axis=1)

    def call(self, x, training=False):
        mu, log_var = self.encode(x)
        zstack = self.reparameterize_many(mu, log_var)
        zfinal = self.quantile_features(zstack)
        xhat = self.decoder(zfinal, training=training)
        return xhat, mu, log_var, zfinal

    def train_step(self, data):
        x = tf.cast(data[0] if isinstance(data, (tuple, list)) else data, tf.float32)
        with tf.GradientTape() as tape:
            xhat, mu, log_var, _ = self(x, training=True)
            recon = tf.reduce_mean(tf.reduce_sum(tf.square(x - xhat), axis=1))
            kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))
            loss = recon + kl
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

def build_baseline_vae(input_dim: int, latent_dim: int, depth: int = 2):
    return BaseVAE(input_dim=input_dim, latent_dim=latent_dim, depth=depth, beta=1.0)

def build_beta_vae(input_dim: int, latent_dim: int, depth: int = 2, beta: float = 4.0):
    return BaseVAE(input_dim=input_dim, latent_dim=latent_dim, depth=depth, beta=beta)

def build_gvae(input_dim: int, latent_dim: int, depth: int = 2, num_samples: int = 10):
    return GVAE(input_dim=input_dim, latent_dim=latent_dim, depth=depth, num_samples=num_samples)
