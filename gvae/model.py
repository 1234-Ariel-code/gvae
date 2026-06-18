#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single source of truth for the gVAE model architecture.

The training pipeline and SNP-prioritization/XAI pipeline both import the model
classes from this file. This avoids duplicate architecture definitions and keeps
the methodology and biological validation workflows tied to the same model.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model


def compute_initial_neurons(num_layers: int, latent_dim: int) -> int:
    """
    Encoder width schedule.

    The encoder starts wide enough so that progressive halving ends at
    2 * latent_dim, matching the concatenated mu/log_var output dimension.
    """
    return int((2 * latent_dim) * (2 ** (num_layers - 1)))


def build_encoder(original_dim: int, latent_dim: int, num_layers: int) -> tf.keras.Sequential:
    """Build encoder ending in [mu, log_var] with dimension 2 * latent_dim."""
    initial_neurons = compute_initial_neurons(num_layers, latent_dim)
    layers_list = [layers.InputLayer(input_shape=(original_dim,))]

    neurons = initial_neurons
    for _ in range(num_layers):
        layers_list.append(layers.Dense(neurons, activation="relu"))
        neurons = max(neurons // 2, 2 * latent_dim)

    layers_list.append(layers.Dense(latent_dim * 2, name="z_params"))
    return tf.keras.Sequential(layers_list, name="encoder")


def build_gvae_decoder(latent_dim: int, num_layers: int, original_dim: int) -> tf.keras.Sequential:
    """
    Build gVAE decoder.

    The gVAE representation concatenates q25 and q75 posterior quantiles, so the
    decoder input dimension is 2 * latent_dim.
    """
    layers_list = [layers.InputLayer(input_shape=(latent_dim * 2,))]

    neurons = 2 * latent_dim
    for _ in range(num_layers):
        layers_list.append(layers.Dense(neurons, activation="relu"))
        neurons *= 2

    layers_list.append(layers.Dense(original_dim, name="x_hat"))
    return tf.keras.Sequential(layers_list, name="gvae_decoder")


def build_baseline_decoder(latent_dim: int, num_layers: int, original_dim: int) -> tf.keras.Sequential:
    """Build baseline VAE decoder receiving one latent vector of size latent_dim."""
    layers_list = [layers.InputLayer(input_shape=(latent_dim,))]

    neurons = latent_dim
    for _ in range(num_layers):
        layers_list.append(layers.Dense(neurons, activation="relu"))
        neurons *= 2

    layers_list.append(layers.Dense(original_dim, name="x_hat"))
    return tf.keras.Sequential(layers_list, name="baseline_decoder")


def reconstruction_mse_loss(x, x_hat):
    """Standard mean-squared reconstruction loss."""
    x = tf.cast(x, tf.float32)
    x_hat = tf.cast(x_hat, tf.float32)
    return tf.reduce_mean(tf.square(x - x_hat))


def kl_divergence(mu, log_var):
    """KL divergence between q(z|x)=N(mu,sigma²) and the standard normal prior."""
    mu = tf.cast(mu, tf.float32)
    log_var = tf.cast(log_var, tf.float32)
    kl = -0.5 * tf.reduce_sum(
        1.0 + log_var - tf.square(mu) - tf.exp(log_var),
        axis=-1,
    )
    return tf.reduce_mean(kl)


class GVAE(Model):
    """
    Quantile-gated genomic VAE.

    The encoder learns a Gaussian posterior q(z|x). During the forward pass,
    the posterior is sampled K times and summarized by q25 and q75 per latent
    variable. The decoder receives the concatenated [q25, q75] representation.
    """

    def __init__(
        self,
        original_dim: int,
        latent_dim: int,
        num_samples: int = 10,
        num_layers: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.original_dim = int(original_dim)
        self.latent_dim = int(latent_dim)
        self.num_samples = int(num_samples)

        self.encoder = build_encoder(original_dim, latent_dim, num_layers)
        self.decoder = build_gvae_decoder(latent_dim, num_layers, original_dim)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def encode(self, x):
        z_params = self.encoder(x)
        mu, log_var = tf.split(z_params, num_or_size_splits=2, axis=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        mu = tf.cast(mu, tf.float32)
        log_var = tf.cast(log_var, tf.float32)
        batch_size = tf.shape(mu)[0]

        eps = tf.random.normal(
            shape=(self.num_samples, batch_size, self.latent_dim),
            dtype=tf.float32,
        )
        return tf.expand_dims(mu, 0) + eps * tf.exp(0.5 * tf.expand_dims(log_var, 0))

    def compute_row_wise_quantiles(self, z_samples):
        """Return concatenated q25/q75 latent representation with shape (B, 2*LD)."""
        z_sorted = tf.sort(z_samples, axis=0)
        n = tf.shape(z_sorted)[0]

        idx_25 = tf.cast(0.25 * tf.cast(n, tf.float32), tf.int32)
        idx_75 = tf.cast(0.75 * tf.cast(n, tf.float32), tf.int32)

        q25 = tf.gather(z_sorted, idx_25, axis=0)
        q75 = tf.gather(z_sorted, idx_75, axis=0)

        return tf.cast(tf.concat([q25, q75], axis=-1), tf.float32)

    def decode(self, z_quantiles):
        return self.decoder(z_quantiles)

    def call(self, inputs, training=False):
        mu, log_var = self.encode(inputs)
        z_samples = self.reparameterize(mu, log_var)
        z_quantiles = self.compute_row_wise_quantiles(z_samples)
        x_hat = self.decode(z_quantiles)

        if training:
            return x_hat, mu, log_var, z_quantiles
        return x_hat, z_quantiles

    def train_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        x = tf.cast(x, tf.float32)

        with tf.GradientTape() as tape:
            x_hat, mu, log_var, _ = self(x, training=True)
            recon_loss = reconstruction_mse_loss(x, x_hat)
            kl_loss = kl_divergence(mu, log_var)
            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        x = tf.cast(x, tf.float32)

        x_hat, mu, log_var, _ = self(x, training=True)
        recon_loss = reconstruction_mse_loss(x, x_hat)
        kl_loss = kl_divergence(mu, log_var)
        total_loss = recon_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class BaselineVAE(Model):
    """Baseline VAE using a single sampled latent vector."""

    def __init__(self, original_dim: int, latent_dim: int, num_layers: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.original_dim = int(original_dim)
        self.latent_dim = int(latent_dim)

        self.encoder = build_encoder(original_dim, latent_dim, num_layers)
        self.decoder = build_baseline_decoder(latent_dim, num_layers, original_dim)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def encode(self, x):
        z_params = self.encoder(x)
        mu, log_var = tf.split(z_params, 2, axis=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        eps = tf.random.normal(tf.shape(mu), dtype=tf.float32)
        return tf.cast(mu, tf.float32) + eps * tf.exp(0.5 * tf.cast(log_var, tf.float32))

    def decode(self, z):
        return self.decoder(z)

    def call(self, inputs, training=False):
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        if training:
            return x_hat, mu, log_var
        return x_hat

    def train_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        x = tf.cast(x, tf.float32)

        with tf.GradientTape() as tape:
            x_hat, mu, log_var = self(x, training=True)
            recon_loss = reconstruction_mse_loss(x, x_hat)
            kl_loss = kl_divergence(mu, log_var)
            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        x = tf.cast(x, tf.float32)

        x_hat, mu, log_var = self(x, training=True)
        recon_loss = reconstruction_mse_loss(x, x_hat)
        kl_loss = kl_divergence(mu, log_var)
        total_loss = recon_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class BetaVAE(BaselineVAE):
    """Beta-VAE baseline with beta-weighted KL term."""

    def __init__(
        self,
        original_dim: int,
        latent_dim: int,
        beta: float = 4.0,
        num_layers: int = 1,
        **kwargs,
    ):
        super().__init__(original_dim, latent_dim, num_layers=num_layers, **kwargs)
        self.beta = float(beta)

    def train_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        x = tf.cast(x, tf.float32)

        with tf.GradientTape() as tape:
            x_hat, mu, log_var = self(x, training=True)
            recon_loss = reconstruction_mse_loss(x, x_hat)
            kl_loss = kl_divergence(mu, log_var)
            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        x = tf.cast(x, tf.float32)

        x_hat, mu, log_var = self(x, training=True)
        recon_loss = reconstruction_mse_loss(x, x_hat)
        kl_loss = kl_divergence(mu, log_var)
        total_loss = recon_loss + self.beta * kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
