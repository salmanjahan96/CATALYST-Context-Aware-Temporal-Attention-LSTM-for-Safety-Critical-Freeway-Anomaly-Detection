"""
attention_layer.py
Bahdanau-style Attention Mechanism for CATALYST model.
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense


class BahdanauAttention(tf.keras.layers.Layer):
    """
    Bahdanau-style Attention Layer.

    Allows the model to learn which timesteps are most relevant
    for anomaly detection — e.g., the moments immediately before
    a crash event carry more signal than normal traffic periods.

    Args:
        units (int): Dimension of the attention weight space.

    Input:
        inputs: Tensor of shape (batch, timesteps, features)

    Output:
        context_vector:    Tensor of shape (batch, features)
        attention_weights: Tensor of shape (batch, timesteps, 1)
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = Dense(self.units)
        self.V = Dense(1)
        super().build(input_shape)

    def call(self, inputs):
        # Score: (batch, timesteps, units)
        score = tf.nn.tanh(self.W(inputs))

        # Attention weights: (batch, timesteps, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # Weighted sum: (batch, features)
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)

        return context_vector, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config
