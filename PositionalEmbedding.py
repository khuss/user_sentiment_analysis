import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

class PositionalEmbedding(layers.Layer):

  def __init__(self, sequence_length, input_dim, output_dim, **kwarg):
    super().__init__(**kwarg)
    self.sequence_length = sequence_length
    self.input_dim = input_dim
    self.output_dim = output_dim

    self.token_embeddings = layers.Embedding(
        input_dim = input_dim, output_dim = output_dim
        )
    self.position_embeddings = layers.Embedding(
        input_dim=sequence_length, output_dim = output_dim
        )

  def call(self, inputs):
    embedded_tokens = self.token_embeddings(inputs)

    length = tf.shape(inputs)[-1]
    positions = tf.range(start = 0, limit = length, delta = 1)
    embedded_positions = self.position_embeddings(positions)
    return embedded_tokens + embedded_positions