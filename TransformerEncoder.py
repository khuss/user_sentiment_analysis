from tensorflow import keras
from tensorflow.keras import layers, models

class TransformerEncoder(layers.Layer):

  def __init__(self, embed_dim, dense_dim, num_heads, **kwarg):
    super().__init__(**kwarg)
    self.embed_dim = embed_dim
    self.dense_dim = dense_dim
    self.num_heads = num_heads

    self.attention = layers.MultiHeadAttention(num_heads = num_heads, key_dim = embed_dim)
    self.dense_proj = keras.Sequential([
                         layers.Dense(dense_dim, activation = "relu"),
                         layers.Dense(embed_dim, activation = "linear")
                         ])
    self.layernorm_1 = layers.LayerNormalization()
    self.layernorm_2 = layers.LayerNormalization()


  def call(self, inputs):
    attention_output = self.attention(inputs, inputs)
    proj_input = self.layernorm_1(inputs+attention_output)
    proj_output = self.dense_proj(proj_input)
    return self.layernorm_2(proj_input+ proj_output)