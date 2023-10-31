import tensorflow as tf
from keras.layers import LayerNormalization, MultiHeadAttention, Dense
import numpy as np

# encoder Layer
class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(Encoder, self).__init__()

        self.multihead_attention = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        self.feed_forward = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        attn_output = self.multihead_attention(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ff_output = self.feed_forward(out1)
        ff_output = self.dropout2(ff_output, training=training)
        out2 = self.layernorm2(out1 + ff_output)

        return out2

# Define the Transformer Encoder
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(TransformerEncoder, self).__init__()

        # input embeddings
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(maximum_position_encoding, self.d_model)

        # stacking of encoder layer
        self.encoder_layers = [Encoder(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def positional_encoding(self, position, d_model):
        angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x, training):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training)

        return x

# # Usage example with original Transformer hyperparameters
# num_layers = 6
# d_model = 512
# num_heads = 8
# dff = 2048
# input_vocab_size = 10000
# maximum_position_encoding = 10000

# model = TransformerEncoder(num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding)

# # Pass some input through the encoder
# input = tf.random.uniform((64, 40), dtype=tf.int64, minval=0, maxval=10000)
# print(input)
# output = model(input, training=False)
# print(output.shape)
