import tensorflow as tf
from keras.layers import LayerNormalization, MultiHeadAttention, Dense, Dropout, Embedding
import numpy as np

import time

# encoder Layer
class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(Encoder, self).__init__()

        self.multihead_attention = MultiHeadAttention(key_dim=d_model//num_heads, num_heads=num_heads)
        self.feed_forward = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.feed_forward.build((None, d_model))

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, attention_mask):
        # start_time = time.time()

        attn_output = self.multihead_attention(x, x, x, attention_mask=attention_mask)

        # print('mha:', time.time() - start_time)
        # start_time = time.time()

        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # print('add & norm:', time.time() - start_time)
        # start_time = time.time()

        ff_output = self.feed_forward(out1)

        # print('ffn:', time.time() - start_time)
        # start_time = time.time()


        ff_output = self.dropout2(ff_output, training=training)
        out2 = self.layernorm2(out1 + ff_output)

        # print('add & norm:', time.time() - start_time)
        # start_time = time.time()

        return out2

# transformer encoder
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(TransformerEncoder, self).__init__()

        # input embeddings
        self.d_model = d_model
        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(maximum_position_encoding, self.d_model)

        # stacking of encoder layer
        self.encoder_layers = [Encoder(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = Dropout(rate)

    def positional_encoding(self, position, d_model):
        angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x, training, attention_mask):
        # start_time = time.time()

        x = self.embedding(x)

        # print('word_embedding:', time.time() - start_time)
        # start_time = time.time()

        # In Vaswani's paper they applied this
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # print('sqrt:', time.time() - start_time)
        # start_time = time.time()

        x += self.pos_encoding

        # print('pos_encoding:', time.time() - start_time)
        # start_time = time.time()

        x = self.dropout(x, training=training)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training, attention_mask)

        return x

# transformer encoder with MLM head
class MLMTransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(MLMTransformerEncoder, self).__init__()

        self.transformer_encoder = TransformerEncoder(num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding)

        self.mlm_head = Dense(input_vocab_size, activation='softmax')
        self.mlm_head.build(input_shape=(None, d_model))

        self.dropout = Dropout(rate)

    def call(self, inputs, training):
        x, attention_mask = inputs

        x = self.transformer_encoder(x, training, attention_mask)

        # start_time = time.time()

        x = self.mlm_head(x)

        # print('mlm_head:', time.time() - start_time)
        # start_time = time.time()

        x = self.dropout(x, training=training)
        
        return x

    def remove_mlm_head(self):
        return self.transformer_encoder