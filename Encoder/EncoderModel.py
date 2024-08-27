import tensorflow as tf
from Encoder.Multi_Head_Attention import Multi_Head_Attention
from Encoder.Built_Transformer import Built_Transformer as utils

class EncoderModel(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderModel, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        # Multi-head attention layer
        self.multi_head_attention = Multi_Head_Attention(d_model, num_heads)

        # Point-wise feed-forward network
        self.ffn = utils.point_wise_feed_forward_network(d_model, dff)

        # Layer normalization layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout layers
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def Execute_Attention(self, inputs, training=None, mask=None):
        # Apply multi-head attention and retrieve the output and attention weights
        mha_output, attention_weights = self.multi_head_attention(inputs, inputs, inputs, mask)
        mha_output = self.dropout1(mha_output, training=training)

        # Apply layer normalization and residual connection
        norm1_output = self.layernorm1(inputs + mha_output)

        # Apply point-wise feed-forward network
        ffn_output = self.ffn(norm1_output)
        ffn_output = self.dropout2(ffn_output, training=training)

        # Apply final layer normalization and residual connection
        final_output = self.layernorm2(norm1_output + ffn_output)

        return final_output, attention_weights