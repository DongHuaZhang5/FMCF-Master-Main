import tensorflow as tf
from Encoder.Multi_Head_Attention import Multi_Head_Attention
from Encoder.Built_Transformer import Built_Transformer as utils
class EncoderModel(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderModel, self).__init__()

        self.mha = Multi_Head_Attention(d_model, num_heads)
        self.ffn = utils.point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    def call(self, mha_x, training, mask):

        mha_output, mha_attn = self.mha(mha_x, mha_x, mha_x, mask)  # (batch_size, input_seq_len, d_model)
        mha_output = self.dropout1(mha_output, training=training)
        mha_output = self.layernorm1(mha_x + mha_output)  # (batch_size, input_seq_len, d_model)
        ffn_mha_output = self.ffn(mha_output)  # (batch_size, input_seq_len, d_model)
        ffn_mha_output = self.dropout2(ffn_mha_output, training=training)
        out_mha = self.layernorm2(mha_output + ffn_mha_output)  # (batch_size, input_seq_len, d_model)

        return out_mha, mha_attn
