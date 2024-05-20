
import tensorflow as tf
from Encoder.Built_Transformer import Built_Transformer as utils
from Encoder.EncoderModel import EncoderModel
from Encoder.GAT_Keras import GAT_Keras


class GAT_GraphEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, asthop,
               maximum_position_encoding, rate=0.1):
        super(GAT_GraphEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, name="graph_embed")
        self.pos_encoding = utils.positional_encoding(maximum_position_encoding,
                                                self.d_model)
       # self.gcn_layer = GCNLayer(d_model)
        self.gat_layer = GAT_Keras(d_model)
        self.asthop = asthop
        self.enc_layers = [EncoderModel(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, node_input, edge_input, training, mha_mask):
        node_ebd = self.embedding(node_input)

        for i in range(self.asthop):
            node_ebd = self.gat_layer([node_ebd, edge_input])
        x = node_ebd

        seq_len = tf.shape(x)[1]
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
          x, mha_attn = self.enc_layers[i](x, training, mha_mask)
        return x, mha_attn # (batch_size, 2*input_seq_len, d_model)

