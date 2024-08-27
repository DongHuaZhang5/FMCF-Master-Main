import tensorflow as tf
from Encoder.Built_Transformer import Built_Transformer as utils
from Encoder.EncoderModel import EncoderModel
from Encoder.GAT_Keras import GAT_Keras

class GAT_GraphEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, asthop,
                 maximum_position_encoding, dropout_rate):
        super(GAT_GraphEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Embedding layer for the nodes
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, name="graph_embed")

        # Positional encoding for the sequence
        self.pos_encoding = utils.positional_encoding(maximum_position_encoding, self.d_model)

        # GAT layer for graph attention
        self.gat_layer = GAT_Keras(d_model)

        # Number of attention hops
        self.asthop = asthop

        # List of Transformer Encoder layers
        self.enc_layers = [EncoderModel(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]

        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def Execute_Attention(self, node_input, edge_input, training, mha_mask):
        # Embed the node input
        node_ebd = self.embedding(node_input)

        # Apply GAT layer for the specified number of attention hops
        for _ in range(self.asthop):
            node_ebd = self.gat_layer([node_ebd, edge_input])

        # Combine node embedding with positional encoding and apply dropout
        seq_len = tf.shape(node_ebd)[1]
        final_output = node_ebd * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        final_output += self.pos_encoding[:, :seq_len, :]
        final_output = self.dropout(x, training=training)

        # Pass the output through each Transformer Encoder layer
        mha_attn = None
        for layer in self.enc_layers:
            x, attn = layer(x, training, mha_mask)
            if mha_attn is None:
                mha_attn = attention_weights
            else:
                mha_attn += attention_weights

        return  final_output, attention_weights