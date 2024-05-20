
import tensorflow as tf #一个抽象的框架，一个实体的执行分工明确的调包侠拼凑侠
from Encoder.Built_Transformer import Built_Transformer as utils
from Decoder.DecoderModel import DecoderModel

class Joint_Fusion_Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
        super(Joint_Fusion_Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = utils.positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers1 = [DecoderModel(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dec_layers2 = [DecoderModel(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        # self.dec_layers3 = [DecoderModel(d_model, num_heads, dff, rate)
        #                    for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.dropout1 = tf.keras.layers.Dropout(rate)
    def call(self, comm_tar, sbt_output, graph_output, training,
            sbt_padding_mask, node_padding_mask, look_ahead_mask):
        seq_len = tf.shape(comm_tar)[1]
        attention_weights = {}

        comm_tar = self.embedding(comm_tar)  # (batch_size, target_seq_len, d_model)
        comm_tar *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        comm_tar += self.pos_encoding[:, :seq_len, :]
        comm_tar = self.dropout(comm_tar, training=training)

        comm_tar1 = tf.identity(comm_tar)
        comm_tar2 = tf.identity(comm_tar)


        # sbt & comm
        for i in range(self.num_layers):
            comm_tar1, block_comm, block_sbt_comm = self.dec_layers1[i](comm_tar1, sbt_output, training,
                                                 look_ahead_mask, sbt_padding_mask)
            attention_weights['decoder_layer{}_block_comm'.format(i+1)] = block_comm
            attention_weights['decoder_layer{}_sbt_comm'.format(i+1)] = block_sbt_comm

        # graph & comm
        for i in range(self.num_layers):
            comm_tar2, block_comm, block_graph_comm = self.dec_layers2[i](comm_tar2, graph_output, training,
                                                 look_ahead_mask, node_padding_mask)
            # attention_weights['decoder_layer{}_block_comm2'.format(i+1)] = block_comm
            attention_weights['decoder_layer{}_graph_comm'.format(i+1)] = block_graph_comm

        comm_tar3 = tf.concat([comm_tar1, comm_tar2], axis=-1)


        return comm_tar3, attention_weights
