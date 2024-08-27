import tensorflow as tf
from Encoder.CodeBERT_Encoder import CodeBERT_Encoder
from Decoder.Joint_Fusion_Decoder import Joint_Fusion_Decoder
from Encoder.GAT_GraphEncoder import GAT_GraphEncoder

class FMCF(tf.keras.Model):
    def __init__(self, layers, model_dimension, attention_heads, feed_forward_dim, source_vocab_size, graph_vocab_size, 
                 attention_hops, target_vocab_size, source_position_encoding, graph_position_encoding, target_position_encoding, dropout_rate):
        super(FMCF, self).__init__()
        self.source_encoder = CodeBERT_Encoder(layers, model_dimension, attention_heads, feed_forward_dim,
                                              source_vocab_size, source_position_encoding, dropout_rate)
        self.graph_encoder = GAT_GraphEncoder(layers, model_dimension, attention_heads, feed_forward_dim,
                                              graph_vocab_size, attention_hops, graph_position_encoding, dropout_rate)
        self.joint_decoder = Joint_Fusion_Decoder(layers, model_dimension, attention_heads, feed_forward_dim,
                                                  target_vocab_size, target_position_encoding, dropout_rate)
        self.output_projection = tf.keras.layers.Dense(target_vocab_size)

    def Execute_Attention(self, source_inputs, graph_inputs, target_inputs, training, source_mask, graph_mask, target_mask):
        source_encoded, source_attention = self.source_encoder(source_inputs, training, source_mask)
        graph_encoded, graph_attention = self.graph_encoder(graph_inputs, training, graph_mask)

        combined_output, fusion_attention = self.joint_decoder(target_inputs, source_encoded, graph_encoded, training,
                                                               source_mask, graph_mask, target_mask)

        projected_output = self.output_projection(combined_output)

        return projected_output, fusion_attention, source_attention, graph_attention

# Example usage:
# Define the parameters for the model
num_layers = 6
model_dimension = 512
attention_heads = 8
feed_forward_dim = 2048
source_vocab_size = 10000
graph_vocab_size = 10000
attention_hops = 2
target_vocab_size = 10000
source_position_encoding = 1000
graph_position_encoding = 1000
target_position_encoding = 1000
dropout_rate = 0.1

# Create the model
model = FMCF(num_layers, model_dimension, attention_heads, feed_forward_dim,
                         source_vocab_size, graph_vocab_size, attention_hops, 
                         target_vocab_size, source_position_encoding, 
                         graph_position_encoding, target_position_encoding, dropout_rate)

# Dummy input data (You need to provide actual data in practice)
source_inputs = tf.random.uniform((32, 128), dtype=tf.int32)
graph_inputs = tf.random.uniform((32, 128), dtype=tf.int32)
target_inputs = tf.random.uniform((32, 50), dtype=tf.int32)
training = True
source_mask = None
graph_mask = None
target_mask = None

# Run the model
output, attentions = model(source_inputs, graph_inputs, target_inputs, training, source_mask, graph_mask, target_mask)