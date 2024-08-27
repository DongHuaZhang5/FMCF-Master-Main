import tensorflow as tf
import numpy as np

class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, key_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.key_dim = key_dim

    def Execute_Attention(self, queries, keys, values):
        # Calculate the attention scores
        scores = tf.matmul(queries, keys, transpose_b=True) / np.sqrt(self.key_dim)
        # Apply softmax to the scores
        attention_weights = tf.nn.softmax(scores, axis=-1)
        # Calculate the context using the attention weights and values
        context = tf.matmul(attention_weights, values)
        return context, attention_weights

class Feature_Fusion(tf.keras.layers.Layer):
    def __init__(self, ast_node_dim, src_seq_length, key_dim):
        super(FusionMultiModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=ast_node_dim, kernel_size=1, strides=1)
        self.conv2 = tf.keras.layers.Conv1D(filters=ast_node_dim, kernel_size=1, strides=1)
        self.encoder_self_attention = ScaledDotProductAttention(key_dim)

    def Execute_Attention(self, GAT_embeddings, source_embeddings, ast_embeddings):
        # Apply convolution to the inputs
        GAT_output = self.conv1(GAT_embeddings_embeddings)
        source_output = self.conv2(source_embeddings)
        # Apply self-attention to the outputs of the convolutions
        outputs, attention = self.encoder_self_attention(gcn_output, source_output, source_embeddings)
        # Fuse the attention outputs with the AST embeddings
        fused_output = outputs + ast_embeddings
        return fused_output

# Example usage:
# ast_node_dim = 64  # Example dimension for AST node embeddings
# src_seq_length = 50  # Example source sequence length
# key_dim = 64  # Example dimension for keys in attention mechanism
# model = FusionMultiModel(ast_node_dim, src_seq_length, key_dim)
# gcn_embed, src_embed, AST_embed = ...
# output = model(gcn_embed, src_embed, AST_embed)