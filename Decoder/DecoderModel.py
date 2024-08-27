import tensorflow as tf
from Encoder.Multi_Head_Attention import Multi_Head_Attention
from Encoder.Built_Transformer import Built_Transformer as TransformerUtils

class DecoderModel(tf.keras.layers.Layer):
    def __init__(self, dimensionality, focus_points, feed_forward_expansion, dropout_factor):
        super(ConfluenceDecoder, self).__init__()
        self.dimensionality = dimensionality
        self.focus_points = focus_points
        self.feed_forward_expansion = feed_forward_expansion
        self.dropout_factor = dropout_factor

        # Initialize two distinct multi-head attention mechanisms
        self.intra_attention = Multi_Head_Attention(dimensionality, focus_points)
        self.inter_attention = Multi_Head_Attention(dimensionality, focus_points)

        # Initialize a feed-forward network with the specified dimensionality and expansion
        self.ffdn = TransformerUtils.point_wise_feed_forward_network(dimensionality, feed_forward_expansion)

        # Layer normalization instances
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout layers to prevent overfitting
        self.noise1 = tf.keras.layers.Dropout(dropout_factor)
        self.noise2 = tf.keras.layers.Dropout(dropout_factor)
        self.noise3 = tf.keras.layers.Dropout(dropout_factor)

    def Confluence_Attention_Mechanism(self, decoder_inputs, encoder_sequence, is_training,
                                      future_mask, external_mask):
        # First phase of attention: intra-decoder self-attention
        intermediate_output, intra_weights = self.intra_attention(
            decoder_inputs, decoder_inputs, decoder_inputs, future_mask
        )
        refined_output = self.noise1(intermediate_output, training=is_training)
        combined_features = self.norm1(refined_output + decoder_inputs)

        # Second phase of attention: cross-attention between decoder and encoder
        cross_output, cross_weights = self.inter_attention(
            encoder_sequence, encoder_sequence, combined_features, external_mask
        )
        refined_cross_output = self.noise2(cross_output, training=is_training)
        integrated_features = self.norm2(refined_cross_output + combined_features)

        # Feed-forward network with dropout
        feed_forward_result = self.ffdn(integrated_features)
        final_output = self.noise3(feed_forward_result, training=is_training)
        output_with_residual = self.norm3(final_output + integrated_features)

        return output_with_residual, intra_weights, cross_weights

# Example usage:
# dimensionality = 512
# focus_points = 8
# feed_forward_expansion = 2048
# dropout_factor = 0.1
# decoder = ConfluenceDecoder(dimensionality, focus_points, feed_forward_expansion, dropout_factor)
# decoder_inputs, encoder_sequence = ...  # Your input tensors
# is_training, future_mask, external_mask = ...  # Your training flag and masks
# output, intra_weights, cross_weights = decoder.Confluence_Attention_Mechanism(decoder_inputs, encoder_sequence, is_training, future_mask, external_mask)