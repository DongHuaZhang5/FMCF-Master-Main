import tensorflow as tf
from Encoder.Built_Transformer import Built_Transformer as utils


class Joint_Fusion_Decoder(tf.keras.layers.Layer):
    def __init__(self, model_dim, attention_heads, feed_forward_dim, dropout_rate):
        super(IntegratedDecoderModule, self).__init__()
        self.decoder_layer = DecoderModel(model_dim, attention_heads, feed_forward_dim, dropout_rate)

    def Execute_Attention(self, inputs, additional_input, training, mask, padding_mask):
        output, attention = self.decoder_layer(inputs, additional_input, training, mask, padding_mask)
        return output, attention


class UnifiedDecoder(tf.keras.layers.Layer):
    def __init__(self, layers, model_dim, attention_heads, feed_forward_dim, vocab_size,
                 max_position_encoding, dropout_rate):
        super(UnifiedDecoder, self).__init__()
        self.model_dim = model_dim
        self.layers = layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_dim)
        self.positional_encoding = utils.positional_encoding(max_position_encoding, model_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.decoder_modules = [IntegratedDecoderModule(model_dim, attention_heads, feed_forward_dim, dropout_rate)
                                for _ in range(layers)]

    def Execute_Attention(self, target, sbt_features, graph_features, training,
             sbt_padding_mask, node_padding_mask, look_ahead_mask):
        seq_len = tf.shape(target)[1]

        # Embedding and position encoding
        embedded_target = self.embedding(target)
        embedded_target *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        embedded_target += self.positional_encoding[:, :seq_len, :]
        embedded_target = self.dropout(embedded_target, training=training)

        # Separate streams for SBT and graph features
        sbt_stream = sbt_features
        graph_stream = graph_features

        attention_weights = {}

        # Processing through decoder modules
        for i, module in enumerate(self.decoder_modules):
            if i < (self.layers // 2):
                # Update the SBT stream
                embedded_target, (sbt_attention, _) = module(embedded_target, sbt_stream,
                                                             training, look_ahead_mask, sbt_padding_mask)
                attention_weights[f'sbt_attention_{i + 1}'] = sbt_attention
            else:
                # Update the graph stream
                embedded_target, (graph_attention, _) = module(embedded_target, graph_stream,
                                                               training, look_ahead_mask, node_padding_mask)
                attention_weights[f'graph_attention_{i + 1}'] = graph_attention

        # Integrate the SBT and graph streams
        integrated_output = tf.concat([embedded_target, sbt_stream, graph_stream], axis=-1)

        return integrated_output, attention_weights

# Example usage:
# layers = 6
# model_dim = 512
# attention_heads = 8
# feed_forward_dim = 2048
# vocab_size = 10000
# max_position_encoding = 1000
# dropout_rate = 0.1
# decoder = UnifiedDecoder(layers, model_dim, attention_heads, feed_forward_dim, vocab_size, max_position_encoding, dropout_rate)
# target, sbt_features, graph_features = ...  # Your input tensors
# training = ...  #  training flag
# sbt_padding_mask, node_padding_mask, look_ahead_mask = ...  # Your masks
# output, attention_weights = decoder(target, sbt_features, graph_features, training, sbt_padding_mask, node_padding_mask, look_ahead_mask)