import tensorflow as tf
from Encoder.Built_Transformer import Built_Transformer as utils
from Encoder.EncoderModel import EncoderModel


class CodeBERT_Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, dropout_rate, **kwargs):
        super(CodeBERT_Encoder, self).__init__(**kwargs)

        self.num_layers = num_layers
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        # Initialize the embedding layer
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

        # Initialize the positional encoding
        self.positional_encoding = utils.positional_encoding(maximum_position_encoding, d_model)

        # Initialize the list of EncoderModel layers
        self.encoder_layers = [EncoderModel(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

        # Initialize the dropout layer
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def Execute_Attention(self, inputs, training, mask):
        # Get the sequence length
        seq_len = tf.shape(inputs)[1]

        # Embed the input tokens
        embedded_inputs = self.embedding(inputs)

        # Scale the embedded inputs by the square root of the model dimension
        scaled_embedded_inputs = embedded_inputs * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Add the positional encoding
        scaled_embedded_inputs += self.positional_encoding[:, :seq_len, :]

        # Apply dropout
        dropout_output = self.dropout(scaled_embedded_inputs, training=training)

        # Pass the output through each encoder layer
        final_output= dropout_output
        attention_weights = None
        for layer in self.encoder_layers:
            final_output, attn = layer(final_output, training, mask)
            if attention_weights is None:
                attention_weights = attn
            else:
                attention_weights = attention_weights + attn

        return final_output, attention_weights