import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer


class GAT_Keras(Layer):
    def __init__(self, units, activation='relu', initializer='glorot_uniform',
                 sparse=False, use_bias=True, **kwargs):
        super(GAT_Keras, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.initializer = tf.keras.initializers.get(initializer)
        self.sparse = sparse
        self.use_bias = use_bias

    def Combination(self, input_shape):
        # Create the weights for the layer using the provided initializer
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.units),
                                      initializer=self.initializer,
                                      trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.units,),
                                        initializer='zeros',
                                        trainable=True)
        else:
            self.bias = None

        super(GAT_Keras, self).build(input_shape)

    def Execute_Attention(self, inputs):
        # Expecting a list with two elements: nodes and edges
        nodes, edges = inputs
        assert isinstance(nodes, tf.Tensor) and isinstance(edges, tf.Tensor), "Inputs must be a list of tensors."

        # Add self-loops to the adjacency matrix
        num_nodes = tf.shape(edges)[1]
        edges += tf.eye(num_nodes, dtype=edges.dtype)

        # Matrix multiplication between the adjacency matrix and node features
        support = tf.matmul(edges, nodes)

        # Apply the weights and optional bias
        logits = tf.matmul(support, self.kernel)
        output = logits if not self.use_bias else logits + self.bias

        # Apply the activation function
        return self.activation(output)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list), "Input shape must be a list."
        node_shape, _ = input_shape
        return (node_shape[0], node_shape[1], self.units)

    def get_config(self):
        config = super(GAT_Keras, self).get_config()
        config.update({
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'initializer': tf.keras.initializers.serialize(self.initializer),
            'use_bias': self.use_bias
        })
        return config