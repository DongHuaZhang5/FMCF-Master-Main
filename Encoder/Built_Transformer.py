import tensorflow as tf
import numpy as np

class Built_Transformer:

    @staticmethod
    @tf.function
    def get_angles(pos, i, d_model):
        angle_rates = 1 / tf.math.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    @staticmethod
    @tf.function
    def positional_encoding(position, d_model):
        angle_rads = Built_Transformer.get_angles(tf.range(position)[:, tf.newaxis],
                                                  tf.range(d_model)[tf.newaxis, :],
                                                  d_model)
        angle_rads = tf.where(tf.math.mod(angle_rads, 2) == 0, tf.sin(angle_rads), tf.cos(angle_rads))
        pos_encoding = tf.expand_dims(angle_rads, axis=0)
        return tf.cast(pos_encoding, tf.float32)

    @staticmethod
    @tf.function
    def create_padding_mask(seqs):
        return tf.where(tf.equal(seqs, 0), 1.0, 0.0)

    @staticmethod
    @tf.function
    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    @staticmethod
    @tf.function
    def scaled_dot_product_attention(q, k, v, mask):
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
        scaled_attention_logits = tf.where(mask is not None, scaled_attention_logits + (mask * -1e9), scaled_attention_logits)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    @staticmethod
    @tf.function
    def point_wise_feed_forward_network(d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

    @staticmethod
    @tf.function
    def loss_function(real, pred):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        mask = tf.math.logical_not(tf.equal(real, 0))
        loss_ = loss_object(real, pred)
        loss_ *= tf.cast(mask, loss_.dtype)
        return tf.reduce_mean(loss_)

    @staticmethod
    @tf.function
    def create_masks(sbt_inp, node_inp, tar):
        sbt_padding_mask = Built_Transformer.create_padding_mask(sbt_inp)
        node_padding_mask = Built_Transformer.create_padding_mask(node_inp)
        look_ahead_mask = Built_Transformer.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = Built_Transformer.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return sbt_padding_mask, node_padding_mask, combined_mask

    @staticmethod
    @tf.function
    def self_attention(x, y, mask):
        attn = tf.keras.layers.dot([x, y], axes=[2, 2])
        attn = tf.where(mask is not None, attn + (mask * -1e9), attn)
        attn = tf.nn.softmax(attn)
        output = tf.keras.layers.dot([attn, x], axes=[2, 1])
        return output