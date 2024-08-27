import os
import tensorflow as tf
import numpy as np
import pickle as pkl
import random

BUFFER_SIZE = 20000
BATCH_SIZE = 100  # Adjust based on whether generated comments are shown one by one
MAX_LENGTH_SBT = 602
MAX_LENGTH_NODE = 200
MAX_LENGTH_COMM = 22


def truncate_sequences(sbt_seq, node, edge, comm):
    if tf.size(sbt_seq) > MAX_LENGTH_SBT:
        sbt_seq = tf.slice(sbt_seq, [0], [MAX_LENGTH_SBT])
    if tf.size(node) > MAX_LENGTH_NODE:
        node = tf.slice(node, [0], [MAX_LENGTH_NODE])
        edge = tf.slice(edge, [0, 0], [MAX_LENGTH_NODE, tf.shape(edge)[1]])
    return sbt_seq, node, edge, comm


def create_dataset_train_val(sbts_train, nodes_train, edges_train, comms_train,
                             sbts_val, nodes_val, edges_val, comms_val):
    def training_generator():
        for sbt, node, edge, comm in zip(sbts_train, nodes_train, edges_train, comms_train):
            yield (sbt, node, edge.numpy().astype(np.int32), comm)

    def validation_generator():
        for sbt, node, edge, comm in zip(sbts_val, nodes_val, edges_val, comms_val):
            yield (sbt, node, edge.numpy().astype(np.int32), comm)

    train_dataset = tf.data.Dataset.from_generator(training_generator,
                                                   output_types=(tf.int32, tf.int32, tf.int32, tf.int32))
    train_dataset = train_dataset.map(truncate_sequences)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=(
    (None,), (None,), (None, None), (None,)))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    validation_dataset = tf.data.Dataset.from_generator(validation_generator,
                                                        output_types=(tf.int32, tf.int32, tf.int32, tf.int32))
    validation_dataset = validation_dataset.map(truncate_sequences)
    validation_dataset = validation_dataset.cache()
    validation_dataset = validation_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=(
    (None,), (None,), (None, None), (None,)))
    validation_dataset = validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, validation_dataset


def prepare_data(sub_data_folder):
    with open(f"./datasets/{sub_data_folder}/tokens_idx/sbts_train.pkl", "rb") as fr:
        srcs_train = pkl.load(fr)
    # ... (other data loading as above)

    return create_dataset_train_val(srcs_train, nodes_train, edges_train, comms_train,
                                    srcs_val, nodes_val, edges_val, comms_val)


def create_dataset_test(srcs_test, nodes_test, edges_test, comms_test):
    def test_generator():
        for src, node, edge, comm in zip(srcs_test, nodes_test, edges_test, comms_test):
            yield (src, node, edge.numpy().astype(np.int32), comm)

    test_dataset = tf.data.Dataset.from_generator(test_generator, output_types=(tf.int32, tf.int32, tf.int32, tf.int32))
    test_dataset = test_dataset.map(truncate_sequences)
    test_dataset = test_dataset.cache()
    test_dataset = test_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None,), (None,), (None, None), (None,)))
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return test_dataset


def get_test_data(sub_data_folder):
    with open(f"./datasets/{sub_data_folder}/tokens_idx/sbts_test.pkl", "rb") as fr:
        srcs_test = pkl.load(fr)
    # ... (other data loading as above)

    return create_dataset_test(srcs_test, nodes_test, edges_test, comms_test)