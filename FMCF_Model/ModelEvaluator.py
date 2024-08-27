import tensorflow as tf
from modules import MMTrans, TransformerUtils, CustomSchedule
from Data_Preprocessing import Split_comments, test_data_prepare
from DataOutput import MAX_LENGTH_COMM
from EvaluationMetrics import EvaluationMetrics
from Configs import Eval_args
import numpy as np
import matplotlib.pyplot as plt


class ModelEvaluator:
    def __init__(self, dataset_path, model_config):
        self.dataset_path = dataset_path
        self.model = MMTrans(**model_config)
        self.dynamic_lr = CustomSchedule(model_config['d_model'])
        self.training_optimizer = tf.keras.optimizers.Adam(self.dynamic_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    def inference_step(self, sources, nodes, edges, targets, metrics):
        batch_size = targets.shape[0]
        target_sequences = targets[:, 1:]
        vocab_dict = Split_comments.get_vocab(self.dataset_path)
        initial_token_id = len(vocab_dict) + 1
        sequence_predictions = [initial_token_id] * batch_size
        prediction_sequence = tf.expand_dims(sequence_predictions, 1)

        for _ in range(MAX_LENGTH_COMM - 1):
            masks = TransformerUtils.create_masks(sources, nodes, prediction_sequence)
            sequence_output, attention_data, _, _ = self.model(sources, nodes, edges, prediction_sequence,
                                                               training=False, **masks)

            next_token_predictions = sequence_output[:, -1, :]
            next_token_ids = tf.cast(tf.argmax(next_token_predictions, axis=-1), tf.int32)

            prediction_sequence = tf.concat([prediction_sequence, next_token_ids], axis=-1)

            # Stop if all sequences have reached the end token
            if all(len(vocab_dict) + 2 in sequence_prediction.numpy() for sequence_prediction in prediction_sequence):
                break

        decoded_sequences = EvaluationMetrics.remove_padding(
            prediction_sequence.numpy().tolist(), vocab_dict['<end>'], mode="candidates")
        actual_sequences = EvaluationMetrics.remove_padding(
            target_sequences.numpy().tolist(), vocab_dict['<end>'], mode="references")

        for actual, predicted in zip(actual_sequences, decoded_sequences):
            metrics['sentence_bleu'].append(EvaluationMetrics.bleu_score(actual, predicted))
            metrics['rouge'].append(EvaluationMetrics.rouge_score(actual, predicted))
            metrics['meteor'].append(EvaluationMetrics.meteor_score(actual, predicted))

        metrics['corpus_bleu'].append(EvaluationMetrics.corpus_bleu_score(actual_sequences, decoded_sequences))

    def perform_evaluation(self):
        test_data = test_data_prepare(self.dataset_path)
        metrics = {
            'sentence_bleu': [],
            'corpus_bleu': [],
            'rouge': [],
            'meteor': []
        }
        print("Starting evaluation...")

        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.training_optimizer)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, './checkpoints/' + self.dataset_path, max_to_keep=10)

        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print("Loaded the latest checkpoint.")

        for batch, data in enumerate(test_data):
            self.inference_step(*data, metrics)
            if batch % 50 == 0:
                print(f"Processed batch {batch}")

        print("Evaluation complete.")
        for metric_name, values in metrics.items():
            print(f"{metric_name}: {np.mean(values):.4f}")

    def visualize_attention(self, attention_weights, source_sequence, generated_sequence, attention_type):
        # This method would visualize the attention weights; implementation details would vary.
        pass

