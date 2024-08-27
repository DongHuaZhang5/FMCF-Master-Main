import tensorflow as tf
from FMCF_Model.FMCF import IntegratedModel as LanguageModel
from Decoder.Learning_Rate import AdaptiveLearningRate
from Encoder.Built_Transformer import Built_Transformer as TransformerUtils
import time
from DataOutput import data_prepare
from Data_Preprocessing.Split_comments import get_vocab
from FMCF_Model.EvaluationMetrics import TextEvaluator as Evaluator
import numpy as np
from FMCF_Model.Configs import Train_args
import os

class ModelTrainer:
    def __init__(self, training_cycles, dataset_path, checkpoint_limit, improvement_patience, model_config):
        self.training_cycles = training_cycles
        self.dataset_path = dataset_path
        self.checkpoint_limit = checkpoint_limit
        self.improvement_patience = improvement_patience
        self.model_performance = tf.keras.metrics.Mean(name='model_performance')
        self.model = LanguageModel(**model_config)
        self.dynamic_lr = AdaptiveLearningRate(model_config['d_model'])
        self.adam_optimizer = tf.keras.optimizers.Adam(self.dynamic_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    def compile_train_step(self, inputs, targets):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def validation_step(self, sbts_val, nodes_val, edges_val, comms_val, bleu_scores):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def evaluate_model(self, validation_data, best_bleu_score, checkpoint_manager, epoch_index, evaluation_metric):
        bleu_scores = []
        print("Starting evaluation...")

        for batch, (sbts_val, nodes_val, edges_val, comms_val) in enumerate(validation_data):
            self.validation_step(sbts_val, nodes_val, edges_val, comms_val, bleu_scores)

        average_bleu_score = np.mean(bleu_scores)
        print("Evaluation completed! Average {} score: {:.4f}".format(evaluation_metric, average_bleu_score))

        if average_bleu_score > best_bleu_score:
            best_bleu_score = average_bleu_score
            checkpoint_path = checkpoint_manager.save(checkpoint_directory="./checkpoints/" + self.dataset_path,
                                                      checkpoint_number=epoch_index)
            print('Checkpoint saved at epoch {} to {}'.format(epoch_index + 1, checkpoint_path))
            print("New best {} score: {}".format(evaluation_metric, best_bleu_score))

        return best_bleu_score

    def apply_early_stopping(self, best_scores, current_score):
        best_scores.append(current_score)
        if len(best_scores) >= self.improvement_patience:
            if all(score == best_scores[0] for score in best_scores[-self.improvement_patience:]):
                print("No improvement in the last {} epochs. Stopping training.".format(self.improvement_patience))
                return True
        return False

    def start_training(self):
        train_data, validation_data = data_prepare(self.dataset_path)
        checkpoint_path = "./checkpoints/" + self.dataset_path
        checkpoint_manager = tf.train.CheckpointManager(self.model, checkpoint_path, max_to_keep=self.checkpoint_limit)

        best_bleu_score = 0
        best_scores = []
        stop_training = False

        for epoch in range(self.training_cycles):
            start_time = time.time()

            # Training logic here
            # ...

            # Evaluate after every epoch
            best_bleu_score = self.evaluate_model(validation_data, best_bleu_score, checkpoint_manager, epoch,
                                                  "bleu_score")

            # Check for early stopping
            if self.apply_early_stopping(best_scores, best_bleu_score):
                stop_training = True
                break

            print('Epoch {} completed in {} seconds\n'.format(epoch + 1, time.time() - start_time))

