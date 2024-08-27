from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import numpy as np
from rouge import Rouge


class TextEvaluator:
    @staticmethod
    def uniform_bleu_score(references, hypotheses, smoothing_function):
        return sentence_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=smoothing_function)

    @staticmethod
    def aggregate_bleu_score(references, hypotheses, smoothing_function):
        return corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                           smoothing_function=smoothing_function)

    @staticmethod
    def rouge_l_score(reference, hypothesis):
        rouge_instance = Rouge()
        scores = rouge_instance.get_scores(" ".join(hypothesis), " ".join(reference))
        return scores[0]['rouge-l']['f']

    @staticmethod
    def meteor_score(references, hypothesis):
        return meteor_score([references], hypothesis)

    @staticmethod
    def filter_sequences(token_sequences, end_token, output_type):
        processed = []
        for sequence in token_sequences:
            filtered = [token for token in sequence if token != end_token]
            if output_type == "hypotheses":
                processed.append(filtered[1:])  # Exclude the start token
            elif output_type == "references":
                processed.append([filtered])  # Return as list of lists
        return processed

    @staticmethod
    def evaluate(candi, refs):
        bleu_scores = [
            TextEvaluator.uniform_bleu_score([refs[i][0]], [candi[i]], SmoothingFunction().method1)
            for i in range(len(candi))
        ]
        print("Sentence BLEU scores:", np.mean(bleu_scores))
        print("Corpus BLEU score:", TextEvaluator.aggregate_bleu_score(
            *zip(*refs), *[[c] for c in candi], SmoothingFunction().method1))


if __name__ == "__main__":
    candidates = [
        [1234, 12, 4, 5, 34, 1235],
        [1234, 22, 41, 35, 12, 1235],
        [1234, 34, 23, 22, 34, 123, 33, 23]
    ]
    references = [
        [12, 4, 5, 34, 1235],
        [22, 41, 34, 12, 1235],
        [34, 23, 22, 34, 123, 33, 23]
    ]

    candidates = TextEvaluator.filter_sequences(candidates, 1235, "hypotheses")
    references = TextEvaluator.filter_sequences(references, 1235, "references")

    TextEvaluator.evaluate(candidates, references)