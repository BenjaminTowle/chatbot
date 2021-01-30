import tensorflow as tf
from transformers import MobileBertTokenizer, TFMobileBertForNextSentencePrediction
from Reranking.train import RERANKER_NAME, BASE_MODEL_NAME


class Reranker:
    def __init__(self):
        self.bert = TFMobileBertForNextSentencePrediction.from_pretrained(RERANKER_NAME)
        self.tokeniser = MobileBertTokenizer.from_pretrained(BASE_MODEL_NAME)

    def __call__(self, context, responses):
        """
        Reranks a given list of response candidates against a context input and outputs
        :param context: string
        :param responses: list of strings
        :return: 1D numpy array of re-ranking scores from 0 to 1, with 1 indicating maximum score
        """
        contexts = [context for _ in range(len(responses))]
        data = self.tokeniser(contexts, responses, max_length=40, truncation=True, padding=True, return_tensors="tf")

        logits = self.bert(data)
        outputs = tf.nn.softmax(logits["logits"], axis=-1).numpy()[:, 0]

        return outputs
