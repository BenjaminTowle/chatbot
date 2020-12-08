import tensorflow as tf
import numpy as np


class Retriever(tf.keras.models.Model):
    """
    Retrieval model for the chatbot which functions through a siamese network, that separately maps both contexts and
    responses into shared hidden space.  Sentence embeddings for both contexts and responses are initialised with the
    Universal Sentence Encoder (USE).  In this implementation, they are loaded into RAM in large batches defined by the
    NUM_SAMPLES hyper-parameter.  Both are then put into a separate Feed-forward Network, outputing a vector of length
    model_size.  Loss used is contrastive loss on the cosine similarity of the vectors.
    """
    def __init__(self, model_size=256, activation="swish"):
        super(Retriever, self).__init__()

        # Define output layer
        self.outputs_contexts = tf.keras.layers.Dense(model_size, activation=activation)
        self.outputs_responses = tf.keras.layers.Dense(model_size, activation=activation)


    def call(self, contexts, responses, labels):
        """
        This method is called during the training loop and outputs the contrastive loss
        :param contexts: tensor of n x d samples
        :param responses: tensor of n x d samples, containing split of true and false responses
        :param labels: tensor of n samples, with each value as 0 (true) or 1 (false) corresponding to the response
        :return: tensor of n elements for contrastive loss
        """
        contexts_out = self.outputs_contexts(contexts)
        responses_out = self.outputs_responses(responses)

        l2_contexts = tf.nn.l2_normalize(contexts_out, axis=-1)
        l2_responses = tf.nn.l2_normalize(responses_out, axis=-1)

        cosim = 1.0 - (tf.losses.cosine_similarity(l2_contexts, l2_responses) * -1.0)

        positive = (1.0 - labels) * 0.5 * tf.math.pow(cosim, 2)
        negative = labels * 0.5 * tf.math.pow(tf.maximum(0.0, 1.0-cosim), 2)

        contrastive_loss = tf.add(positive, negative)

        return contrastive_loss

    def initialise(self):
        """
        This method needs to be called before pretrained weights are loaded, as the h5 file format that weight are saved
        in does not know how layers are connected.
        """
        self(tf.constant([np.zeros(512, dtype=np.float32)]), tf.constant([np.zeros(512, dtype=np.float32)]),
             tf.constant([np.zeros(1, dtype=np.float32)]))

    def predict(self, contexts, responses):
        """
        This method makes predictions on a given context and response on the likelihood of them being a true pair,
        and outputs a cosine similarity.
        :param contexts: 2d tensor containing batch-size x 512-d encoded sentence(s)
        :param responses: 2d tensor containing candidates x 512-d encoded sentences
        :return: 1d tensor of cosine similarity for each candidate with 1 implying a perfect match and 0 no match.
        """
        contexts_out = self.outputs_contexts(contexts)
        responses_out = self.outputs_responses(responses)

        l2_contexts = tf.nn.l2_normalize(contexts_out, axis=-1)
        l2_responses = tf.nn.l2_normalize(responses_out, axis=-1)

        cosim = tf.matmul(l2_responses, l2_contexts, transpose_b=True)

        return cosim
