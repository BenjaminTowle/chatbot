import tensorflow as tf
import sentencepiece as spm

# This file loads the lite version of Universal Sentence Encoder
# the model can be downloaded from https://tfhub.dev/google/universal-sentence-encoder-lite/2?tf-hub-format=compressed

try:
    model = tf.saved_model.load("use-lite3")
except OSError:
    print("Model must first be downloaded from: https://tfhub.dev/google/universal-sentence-encoder-lite/2?tf-hub-format=compressed")



inference = model.signatures["default"]

sp = spm.SentencePieceProcessor()
sp.Load("use-lite/assets/universal_encoder_8k_spm.model")


def process_to_IDs_in_sparse_format(sp, sentences):
    # An utility method that processes sentences with the sentence piece processor
    # 'sp' and returns the results in tf.SparseTensor-similar format:
    # (values, indices, dense_shape)
    ids = [sp.EncodeAsIds(x) for x in sentences]
    max_len = max(len(x) for x in ids)
    dense_shape=(len(ids), max_len)
    values=[item for sublist in ids for item in sublist]
    indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return values, indices, dense_shape


def encode(sentences):
    values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, sentences)
    outputs = inference(values=tf.constant(values, dtype=tf.int64), indices=tf.constant(indices, dtype=tf.int64), dense_shape=tf.constant(dense_shape, dtype=tf.int64))

    array = outputs["default"]
    array = array.numpy()

    return array
