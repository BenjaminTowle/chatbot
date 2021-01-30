from transformers import MobileBertTokenizer, TFMobileBertForNextSentencePrediction
import tensorflow as tf
import random
import numpy as np
from Datasets.dataloader import load_dataset

#################################
# MODEL HYPER-PARAMETERS
#################################
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
BASE_MODEL_NAME = "google/mobilebert-uncased"
RERANKER_NAME = "reranker_bert"


def train():
    bert = TFMobileBertForNextSentencePrediction.from_pretrained(BASE_MODEL_NAME)
    tokenizer = MobileBertTokenizer.from_pretrained(BASE_MODEL_NAME)

    contexts, responses = load_dataset()
    distractors = []
    for _ in range(len(contexts)):
        distractors.append(random.choice(responses))

    new_contexts = []
    new_responses = []
    targets = []
    for i in range(len(contexts)):
        new_contexts.append(contexts[i])
        new_contexts.append(contexts[i])
        new_responses.append(responses[i])
        new_responses.append(distractors[i])
        targets.append([1., 0.])
        targets.append([0., 1.])

    targets = np.asarray(targets)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    def train_step(inputs, outputs):
        with tf.GradientTape() as tape:
            logits = bert(inputs)["logits"]
            softmax = tf.nn.softmax(logits, axis=-1)
            loss = tf.keras.losses.CategoricalCrossentropy()(outputs, softmax)

        variables = bert.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return loss

    num_steps = int(len(new_contexts)/BATCH_SIZE)
    for i in range(EPOCHS):
        losses = []
        for j in range(num_steps):
            data = tokenizer(new_contexts[j*BATCH_SIZE: j*BATCH_SIZE + BATCH_SIZE], new_responses[j*BATCH_SIZE: j*BATCH_SIZE + BATCH_SIZE], truncation=True, max_length=40, padding=True, return_tensors="tf")
            batch_targets = targets[j*BATCH_SIZE: j*BATCH_SIZE + BATCH_SIZE]
            loss = train_step(data, batch_targets)
            losses.append(loss.numpy())

            print(f"Epoch: {i+1}; Step: {j+1}; Loss: {np.mean(losses)}")

            bert.save_pretrained(RERANKER_NAME)
