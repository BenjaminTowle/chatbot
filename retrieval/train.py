import pickle
import tensorflow as tf
import numpy as np
import random
import os
import chatbot_db as db
import pandas as pd
from retrieval.retriever import Retriever

#######################################
# MODEL HYPER-PARAMETERS
#######################################

EMBEDDING_DIM = 512
EPOCHS = 100
BATCH_SIZE = 30
LEARNING_RATE = 1e-4
NUM_SAMPLES = 30000
TRAIN_SPLIT = 0.9


def generator():
    """
    This function is the generator for the tensorflow Dataset object, which iterates through loading context-response
    embedding pairs into RAM, and then iterating through BATCH_SIZE sets of elements.  The function also applies a
    training / test split as determined by TRAIN_SPLIT.
    :return: BATCH_SIZE x 512 array of train/test contexts/responses and BATCH_SIZE array of train/test labels.
    """
    # Load training meta data if exists, otherwise initialise it
    if not os.path.exists("training_meta"):
        start_point = db.get_lowest_id()
        data_size = db.get_length()
        training_meta = {"start_point": start_point, "end_point": data_size+start_point, "epoch": 0,
                         "model_size": 256, "activation": "swish"}
    else:
        training_meta = pickle.load(open("training_meta", "rb"))

    while True:
        # Load section of relevant data into RAM and reset start_point if epoch completed
        data = db.retrieve(training_meta["start_point"], training_meta["start_point"] + NUM_SAMPLES)

        if training_meta["start_point"] + (NUM_SAMPLES*2) > training_meta["end_point"]:
            training_meta["start_point"] = 0
            training_meta["epoch"] += 1
        else:
            training_meta["start_point"] += NUM_SAMPLES

        # Save training meta every RAM load
        pickle.dump(training_meta, open("training_meta", "wb"))

        # Convert to pandas dataframe
        dataframe = pd.DataFrame(data, columns=["context", "response"])
        base_contexts = np.asarray(dataframe["context"].to_list())
        base_responses = np.asarray(dataframe["response"].to_list())

        contexts = np.zeros((NUM_SAMPLES * 2, 512), dtype=np.float)
        responses = np.zeros((NUM_SAMPLES * 2, 512), dtype=np.float)
        labels = np.zeros((NUM_SAMPLES * 2), dtype=np.float)
        for i in range(0, NUM_SAMPLES, 2):
            contexts[i] = base_contexts[i]
            contexts[i+1] = base_contexts[i]
            responses[i] = base_responses[i]
            responses[i+1] = base_responses[random.randint(0, NUM_SAMPLES-1)]
            labels[i] = 0.0
            labels[i+1] = 1.0

        # Iterate over mini-batches and split into test and train
        for j in range(int(NUM_SAMPLES*2/BATCH_SIZE)):
            start_ind = j * BATCH_SIZE
            end_ind = (j+1) * BATCH_SIZE
            end_ind_train = start_ind + int((end_ind-start_ind) * TRAIN_SPLIT)
            batch_contexts_train = contexts[start_ind:end_ind_train]
            batch_responses_train = responses[start_ind:end_ind_train]
            batch_labels_train = labels[start_ind:end_ind_train]
            batch_contexts_test = contexts[end_ind_train:end_ind]
            batch_responses_test = responses[end_ind_train:end_ind]
            batch_labels_test = labels[end_ind_train:end_ind]

            yield batch_contexts_train, batch_responses_train, batch_labels_train, batch_contexts_test, batch_responses_test, batch_labels_test


def main():
    @tf.function
    def train(c_train, r_train, l_train):
        """
        This function is called during the training loop and covers forwards and backwards propagation of the training
        data
        :param c_train: 2d tensor of BATCH_SIZE x 512
        :param r_train: 2d tensor of BATCH_SIZE x 512
        :param l_train: 1d tensor of BATCH_SIZE
        :return: 1d tensor of BATCH_SIZE representing contrastive loss
        """
        with tf.GradientTape() as tape:
            train_loss = retriever(c_train, r_train, l_train)

        variables = retriever.trainable_variables
        gradients = tape.gradient(train_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return train_loss

    # Convert to dataset object
    dataset = tf.data.Dataset.from_generator(generator,
        (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))

    # Create the model and load latest checkpoint if exists
    retriever = Retriever(model_size=EMBEDDING_DIM)
    if os.path.exists("retriever.h5"):
        retriever.load_weights("retriever.h5")
        retriever.initialise()
    optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

    print("Commencing training ...")
    display_loss_freq = 1000  # How often to show the current loss in terms of steps
    steps_per_epoch = int(db.get_length() / BATCH_SIZE)

    # Losses should be stored in lists so that average can be taken over the entire epoch
    train_losses = []
    test_losses = []
    for step, (c_train, r_train, l_train, c_test, r_test, l_test) in enumerate(dataset.take(-1)):
        loss = train(c_train, r_train, l_train)
        if loss.numpy().size > 0:
            train_losses.append(np.mean(loss.numpy()))
        else:
            # In case the loss output is empty list, note where this occurred
            print(step, c_train)

        # Print latest results every display_loss_freq steps
        if (step+1) % display_loss_freq == 0:
            test_loss = retriever(c_test, r_test, l_test)
            test_losses.append(np.mean(test_loss.numpy()))
            print(f"Steps completed: {step+1}; Train loss: {np.mean(train_losses)}; Test loss: {np.mean(test_losses)}")
            retriever.save_weights("retriever.h5")

        # Reset losses after every epoch
        if (step+1) % steps_per_epoch == 0:
            train_losses = []
            test_losses = []


if __name__ == "__main__":
    main()
