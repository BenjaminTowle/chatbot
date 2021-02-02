import pickle
import faiss
import numpy as np
from Retrieval.index import CONTEXT_INDEX_PATH, RESPONSE_INDEX_PATH, create_index
from Retrieval.retriever import Retriever, MODEL_NAME
from Retrieval.train import train as retrieval_train
from Reranking.reranker import Reranker
from Reranking.train import RERANKER_NAME, train as reranking_train
from Datasets.dataloader import load_dataset
from Datasets.reddit import get_subreddits
from Retrieval.Unpaired import unpaired_manager
from Encoding.encoder import encode
from itertools import chain
import os
import sys

########################################
# HYPER-PARAMETERS
########################################
N_PROBE = 5  # Number of clusters the indexer should visit
K = 10  # Number of response candidates to be retrieved at each turn


# This is the main file to run to interact with the model, once the components have been trained
def main():
    """
    This file coordinates the entire preprocessing, training and inference pipeline.  Models are saved regularly
    throughout training (hyper-parameters can be specified in the relevant train.py file), so file does not need
    to be ran end-to-end in one go.  Simply set the pipeline variable to the desired value.
    Model training to inference pipeline:
    1) Load data from reddit.py and store in database
    2) Train retrieval model with contrastive loss on reddit data
    3) Train a reranker model on Wizard of Wikipedia, Persona Chat and Empathetic Dialogues data
    4) Run inference
    """

    # Change this value to whichever step in the pipeline you want to start/continue with
    pipeline = 4

    #########################
    # PREPROCESSING STAGE
    #########################
    if pipeline == 1:
        get_subreddits()
        pipeline = 2

    #########################
    # TRAINING STAGE
    #########################
    if pipeline == 2:
        retrieval_train()
        pipeline = 3

    if pipeline == 3:
        reranking_train()

    ########################
    # INFERENCE STAGE
    ########################

    # Check that required models exist
    if not os.path.exists(MODEL_NAME):
        print("Retrieval model not found: set pipeline to 2 to train the retrieval model")
        sys.exit()

    if not os.path.exists(RERANKER_NAME):
        print("Reranker model not found: set pipeline to 3 to train the reranking model")
        sys.exit()

    if not os.path.exists(CONTEXT_INDEX_PATH):
        print("Initialising response candidate index")
        create_index()

    # Load indexers
    context_index = faiss.read_index(CONTEXT_INDEX_PATH, faiss.IO_FLAG_MMAP)
    response_index = faiss.read_index(RESPONSE_INDEX_PATH, faiss.IO_FLAG_MMAP)

    # Load context candidates
    if not os.path.exists("responses.pickle"):
        _, responses = load_dataset()
    else:
        responses = pickle.load(open("responses.pickle", "rb"))

    # Sets the number of clusters to be visited by the index
    context_index.nprobe = N_PROBE
    response_index.nprobe = N_PROBE
    dialog_history = []

    # Load retrieval and reranking models
    retriever = Retriever()
    reranker = Reranker()
    while True:
        msg = input("you: ")

        # Encode context
        use_encoding, map_encoding = retriever.encode_contexts([msg])

        k = K
        # Get context-context candidates
        _, con_indices = context_index.search(use_encoding, k)

        # Get context-response candidates
        _, res_indices = response_index.search(map_encoding, k)
        candidates = []
        for ind in chain(con_indices[0], res_indices[0]):
            # Filters out responses that have already been used in previous turns
            if responses[ind] not in dialog_history and responses[ind] not in candidates:
                candidates.append(responses[ind])

        # Get BERT re-ranking score (given as probability of context and response being a true pair)
        bert_scores = reranker(msg, candidates)
        dialog_history.append(candidates[np.argmax(bert_scores)])
        print(bert_scores[np.argmax(bert_scores)], candidates[np.argmax(bert_scores)])


if __name__ == "__main__":
    main()
