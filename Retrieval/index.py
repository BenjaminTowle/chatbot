import numpy as np
import pickle
import faiss
from Retrieval.retriever import Retriever
from Retrieval.train import EMBEDDING_DIM
from Datasets.dataloader import load_dataset
from Encoding.encoder import encode

#####################################
# INDEX HYPER-PARAMETERS
#####################################
N_CLUSTERS = 10
BATCH_SIZE = 100
RESPONSE_INDEX_PATH = "response_vectors.index"
CONTEXT_INDEX_PATH = "context_vectors.index"

retriever = Retriever(EMBEDDING_DIM)


# Indexes the Empathetic Dialogues dataset according to embedding of context for context-context matching, as well as
# mapping of response for context-response matching
def create_index():
    context_quantiser = faiss.IndexFlatL2(EMBEDDING_DIM)
    context_index = faiss.IndexIVFFlat(context_quantiser, EMBEDDING_DIM, N_CLUSTERS, faiss.METRIC_INNER_PRODUCT)
    response_quantiser = faiss.IndexFlatL2(EMBEDDING_DIM)
    response_index = faiss.IndexIVFFlat(response_quantiser, EMBEDDING_DIM, N_CLUSTERS, faiss.METRIC_INNER_PRODUCT)

    contexts, responses = load_dataset(["ed"])

    for i in range(0, len(responses) - BATCH_SIZE, BATCH_SIZE):
        print(i)
        batch_contexts = contexts[i:i + BATCH_SIZE]
        batch_responses = responses[i:i + BATCH_SIZE]

        # embed contexts and responses
        contexts_encoded = encode(batch_contexts)
        responses_encoded = retriever.encode_responses(batch_responses)

        context_index.train(contexts_encoded)
        context_index.add(contexts_encoded)
        response_index.train(responses_encoded)
        response_index.add(responses_encoded)

    faiss.write_index(context_index, CONTEXT_INDEX_PATH)
    pickle.dump(contexts, open("contexts.pickle", "wb"))

    faiss.write_index(response_index, RESPONSE_INDEX_PATH)
    pickle.dump(responses, open("responses.pickle", "wb"))
