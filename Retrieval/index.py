import pickle
import faiss
from Retrieval.retriever import Retriever
from Retrieval.train import EMBEDDING_DIM
from Datasets.dataloader import load_dataset

#####################################
# INDEX HYPER-PARAMETERS
#####################################
N_CLUSTERS = 10
BATCH_SIZE = 200
INDEX_PATH = "vectors.index"

retriever = Retriever(EMBEDDING_DIM)


# Indexes the Empathetic Dialogues dataset - this contains much higher quality responses than the Reddit dataset
# and they are usually more broadly applicable to a range of context sentences
def create_index():
    quantiser = faiss.IndexFlatIP(EMBEDDING_DIM)
    index = faiss.IndexIVFFlat(quantiser, EMBEDDING_DIM, N_CLUSTERS, faiss.METRIC_INNER_PRODUCT)
    utterances = []

    _, responses = load_dataset()
    utterances += responses

    for i in range(0, len(responses)-BATCH_SIZE, BATCH_SIZE):
        print(i)
        batch_responses = responses[i:i+BATCH_SIZE]

        # embed responses
        outputs_norm = retriever.encode_responses(batch_responses)

        index.train(outputs_norm)
        index.add(outputs_norm)

    faiss.write_index(index, INDEX_PATH)
    pickle.dump(utterances, open("utterances.pickle", "wb"))
