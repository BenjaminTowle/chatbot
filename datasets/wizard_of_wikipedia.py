import json
import pickle


def load_dataset():
    """
    Load dataset from Wizard of Wikipedia paper: https://arxiv.org/pdf/1811.01241.pdf
    Dataset can be downloaded here: http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz
    :return: list of contexts, list of responses
    """
    f = json.load(open("datasets/wow_train.json"))
    contexts = []
    responses = []
    for _, conv in enumerate(f):
        dialog = []
        for utt in conv["dialog"]:
            dialog.append(utt["text"])

        for i in range(len(dialog)-1):
            contexts.append(dialog[i])
            responses.append(dialog[i+1])

    pickle.dump((contexts, responses), open("wow_data.pickle", "wb"))

    return contexts, responses
    
