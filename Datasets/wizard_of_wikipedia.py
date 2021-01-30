import json


def load_dataset():
    """
    Load dataset from Wizard of Wikipedia paper: https://arxiv.org/pdf/1811.01241.pdf
    :return: list of contexts, list of responses
    """
    f = json.load(open("Datasets/wow/train.json"))
    contexts = []
    responses = []
    for _, conv in enumerate(f):
        dialog = []
        for utt in conv["dialog"]:
            dialog.append(utt["text"])

        for i in range(len(dialog)-1):
            contexts.append(dialog[i])
            responses.append(dialog[i+1])

    return contexts, responses
