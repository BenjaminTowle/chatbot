import pickle


def load_dataset():
    """
    Loads the dataset from the EmpatheticDialogues paper: https://arxiv.org/abs/1811.00207
    Dataset can be downloaded here: https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz
    :return: list of contexts, list of responses
    """
    data = []
    with open("datasets/ed_train.csv", "r") as f:
        lines = f.readlines()
        dialog = []
        cur_id = None
        for line in lines[1:]:  # Skip the first line which contains headers
            elements = line.split(",")
            # Check if next dialogues has started
            if cur_id is None:
                cur_id = elements[0]
            elif cur_id != elements[0]:
                if len(dialog) > 1:  # Reject any singleton dialogues
                    data.append(dialog)
                dialog = []
                cur_id = elements[0]

            dialog.append(elements[5].lower().replace("_comma_", ""))

    contexts = []
    responses = []
    for dialog in data:
        for i in range(1, len(dialog)):
            contexts.append(dialog[i-1])
            responses.append(dialog[i])

    pickle.dump(data, open("ed_dialogues", "wb"))
    pickle.dump((contexts, responses), open("ed_data", "wb"))

    return contexts, responses
