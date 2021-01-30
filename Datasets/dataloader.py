import Datasets.wizard_of_wikipedia as wow
import Datasets.empathetic_dialogues as ed
import Datasets.persona_chat as pc
import requests
import tarfile
import os
import sys
import pickle


def load_dataset(keys=("ed", "wow", "pc")):
    """
    Loads the dataset specified by a list of keys, and returns tuple of contexts and responses
    """
    all_contexts = []
    all_responses = []
    key2dataset = {"wow": wow, "ed": ed, "pc": pc}
    key2url = {"wow": "http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz",
               "ed": "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz",
               "pc": "http://parl.ai/downloads/personachat/personachat.tgz"}

    for key in keys:
        # Check key exists
        if key in key2dataset:
            try:
                new_path = os.path.join("Datasets", key)
                os.mkdir(new_path)
                url = key2url[key]
                print(f"Attempting to download dataset: {url}")
                request = requests.get(url, stream=True)
                if request.status_code != requests.codes.ok:
                    print(f"Connection error: {request.status_code}")
                    sys.exit()
                else:
                    save_path = f"Datasets/{key}.tar.gz"
                    with open(save_path, "wb") as fd:
                        for chunk in request.iter_content(chunk_size=128):
                            fd.write(chunk)

                    # Un-tar file
                    tar = tarfile.open(save_path)
                    tar.extractall(new_path)
                    tar.close()

                    print(f"Successfully downloaded and un-tarred: {url}")

            except FileExistsError:
                # Thrown if file already downloaded
                pass

            contexts, responses = key2dataset[key].load_dataset()
            all_contexts += contexts
            all_responses += responses

        else:
            print(f"key for {key} not recognised, so will not be included in dataset")

    # Save context-response pairs to quicker use in inference
    pickle.dump(all_responses, open("utterances.pickle", "wb"))

    return all_contexts, all_responses
