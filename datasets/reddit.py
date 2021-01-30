from convokit import Corpus
import requests
from bs4 import BeautifulSoup
import re
import zipfile
from nltk.tokenize import word_tokenize
import spacy
from spellchecker import SpellChecker
import chatbot_db as db
import nltk

nltk.download('punkt')

try:
    nlp = spacy.load("en_core_web_md")
except IOError:
    # Exception will be thrown if not downloaded
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")
    

################################
# Filtering hyper-parameters
################################

MIN_LENGTH = 5
MAX_LENGTH = 30
MIN_SCORE = 2
SUBREDDIT_MAX_SIZE = 50000000
SUBREDDIT_MIN_SIZE = 1000000

START = 5
END = 20

total_pairs = 0

# Load blacklist of profane words
profanity = []
with open("profanity.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        profanity.append(line.replace("\n", "").replace("\r", "").lower())


def get_subreddits():
    """
    Loads the Reddit corpus as provided by Cornell University.  The corpus contains over 3 billion comments, so requires
    to be downloaded and processed in batches.  The quality of also highly variable with high toxicity as times and
    also the conversations not exactly being structurally equivalent to a conversational dialogue.  As such several
    heuristic filters are applied which can be enabled or disabled as needed.
    :return:
    """
    url = "https://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/corpus-zipped/"
    request = requests.get(url)
    print("loaded main url ...")
    if request.status_code != requests.codes.ok:
        print(request.status_code)
    else:
        soup = BeautifulSoup(request.text, "html.parser")
        body = soup.find("pre")
        table = body.strings
        for i, (row, link) in enumerate(list(zip(table, body.find_all("a")))[1:]):
            print(i)
            if i >= START:
                # Search through only first 20
                if i == START+END:
                    break

                name = link.string
                print("loading: ", url+name)
                sub_request = requests.get(url + name)
                if request.status_code != requests.codes.ok:
                    print(request.status_code)
                else:
                    soup = BeautifulSoup(sub_request.text, "html.parser")
                    print("loaded sub-url ...")
                    sub_body = soup.find("pre")
                    sub_table = sub_body.strings
                    text = ""
                    for r in sub_table:
                        text += r
                    text = text.split("\r\n")

                    links = []
                    sizes = []
                    for t in text[1:-1]:
                        tup = re.split("  +", t)
                        links.append(tup[0])
                        sizes.append(int(tup[2]))

                    for link_, size in zip(links, sizes):
                        if SUBREDDIT_MIN_SIZE <= size <= SUBREDDIT_MAX_SIZE:
                            file_url = url + name + link_
                            print("Downloading: ", file_url, int(size))
                            file_request = requests.get(file_url, stream=True)
                            save_path = "/home/gilgamesh/.convokit/downloads/" + link_
                            with open(save_path, "wb") as fd:
                                for chunk in file_request.iter_content(chunk_size=128):
                                    fd.write(chunk)

                            # Unzip the file
                            fd = save_path.replace(".corpus.zip", "")
                            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                                zip_ref.extractall(fd)

                            e = load_dataset(fd)
                            if e == -1:
                                print("dataset: ", name, " not found")


def load_dataset(filename):
    contexts = []
    responses = []
    try:
        corpus = Corpus(filename=filename)
    except KeyError:
        return -1
    for convo in corpus.iter_conversations():
        utts = []
        for utt in convo.iter_utterances():
            utts.append(utt)

        contexts += utts[:-1]
        responses += utts[1:]

    # 1) Check lengths are acceptable
    print(filename)
    print(len(contexts))
    tmp_contexts = []
    tmp_responses = []
    for i in range(len(contexts)):
        if MIN_LENGTH <= len(contexts[i].text.split()) <= MAX_LENGTH and MIN_LENGTH <= len(responses[i].text.split()) <= MAX_LENGTH:
            tmp_contexts.append(contexts[i])
            tmp_responses.append(responses[i])
    contexts = tmp_contexts
    responses = tmp_responses

    # 2) Check for score of response
    tmp_contexts = []
    tmp_responses = []
    for i in range(len(contexts)):
        if responses[i].meta["score"] >= MIN_SCORE:
            tmp_contexts.append(contexts[i])
            tmp_responses.append(responses[i])
    contexts = tmp_contexts
    responses = tmp_responses

    # 3) Check for profanity in response
    tmp_contexts = []
    tmp_responses = []
    for i in range(len(contexts)):
        is_profane = False
        for word in profanity:
            if word.lower() in responses[i].text.lower():
                is_profane = True
        if not is_profane:
            tmp_contexts.append(contexts[i].text)
            tmp_responses.append(responses[i].text)
    contexts = tmp_contexts
    responses = tmp_responses

    # 4) filter incorrect spelling
    tmp_contexts = []
    tmp_responses = []
    spell = SpellChecker()
    for i in range(len(contexts)):

        if len(spell.unknown(word_tokenize(contexts[i]))) == 0 and len(spell.unknown(word_tokenize(responses[i]))) == 0:
            tmp_contexts.append(contexts[i])
            tmp_responses.append(responses[i])
    contexts = tmp_contexts
    responses = tmp_responses

    # 5) named-entity recognition (remove responses with named-entities and tag contexts with named-entities -
    # these are typically very context specific and not likely to be appropriate for a general retrieval chatbot)
    tmp_contexts = []
    tmp_responses = []
    for i, doc in enumerate(nlp.pipe(responses)):
        if len(doc.ents) == 0:
            tmp_contexts.append(contexts[i])
            tmp_responses.append(responses[i])
    contexts = tmp_contexts
    responses = tmp_responses

    BATCH_SIZE = 50000
    contexts = contexts[:BATCH_SIZE]
    responses = responses[:BATCH_SIZE]
    # Add new data to database
    values = db.preprocess(contexts, responses)

    # If preprocessing encountered an error, None is returned
    if values is not None:
        db.insert(values)
        print(len(contexts))
        global total_pairs
        total_pairs += len(contexts)
        print("Total pairs: ", total_pairs)
        print("========================")
    else:
        print("error encountered during preprocessing, so skipping this batch")


if __name__ == "__main__":
    get_subreddits()
