# chatbot

This is a retrieval chatbot designed to engage in social interaction with the user.  It currently uses the well-established strategy of an initial dot product-based similarity measurement between the user input and a large set of candidate responses, followed by a more granular reranker which applies attention between each pair across the top-k candidates to determine which one to reveal to the user.

The entire flow of the code can be operated via the main.py file, with a variable "pipeline" determining which step of the pipeline is called, whether preprocessing, training or inference.

The model currently is not multi-turn in its context window, although that could be altered relatively easily simply by changing the Datasets files slightly.  The 2-step retrieve and rerank process also would allow alternative retrieval strategies such as deep context-context, TfIDf, pattern-matching, and generative models to all be used and then evaluated on equal footing by the reranker.

Dependencies:

spacy == 2.3.5

beautifulsoup4 == 4.9.3

faiss-gpu == 1.6.5

nltk == 3.5

numpy == 1.18.5

pandas == 1.1.5

tensorflow == 2.3.1

tensorflow-hub == 0.10.0

transformers == 4.0.1

sentencepiece == 0.1.94

convokit == 2.4.3

Expect many of these dependencies will also work on slightly earlier versions, though I haven't checked, so have provided the versions I used.
