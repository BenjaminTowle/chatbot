import re


def load_dataset(return_metadata=False):
    """
    Load dataset from Persona Chat paper: https://arxiv.org/abs/1801.07243
    :return: list of contexts, list of responses, list of personae
    """
    with open("Datasets/pc/personachat/train_both_revised.txt") as f:
        persona_a = []
        personae_a = []
        persona_b = []
        personae_b = []
        dialogue = []
        dialogues = []
        reading_persona = True
        lines = f.readlines()
        for line in lines:
            if "your persona:" in line:
                if reading_persona is False:
                    personae_a.append(persona_a)
                    personae_b.append(persona_b)
                    dialogues.append(dialogue)
                    persona_a = []
                    persona_b = []
                    dialogue = []
                    reading_persona = True
                persona_a.append(re.sub(r"\A[0-9]+ (your persona: |partner's persona: )", "", line))
            elif "partner's persona:" in line:
                persona_b.append(re.sub(r"\A[0-9]+ (your persona: |partner's persona: )", "", line))
            else:
                for utt in line.split("\t")[:2]:
                    utt = re.sub(r"\A[0-9]+ ", "", utt)  # remove line numbering
                    dialogue.append(utt)
                    reading_persona = False

    contexts = []
    responses = []
    metadata = []
    for pa, pb, dialog in zip(personae_a, personae_b, dialogues):
        for i in range(1, len(dialog)):
            history = dialog[:i-1]
            persona = []
            # Select which persona belongs to the responder
            if i % 2 == 0:
                for p in pa:
                    persona.append(p)
            else:
                for p in pb:
                    persona.append(p)

            meta = persona + history
            metadata.append(meta)

            contexts.append(dialog[i-1])
            responses.append(dialog[i])
            
    if return_metadata:   
        return contexts, responses, metadata
    else:
        return contexts, responses
