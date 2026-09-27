import json
import os
import re

import nltk
from datasets import load_dataset

#Intantem utilitzar el NLTK per a dividir el text en frases. Si no està disponible, utilitzarem una alternativa més simple.
def get_sentence_splitter():
    try:
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        nltk.sent_tokenize("Test. Test.")
        return nltk.sent_tokenize

    except Exception:
        return lambda t: re.split(r"(?<=[.!?])\s+", t) #regex simple


def prepare_corpus():
    print("Extreient 10.000 frases de BookCorpusOpen...")

    dataset = load_dataset(
        "lucadiliello/bookcorpusopen",
        split="train",
        streaming=True,
        token="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",
    )

    split = get_sentence_splitter()

    sentences = []

    for item in dataset:
        full_text = item.get("text", "")

        # Dividim el text realment per frases
        extracted_sentences = split(full_text)

        for sentence in extracted_sentences:
            sentence = " ".join(sentence.split())

            if sentence:
                sentences.append(sentence)

            if len(sentences) >= 10000:
                break

        if len(sentences) >= 10000:
            break

    os.makedirs("data", exist_ok=True)

    with open("data/corpus_10k.json", "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)

    print(
        f"Completat! S'han desat {len(sentences)} frases "
        "a 'data/corpus_10k.json'."
    )


if __name__ == "__main__":
    prepare_corpus()