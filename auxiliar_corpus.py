import json
import os
from datasets import load_dataset


def prepare_corpus():
    print("Extreient 10.000 frases de BookCorpusOpen...")

    HF_TOKEN = "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"

    dataset = load_dataset(
        "lucadiliello/bookcorpusopen", #bookcorpusopen, sinó en donava problemes
        split="train",
        streaming=True,
        token=HF_TOKEN,
    )

    sentences = []

    for item in dataset:
        full_text = item.get("text", "")
        # Dividim el text del llibre en línies/frases individuals
        lines = full_text.split("\n")

        for line in lines:
            clean_line = line.strip()
            # Filtrem línies buides o massa curtes (com títols o números de pàgina)
            if len(clean_line) > 25:
                sentences.append(clean_line)

            if len(sentences) >= 10000:
                break

        if len(sentences) >= 10000:
            break

    os.makedirs("data", exist_ok=True)

    # Sobreescribim l'arxiu anterior i sinó el crea
    with open("data/corpus_10k.json", "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)

    print(
        f"Completat! S'han desat {len(sentences)} frases a 'data/corpus_10k.json'."
    )


if __name__ == "__main__":
    prepare_corpus()