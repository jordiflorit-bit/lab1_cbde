import json
import os
from datasets import load_dataset


def prepare_raw_corpus():
    print("Descarregant fragments de llibres en brut des de BookCorpusOpen...")

    HF_TOKEN = "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"

    dataset = load_dataset(
        "lucadiliello/bookcorpusopen",
        split="train",
        streaming=True,
        token=HF_TOKEN,
    )

    raw_paragraphs = []

    # Recollim paràgrafs o blocs de text en brut fins a tenir volum suficient
    for item in dataset:
        text = item.get("text", "").strip()
        if text:
            # Afegim el bloc de text complet/paràgraf
            raw_paragraphs.append(text)

        # Amb un cert nombre de blocs tindrem més de 10.000 frases
        if len(raw_paragraphs) >= 500:
            break

    os.makedirs("data", exist_ok=True)

    with open("data/bookCorpus_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_paragraphs, f, ensure_ascii=False, indent=2)

    print(
        f"Completat! S'han desat {len(raw_paragraphs)} blocs de text brut a 'data/bookCorpus_raw.json'."
    )


if __name__ == "__main__":
    prepare_raw_corpus()