import json
import time
import shutil
import os

import numpy as np
import chromadb


JSON_PATH = "data/corpus_10k.json"
CHROMA_PATH = "chroma_db_c0"
COLLECTION_NAME = "sentences_text"


def main():

    # Carregar les mateixes 10.000 frases que P0
    print("Carregant les frases del corpus...")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        sentences = json.load(f)

    print(f"{len(sentences)} frases carregades")


    # Netejar la BD de C0 si ja existia
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    print("Creant la base de dades Chroma...")

    client = chromadb.PersistentClient(path=CHROMA_PATH)


    # Crear collection
    collection = client.create_collection(
        name=COLLECTION_NAME
    )


    # Inserir frases i mesurar temps
    print("Inserint text a Chroma...")

    insertion_times = []

    for i, sentence in enumerate(sentences):

        # Chroma necessita un embedding associat a l'element.
        # Utilitzem un vector fictici perquè C0 NO generi
        # els embeddings reals.
        embedding_fictici = [0.0]

        t0 = time.perf_counter()

        collection.add(
            ids=[str(i + 1)],
            documents=[sentence],
            embeddings=[embedding_fictici]
        )

        t1 = time.perf_counter()

        insertion_times.append(t1 - t0)

    print(f"\nElements emmagatzemats a Chroma: {collection.count()}")


    # Estadístiques
    minimum = np.min(insertion_times)
    maximum = np.max(insertion_times)
    average = np.mean(insertion_times)
    std = np.std(insertion_times, ddof=1)

    print("\n--- [C0] RESULTATS D'INSERCIÓ DE TEXT (CHROMA) ---")

    print(f"Mínim: {minimum*1000:.4f} ms ({minimum:.6f} s)")
    print(f"Màxim: {maximum*1000:.4f} ms ({maximum:.6f} s)")
    print(f"Mitjana: {average*1000:.4f} ms ({average:.6f} s)")
    print(f"Desviació Estàndard: {std*1000:.4f} ms ({std:.6f} s)")


if __name__ == "__main__":
    main()