import json
import time
import shutil
import os

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer


JSON_PATH = "data/corpus_10k.json"
CHROMA_PATH = "chroma_db"


def main():

    # Carregar les mateixes 10.000 frases
    print("Carregant les frases del corpus...")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        sentences = json.load(f)

    print(f"{len(sentences)} frases carregades")


    # Generar els embeddings
    print("Generant els embeddings...")

    embeddings = []
    model = SentenceTransformer("all-MiniLM-L6-v2") # Carregar el mateix model que a PostgreSQL

    for sentence in sentences:
        vector = model.encode(sentence).tolist()
        embeddings.append(vector)

    print(f"{len(embeddings)} embeddings generats")


    # Crear una BD Chroma nova
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    print("Creant la base de dades Chroma...")

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )


    # Crear les dues collections
    collection_l2 = client.create_collection(
        name="sentences_l2",
        configuration={
            "hnsw": {
                "space": "l2"
            }
        }
    )

    collection_cosine = client.create_collection(
        name="sentences_cosine",
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        }
    )


    # Inserir embeddings i mesurar temps
    print("Inserint embeddings a Chroma...")

    storage_times = []

    for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):

        item_id = str(i + 1)

        t0 = time.perf_counter()

        collection_l2.add(          #omplim collection L2
            ids=[item_id],
            documents=[sentence],
            embeddings=[embedding]
        )

        t1 = time.perf_counter()

        storage_times.append(t1 - t0)


    # Omplir també la collection Cosine
    # Ho fem per lots perquè aquesta segona còpia no forma part de la mesura de C1.
    BATCH_SIZE = 500

    for start in range(0, len(sentences), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(sentences))

        ids_batch = [
            str(i + 1)
            for i in range(start, end)
        ]

        collection_cosine.add(
            ids=ids_batch,
            documents=sentences[start:end],
            embeddings=embeddings[start:end]
        )


    # Comprovacions
    print(f"\nElements L2: {collection_l2.count()}")
    print(f"Elements Cosine: {collection_cosine.count()}")


    # Estadístiques
    minimum = np.min(storage_times)
    maximum = np.max(storage_times)
    average = np.mean(storage_times)
    std = np.std(storage_times, ddof=1)

    print("\n--- [C1] RESULTATS D'EMMAGATZEMATGE D'EMBEDDINGS (CHROMA) ---")

    print(f"Mínim: {minimum*1000:.4f} ms ({minimum:.6f} s)")
    print(f"Màxim: {maximum*1000:.4f} ms ({maximum:.6f} s)")
    print(f"Mitjana: {average*1000:.4f} ms ({average:.6f} s)")
    print(f"Desviació Estàndard: {std*1000:.4f} ms ({std:.6f} s)")


if __name__ == "__main__":
    main()