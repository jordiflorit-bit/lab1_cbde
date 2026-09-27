import time
import numpy as np
import chromadb


CHROMA_PATH = "chroma_db"


def main():

    # Connectar amb la BD creada per C1
    print("Connectant a Chroma...")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection_l2 = client.get_collection(name="sentences_l2")

    collection_cosine = client.get_collection(name="sentences_cosine")


    # Mateixes 10 frases que P2
    target_ids = ["1", "1001", "2001", "3001", "4001", "5001", "6001", "7001", "8001", "9001"]

    targets = collection_l2.get(
        ids=target_ids,
        include=["documents", "embeddings"]
    )

    target_data = {}

    for i, item_id in enumerate(targets["ids"]):

        target_data[item_id] = {
            "document": targets["documents"][i],
            "embedding": targets["embeddings"][i]
        }


    print("\n--- 10 FRASES D'OBJECTIU SELECCIONADES ---")

    for item_id in target_ids:
        print(f"ID {item_id}: '{target_data[item_id]['document']}'")



    l2_times = []
    cosine_times = []


    print("\nCalculant el Top-2 de similaritat per a les 10 frases...")

    # Fer les consultes
    for item_id in target_ids:

        target_document = target_data[item_id]["document"]
        target_embedding = target_data[item_id]["embedding"]


        # Euclidiana o l2
        t0 = time.perf_counter()

        result_l2 = collection_l2.query(
            query_embeddings=[target_embedding],
            n_results=3,
            include=["documents", "distances"]
        )

        t1 = time.perf_counter()

        l2_times.append(t1 - t0)


        # Cosinus
        t0 = time.perf_counter()

        result_cosine = collection_cosine.query(
            query_embeddings=[target_embedding],
            n_results=3,
            include=["documents", "distances"]
        )

        t1 = time.perf_counter()

        cosine_times.append(t1 - t0)


        # Eliminar la frase objectiu dels resultats (perquè no es compti a si mateixa)
        l2_results = []

        for result_id, document, distance in zip(
            result_l2["ids"][0],
            result_l2["documents"][0],
            result_l2["distances"][0]
        ):
            if result_id != item_id:
                l2_results.append(
                    (result_id, document, distance)
                )

        l2_results = l2_results[:2] #Ens quedem els dos resultats amb menor distància euclidiana


        cosine_results = []

        for result_id, document, distance in zip(
            result_cosine["ids"][0],
            result_cosine["documents"][0],
            result_cosine["distances"][0]
        ):
            if result_id != item_id:
                cosine_results.append(
                    (result_id, document, distance)
                )

        cosine_results = cosine_results[:2] #Ens quedem els dos resultats amb menor distància de cosinus


        # MOSTRAR RESULTATS
        print(f"\n--- Frase objectiu ID {item_id} ---")

        print(f"'{target_document}'")

        print("\nTop-2 L2:")

        for result_id, document, distance in l2_results:
            print(f"  ID {result_id} (distance={distance:.6f}): '{document}'")

        print("\nTop-2 Cosine:")

        for result_id, document, distance in cosine_results:
            print(f"  ID {result_id} (distance={distance:.6f}): '{document}'")


    # -------------------------------------------------
    # 6. Estadístiques
    # -------------------------------------------------

    print("\n--- [C2] RESULTATS DE TEMPS DE CERCA DE SIMILARITAT ---")


    print("\nMètrica 1: Distància Euclidiana (L2)")

    print(f"Mínim: {np.min(l2_times)*1000:.4f} ms ({np.min(l2_times):.6f} s)")
    print(f"Màxim: {np.max(l2_times)*1000:.4f} ms ({np.max(l2_times):.6f} s)")
    print(f"Mitjana: {np.mean(l2_times)*1000:.4f} ms ({np.mean(l2_times):.6f} s)")
    print(f"Desviació Estàndard: {np.std(l2_times, ddof=1)*1000:.4f} ms ({np.std(l2_times, ddof=1):.6f} s)")


    print("\nMètrica 2: Distància de Cosinus")

    print(f"Mínim: {np.min(cosine_times)*1000:.4f} ms ({np.min(cosine_times):.6f} s)")
    print(f"Màxim: {np.max(cosine_times)*1000:.4f} ms ({np.max(cosine_times):.6f} s)")
    print(f"Mitjana: {np.mean(cosine_times)*1000:.4f} ms ({np.mean(cosine_times):.6f} s)")
    print(f"Desviació Estàndard: {np.std(cosine_times, ddof=1)*1000:.4f} ms ({np.std(cosine_times, ddof=1):.6f} s)")


if __name__ == "__main__":
    main()