import time
import numpy as np
import psycopg2

# Configuració de la base de dades
DB_CONFIG = {
    "dbname": "cbde_lab1",
    "user": "postgres",
    "password": "postgresjordi",
    "host": "localhost",
    "port": "5432",
}


def euclidean_distance(a, b):
    """Calcula la distància Euclidiana (L2) entre dos vectors."""
    return np.linalg.norm(a - b)


def cosine_distance(a, b):
    """Calcula la distància de Cosinus (1 - Cosine Similarity) entre dos vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return 1.0 - (dot / (norm_a * norm_b))


def main():
    print("Connectant a PostgreSQL per recuperar els embeddings...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute(
        "SELECT id, text, embedding FROM sentences_pg WHERE embedding IS NOT NULL ORDER BY id;"
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    embeddings = np.array([r[2] for r in rows])

    print(
        f"S'han carregat {len(embeddings)} vectors de la base de dades correctament."
    )

    # Seleccionem 10 frases de mostra espaiades clarament al llarg del conjunt
    target_indices = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]

    print("\n=== [P2] 10 FRASES D'OBJECTIU SELECCIONADES ===")
    for idx in target_indices:
        print(f"ID {ids[idx]}: '{texts[idx]}'")

    euc_times = []
    cos_times = []

    print(
        "\nCalculant el Top-2 de similaritat per a les 10 frases i mesurant temps..."
    )

    for idx in target_indices:
        target_emb = embeddings[idx]

        # 1. Distància Euclidiana
        t0 = time.perf_counter()
        dists_euc = [
            euclidean_distance(target_emb, emb) if i != idx else float("inf")
            for i, emb in enumerate(embeddings)
        ]
        top2_euc_idx = np.argsort(dists_euc)[:2]
        t1 = time.perf_counter()
        euc_times.append(t1 - t0)

        # 2. Distància de Cosinus
        t0 = time.perf_counter()
        dists_cos = [
            cosine_distance(target_emb, emb) if i != idx else float("inf")
            for i, emb in enumerate(embeddings)
        ]
        top2_cos_idx = np.argsort(dists_cos)[:2]
        t1 = time.perf_counter()
        cos_times.append(t1 - t0)

    # Mètriques requerides
    print("\n=== [P2] RESULTATS DE TEMPS DE CERCA DE SIMILARITAT ===")
    print("Mètrica 1: Distància Euclidiana (L2)")
    print(f"  Mínim:              {np.min(euc_times)*1000:.4f} ms ({np.min(euc_times):.6f} s)")
    print(f"  Màxim:              {np.max(euc_times)*1000:.4f} ms ({np.max(euc_times):.6f} s)")
    print(f"  Mitjana:            {np.mean(euc_times)*1000:.4f} ms ({np.mean(euc_times):.6f} s)")
    print(f"  Desviació Estàndard: {np.std(euc_times)*1000:.4f} ms ({np.std(euc_times):.6f} s)")
    print("\nMètrica 2: Distància de Cosinus")
    print(f"  Mínim:              {np.min(cos_times)*1000:.4f} ms ({np.min(cos_times):.6f} s)")
    print(f"  Màxim:              {np.max(cos_times)*1000:.4f} ms ({np.max(cos_times):.6f} s)")
    print(f"  Mitjana:            {np.mean(cos_times)*1000:.4f} ms ({np.mean(cos_times):.6f} s)")
    print(f"  Desviació Estàndard: {np.std(cos_times)*1000:.4f} ms ({np.std(cos_times):.6f} s)")


if __name__ == "__main__":
    main()