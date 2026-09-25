import time
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer

# Configuració de la base de dades
DB_CONFIG = {
    "dbname": "cbde_lab1",
    "user": "postgres",
    "password": "postgresjordi", 
    "host": "localhost",
    "port": "5432",
}


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Corregit: ADD COLUMN IF NOT EXISTS
    cur.execute(
        "ALTER TABLE sentences_pg ADD COLUMN IF NOT EXISTS embedding FLOAT[];"
    )
    conn.commit()

    print("Recuperant les 10.000 frases de PostgreSQL...")
    cur.execute("SELECT id, text FROM sentences_pg ORDER BY id;")
    rows = cur.fetchall()

    print("Carregant el model de Transformer ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generant embeddings i mesurant l'emmagatzematge a la BD...")
    storage_times = []

    for row_id, text in rows:
        # Generació de l'vector (aïllat del cronòmetre de la BD)
        vector = model.encode(text).tolist()

        # Mesurem ÚNICAMENT el temps d'escriptura (UPDATE + COMMIT) a PostgreSQL
        t0 = time.perf_counter()
        cur.execute(
            "UPDATE sentences_pg SET embedding = %s WHERE id = %s;",
            (vector, row_id),
        )
        conn.commit()
        t1 = time.perf_counter()

        storage_times.append(t1 - t0)

    cur.close()
    conn.close()

    # Mètriques requerides
    print("\n=== [P1] RESULTATS D'EMMAGATZEMATGE D'EMBEDDINGS (SQL) ===")
    print(
        f"Mínim:              {np.min(storage_times)*1000:.4f} ms ({np.min(storage_times):.6f} s)"
    )
    print(
        f"Màxim:              {np.max(storage_times)*1000:.4f} ms ({np.max(storage_times):.6f} s)"
    )
    print(
        f"Mitjana:            {np.mean(storage_times)*1000:.4f} ms ({np.mean(storage_times):.6f} s)"
    )
    print(
        f"Desviació Estàndard: {np.std(storage_times)*1000:.4f} ms ({np.std(storage_times):.6f} s)"
    )


if __name__ == "__main__":
    main()