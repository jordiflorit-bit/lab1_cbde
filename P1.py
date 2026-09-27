import time
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer


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

    # Afegim la columna per guardar els embeddings
    cur.execute(
        "ALTER TABLE sentences_pg ADD COLUMN IF NOT EXISTS embedding FLOAT[];"
    )
    conn.commit()

    # Eliminem embeddings anteriors en cas de tornar a executar P1
    cur.execute("UPDATE sentences_pg SET embedding = NULL;")
    conn.commit()

    # Recuperem les frases de PostgreSQL
    print("Recuperant les 10.000 frases de PostgreSQL...")
    cur.execute("SELECT id, text FROM sentences_pg ORDER BY id;")
    rows = cur.fetchall()
    

    print("Generant embeddings i mesurant l'emmagatzematge a la BD...")

    storage_times = []
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for row_id, text in rows:

        # Generació de l'embedding
        vector = model.encode(text).tolist()

        # Mesurem només UPDATE + COMMIT
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

    # Estadístiques
    minimum = np.min(storage_times)
    maximum = np.max(storage_times)
    average = np.mean(storage_times)
    std = np.std(storage_times, ddof=1)

    print("\n--- [P1] RESULTATS D'EMMAGATZEMATGE D'EMBEDDINGS (SQL) ---")
    print(f"Mínim: {minimum*1000:.4f} ms ({minimum:.6f} s)")
    print(f"Màxim: {maximum*1000:.4f} ms ({maximum:.6f} s)")
    print(f"Mitjana: {average*1000:.4f} ms ({average:.6f} s)")
    print(f"Desviació Estàndard: {std*1000:.4f} ms ({std:.6f} s)")



if __name__ == "__main__":
    main()