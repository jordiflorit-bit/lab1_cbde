import json
import time

import numpy as np
import psycopg2


DB_CONFIG = {
    "dbname": "cbde_lab1",
    "user": "postgres",
    "password": "postgresjordi",
    "host": "localhost",
    "port": "5432",
}

JSON_PATH = "data/corpus_10k.json"


def main():

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        sentences = json.load(f)

    print(f"{len(sentences)} frases carregades")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor() #objecte de control per comandes SQL

    cur.execute("DROP TABLE IF EXISTS sentences_pg;")

    cur.execute("""
        CREATE TABLE sentences_pg (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL
        );
    """)

    conn.commit()

    insertion_times = []

    print("Inserint text a PostgreSQL...")

    for sentence in sentences:

        t0 = time.perf_counter()

        cur.execute(
            "INSERT INTO sentences_pg (text) VALUES (%s);",
            (sentence,),
        )

        conn.commit()

        insertion_times.append(
            time.perf_counter() - t0
        )

    cur.close()
    conn.close()

    t = np.array(insertion_times)

    stats = {
        "n": int(t.size),
        "min": float(t.min()),
        "max": float(t.max()),
        "avg": float(t.mean()),
        "std": float(t.std(ddof=1)),
    }

    print("\n--- [P0] TEMPS D'INSERCIO DE TEXT, per frase ---")
    print(f"Minim:     {stats['min']:.6f} s")
    print(f"Maxim:     {stats['max']:.6f} s")
    print(f"Mitjana:   {stats['avg']:.6f} s")
    print(f"Desv. est: {stats['std']:.6f} s")


if __name__ == "__main__":
    main()