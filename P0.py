import json
import re
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

JSON_PATH = "data/bookCorpus_raw.json"
MAX_SENTENCES = 10_000


def get_sentence_splitter():
    """NLTK (punkt) si esta disponible; si no, una regex simple."""
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        nltk.sent_tokenize("Test. Test.")
        return nltk.sent_tokenize
    except Exception:
        return lambda t: re.split(r"(?<=[.!?])\s+", t)


def chunk_text(item):
    """Cada element del JSON pot ser un string o un dict amb la clau 'text'."""
    return item["text"] if isinstance(item, dict) else item


def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        chunks = [chunk_text(x) for x in json.load(f)]

    split = get_sentence_splitter()

    # Dividim en frases ABANS de connectar-nos: no forma part del temps de la BD
    rows = []  # (chunk_id, frase)
    for chunk_id, chunk in enumerate(chunks):
        for s in split(chunk):
            s = s.strip()
            if s:
                rows.append((chunk_id, s))
                if len(rows) >= MAX_SENTENCES:
                    break
        if len(rows) >= MAX_SENTENCES:
            break
    print(f"{len(rows)} frases carregades (limit {MAX_SENTENCES}), "
          f"provinents de {rows[-1][0] + 1} chunks")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS sentences_pg;")
    cur.execute("""
        CREATE TABLE sentences_pg (
            id       SERIAL PRIMARY KEY,
            chunk_id INTEGER NOT NULL,
            text     TEXT NOT NULL
        );
    """)
    conn.commit()

    insertion_times = []
    print("Inserint text a PostgreSQL...")
    for chunk_id, sentence in rows:
        t0 = time.perf_counter()
        cur.execute(
            "INSERT INTO sentences_pg (chunk_id, text) VALUES (%s, %s);",
            (chunk_id, sentence),
        )
        conn.commit()
        insertion_times.append(time.perf_counter() - t0)

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
    print("--- TEMPS D'INSERCIO DE TEXT (P0), per frase ---")
    print(f"Minim:     {stats['min']:.6f} s")
    print(f"Maxim:     {stats['max']:.6f} s")
    print(f"Mitjana:   {stats['avg']:.6f} s")
    print(f"Desv. est: {stats['std']:.6f} s")

    with open("p0_times.json", "w") as f:
        json.dump({"stats": stats, "times": insertion_times}, f)


if __name__ == "__main__":
    main()