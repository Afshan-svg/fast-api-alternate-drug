import json
import psycopg2
from embeddings import generate_embeddings_batch
import time

INPUT_FILE = "cleaned.json"
BATCH_SIZE = 50   # you can increase to 100


def parse_dob(dob_str):
    """Convert DD-MM-YYYY → YYYY-MM-DD."""
    if not dob_str:
        return None
    try:
        d, m, y = dob_str.split("-")
        return f"{y}-{m}-{d}"
    except:
        return None


def build_text(record):
    """Convert record to a single text string for embedding."""
    symptoms = ", ".join(record.get("symptoms", []))
    causes = ", ".join(record.get("causes", []))
    disease = record.get("disease", "")
    medicine = ", ".join(record.get("medicine", []))

    return (
        f"Disease: {disease}. "
        f"Symptoms: {symptoms}. "
        f"Causes: {causes}. "
        f"Medicine: {medicine}. "
        f"Gender: {record.get('gender', '')}."
    )


def main():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    print("Loading cleaned.json...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    total = len(records)
    inserted = 0

    print(f"Total records to insert: {total}")

    for i in range(0, total, BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]

        # 1. Create embedding text for batch
        texts = [build_text(rec) for rec in batch]

        # 2. Generate embeddings in batch
        embeddings = generate_embeddings_batch(texts)

        # 3. Insert each record
        for rec, emb in zip(batch, embeddings):
            dob = parse_dob(rec.get("date_of_birth"))
            gender = rec.get("gender")

            cur.execute(
                """
                INSERT INTO case_records (date_of_birth, gender, data, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (dob, gender, json.dumps(rec), emb)
            )
            inserted += 1

        conn.commit()
        print(f"Inserted {inserted}/{total}")
        time.sleep(0.1)  # cool down

    conn.close()
    print("DONE INGESTION 🎉")


if __name__ == "__main__":
    main()
