from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import json
import math
from db import conn, cursor
from embeddings import generate_embeddings_batch
from schemas import QueryRequest
from openai import OpenAI
from config import OPENAI_API_KEY
client_llm = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# ---- CORS ----
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # "https://your-frontend-domain.com",  # add your deployed frontend here
    # For quick local testing only, you can use "*" (not recommended for production)
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------

def fix_nan(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, list):
        return [fix_nan(x) for x in obj]
    if isinstance(obj, dict):
        return {k: fix_nan(v) for k, v in obj.items()}
    return obj


@app.post("/upload-json")
async def upload_json(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        medicines = json.loads(contents)

        batch_size = 100
        inserted = 0

        for i in range(0, len(medicines), batch_size):
            batch = medicines[i:i + batch_size]

            # fix NaN
            batch = [fix_nan(m) for m in batch]

            # prepare text for embedding
            text_batch = [json.dumps(m) for m in batch]

            # EMBEDDING BATCH
            embeddings = generate_embeddings_batch(text_batch)

            # insert each row in batch
            for med, emb in zip(batch, embeddings):
                name = med.get("name")
                cursor.execute(
                    """
                    INSERT INTO medicines_dataset (name, data, embedding)
                    VALUES (%s, %s, %s)
                    """,
                    (name, json.dumps(med), emb)
                )
                inserted += 1

            conn.commit()  # commit after every 100 inserts

            print(f"Inserted {inserted}/{len(medicines)}")

        return {"status": "success", "records_inserted": inserted}

    except Exception as e:
        conn.rollback()
        return {"error": str(e)}


@app.get("/similar")
async def similar(q: str):
    try:
        query_emb = generate_embedding({"query": q})

        cursor.execute("""
            SELECT id, name, data
            FROM medicines_dataset
            ORDER BY embedding <-> %s
            LIMIT 5
        """, (query_emb,))

        rows = cursor.fetchall()

        results = [
            {"id": row[0], "name": row[1], "data": row[2]}
            for row in rows
        ]

        return {"results": results}

    except Exception as e:
        return {"error": str(e)}


@app.get("/alternates")
async def alternates(q: str):
    try:
        # 1. Create embedding for query
        query = {"query": q}
        text = json.dumps(query)
        emb = generate_embeddings_batch([text])[0]  # batch with 1 item

        # 2. Vector search
        cursor.execute("""
            SELECT name, data
            FROM medicines_dataset
            ORDER BY embedding <-> %s::vector
            LIMIT 5;
        """, (emb,))


        rows = cursor.fetchall()

        results = []
        for row in rows:
            med_name = row[0]
            med_data = row[1]  # full JSON from DB

            results.append({
                "name": med_name,
                "uses": med_data.get("uses", []),
                "substitutes": med_data.get("substitutes", []),
                "therapeuticClass": med_data.get("therapeuticClass"),
                "chemicalClass": med_data.get("chemicalClass"),
                "habitForming": med_data.get("habitForming"),
                "actionClass": med_data.get("actionClass")
            })

        return {"alternates": results}

    except Exception as e:
        return {"error": str(e)}

@app.post("/ask")
async def ask_question(body: QueryRequest):
    try:
        query_text = body.query
        k = body.k

        # 1. Embed the query
        emb = generate_embeddings_batch([query_text])[0]

        # 2. Retrieve nearest medicines
        cursor.execute("""
            SELECT name, data
            FROM medicines_dataset
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """, (emb, k))

        rows = cursor.fetchall()

        # 3. Build RAG context for OpenAI
        context = ""
        for name, data in rows:
            context += f"Medicine: {name}\n"
            context += f"Uses: {data.get('uses', [])}\n"
            context += f"Substitutes: {data.get('substitutes', [])}\n"
            context += "\n"

        print("\n================ RAG CONTEXT SENT TO OPENAI ================\n")
        print(context)
        print("\n============================================================\n")

        # 4. Call OpenAI to generate final answer
        prompt = f"""
        You are an assistant that MUST answer ONLY using the provided data.
        You are NOT allowed to use medical knowledge outside this context.

        Here is the user query:
        {query_text}

        Here is the ONLY data you can use:
        {context}

        Now answer the user's question strictly based on this data.
        """

        response = client_llm.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content

        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}
