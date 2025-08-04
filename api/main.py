from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
import requests, tempfile, os, json, hashlib
import fitz  # PyMuPDF
from pinecone import Pinecone
import google.generativeai as genai
from tqdm import tqdm
import time
# === API Keys ===
INDEX_NAME = "hackrx-rag-llama"
GOOGLE_API_KEY = "AIzaSyBXHgQmUsJEbcEZIMZQ41z1SLsGdCBKXQg"
PINECONE_API_KEY = "pcsk_6syiPH_E34w5TX3cHn74we6fjV41qiiig5McBQxFQ2J1Yo8sMB1JfP6KAKxuNYvd8te495"

# === Pinecone Init ===
pc = Pinecone(api_key=PINECONE_API_KEY)
if not pc.has_index(INDEX_NAME):
    pc.create_index_for_model(
        name=INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "llama-text-embed-v2",
            "field_map": {"text": "chunk_text"}
        }
    )
index = pc.Index(INDEX_NAME)

# === FastAPI App ===
app = FastAPI()

# === Schemas ===
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# === Helpers ===
def get_pdf_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()

def extract_text(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def upsert_in_batches(index, records, namespace="ragtest", batch_size=96):
    for i in tqdm(range(0, len(records), batch_size), desc="Upserting to Pinecone"):
        batch = records[i:i + batch_size]
        if i % 5 == 0:
           time.sleep(5)
        index.upsert_records(namespace=namespace, records=batch)

# === Main Endpoint ===
@app.post("/hackrx/run", response_model=QueryResponse)
def run(body: QueryRequest, authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    doc_id = get_pdf_hash(body.documents)

    try:
        # Step 1: Download PDF
        res = requests.get(body.documents, timeout=30)
        res.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(res.content)
            pdf_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF download error: {str(e)}")

    try:
        # Step 2: Extract and Chunk Text
        text = extract_text(pdf_path)
        chunk_size = 1000
        overlap = 100

        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

        # Step 3: Prepare records with chunk_text (auto-embedded by Pinecone)
        records = [
            {
                "id": f"{doc_id}-{i}",
                "chunk_text": chunk
            }
            for i, chunk in enumerate(chunks)
        ]
        upsert_in_batches(index, records, namespace="hackrx")

        # Step 4: Query Pinecone per-question and ask Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        all_answers = []

        for question in body.questions:
            results = index.search(
                namespace="hackrx",
                query={
                    "top_k": 10,
                    "inputs": {
                        "text": question
                    }
                },
                rerank={
                    "model": "bge-reranker-v2-m3",
                    "top_n": 5,
                    "rank_fields": ["chunk_text"]
                }
            )
            top_chunks = [hit["fields"]["chunk_text"] for hit in results["result"]["hits"]]



            context = "\n\n".join(top_chunks)
            prompt = f"""
You are a helpful assistant.
Use ONLY the following document context to answer the question.



Context:
\"\"\"{context}\"\"\"

Question: {question}

Answer the question clearly in 1-2 sentences.
"""

            try:
                response = model.generate_content(prompt)
                answer = response.text.strip()
                if answer.startswith("```"):
                    answer = answer.replace("```json", "").replace("```", "").strip()
                all_answers.append(answer)
            except Exception as e:
                all_answers.append(f"Gemini Error: {str(e)}")

    except Exception as e:
        all_answers = [f"Gemini Error: {str(e)}"]
    finally:
        try:
            os.remove(pdf_path)
        except:
            pass

    return {"answers": all_answers}

