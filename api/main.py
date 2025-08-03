from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import tempfile
import os
import asyncio
import time

from agno.agent import Agent
from agno.models.google import Gemini
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.pineconedb import PineconeDb
from agno.embedder.google import GeminiEmbedder

# === FastAPI App Initialization ===
app = FastAPI()

# === API Keys ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # flash
GOOGLE_API_KEY_2 = os.getenv("GOOGLE_API_KEY_2")  # embedder
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# === Rate Limiting ===
last_request_time = 0
REQUEST_INTERVAL = 12  # 12 seconds between requests (5 requests/minute)

# === Pinecone Setup ===
PINECONE_INDEX_NAME = "ragtest-0308v1"
vector_db = PineconeDb(
    name=PINECONE_INDEX_NAME,
    dimension=1536,
    metric="cosine",
    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
    api_key=PINECONE_API_KEY,
    embedder=GeminiEmbedder(api_key=GOOGLE_API_KEY_2),
    use_hybrid_search=True,
    hybrid_alpha=0.5,
)

# === Pydantic Models ===
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# === Rate Limited Helper Function ===
async def get_answer_with_rate_limit(agent, question: str, retries: int = 3, timeout: int = 60):
    global last_request_time
    
    for attempt in range(retries):
        try:
            current_time = time.time()
            time_since_last = current_time - last_request_time
            if time_since_last < REQUEST_INTERVAL:
                wait_time = REQUEST_INTERVAL - time_since_last
                await asyncio.sleep(wait_time)
            
            last_request_time = time.time()

            response = await asyncio.wait_for(
                asyncio.to_thread(agent.run, question),
                timeout=timeout
            )
            return response

        except asyncio.TimeoutError:
            if attempt < retries - 1:
                await asyncio.sleep(5)
            else:
                return "Error: Request timed out after 60 seconds"

        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err or "429" in err or "503" in err:
                if attempt < retries - 1:
                    wait_time = 30 * (attempt + 1)
                    await asyncio.sleep(wait_time)
                else:
                    return "Error: Gemini rate limit exceeded. Please try again later."
            else:
                return f"Error: {err}"

    return "Error: All retry attempts failed"

# === POST Endpoint ===
@app.post("/hackrx/run", response_model=QueryResponse)
async def ask_document_questions(
    body: QueryRequest,
    authorization: str = Header(...)
):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    try:
        response = requests.get(body.documents, timeout=30)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(response.content)
            pdf_path = temp_pdf.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading PDF: {str(e)}")

    try:
        knowledge_base = PDFKnowledgeBase(path=pdf_path, vector_db=vector_db)
        knowledge_base.load(recreate=False, upsert=True)  # Only embed once

        agent = Agent(
            system_message="You are a helpful assistant that answers questions based on retrieved documents." \
            " Keep responses concise and relevant. Answer in one or two sentences only. " \
            " If there are any quantifiable data available to support your answer, include the same. "\
            "The answer has to be accurate and found in the retrieved content.",
            model=Gemini(id="gemini-2.5-pro", api_key=GOOGLE_API_KEY),
            knowledge=knowledge_base,
            show_tool_calls=False,
            markdown=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector DB error: {str(e)}")

    # === PARALLEL QUESTION HANDLING ===
    async def ask_question(question):
        try:
            print(f"Processing: {question[:50]}...")
            response = await get_answer_with_rate_limit(agent, question)
            if isinstance(response, str):
                return response
            else:
                return response.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    answers = await asyncio.gather(*(ask_question(q) for q in body.questions))

    try:
        os.remove(pdf_path)
    except Exception:
        pass

    return {"answers": answers}


