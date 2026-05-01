from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import google.genai as genai
from rag import load_pdf, split_text, load_embeddings, store_data
import uvicorn
from dotenv import load_dotenv
import os, shutil

app = FastAPI(title="RAG API")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

embedding_model = load_embeddings()
vectorstore = None

UPLOAD_DIR = "upload"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

class Question(BaseModel):
    question: str
    k: int = 3

class Answer(BaseModel):
    answer: str
    sources: list[str]

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore
    file_name = os.path.basename(file.filename or "")

    if not file_name or not file_name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")
    
    if not file_name:
        raise HTTPException(status_code=400, detail="File must have a name")
    
    file_path = os.path.join(UPLOAD_DIR, file_name)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = load_pdf(file_path)
    chunks = split_text(text)
    vectorstore = store_data(chunks, embedding_model)

    return {"chunk indexed": len(chunks)}

@app.post("/query", response_model=Answer)
async def query(req: Question):
    if vectorstore is None:
        raise HTTPException(400, "Upload a PDF first")

    docs    = vectorstore.similarity_search(req.question, k=req.k)
    context = "".join(d.page_content for d in docs)
    sources = [d.page_content for d in docs]

    prompt = f"""Answer using ONLY this context.
If not found, say "I don't know".

Context: {context}

Question: {req.question}
Answer:"""

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=genai.types.Part.from_text(text=prompt),
        config=genai.types.GenerateContentConfig(
            temperature=0,
            top_p=0.95,
            top_k=20,
        ),
    )
    return Answer(answer=response.text or "", sources=sources)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed": vectorstore is not None
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)



    

    

    


