# PDF RAG API

A Retrieval-Augmented Generation (RAG) API that lets you upload a PDF and ask questions about it. Built with FastAPI, ChromaDB, HuggingFace embeddings, and Google Gemini.

---

## How it works

```
PDF upload → Text extraction → Chunking → Embedding → ChromaDB
                                                           ↓
Question → Embedding → Similarity search → Top-k chunks → Gemini → Answer
```

1. **Upload** a PDF — the text is extracted, split into 500-character chunks, embedded with `sentence-transformers/all-MiniLM-L6-v2`, and stored in ChromaDB
2. **Query** — your question is matched against stored chunks via cosine similarity; the top-k chunks are injected into a Gemini prompt as context
3. **Answer** — Gemini answers using only the retrieved context, with sources returned alongside the answer

---

## Tech stack

| Layer | Tool |
|---|---|
| API framework | FastAPI |
| PDF parsing | PyMuPDF |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB |
| LLM | Google Gemini 2.5 Flash |

---

## Project structure

```
├── main.py          # FastAPI app — upload & query routes
├── rag.py           # RAG pipeline — PDF load, chunking, embedding, storage
├── upload/          # Uploaded PDFs (git-ignored)
├── Chroma_DB/       # ChromaDB persistence (git-ignored)
├── .env             # API keys (git-ignored)
├── requirements.txt
└── README.md
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/PhiViet03/RAGnosis
cd RAGnosis
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set your API key**
```bash
# then edit .env and add your GEMINI_API_KEY
```

**5. Run the server**
```bash
python main.py
```

API docs available at `http://localhost:8000/docs`

---

## API endpoints

### `POST /upload`
Upload a PDF to index.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your-document.pdf"
```

Response:
```json
{ "chunk indexed": 42 }
```

---

### `POST /query`
Ask a question about the uploaded PDF.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "k": 3}'
```

Response:
```json
{
  "answer": "The document is about ...",
  "sources": ["chunk 1 text...", "chunk 2 text...", "chunk 3 text..."]
}
```

---

### `GET /health`
Check if the server is running and a PDF is indexed.

```bash
curl http://localhost:8000/health
```

Response:
```json
{ "status": "ok", "indexed": true }
```

---

## Environment variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Your Google Gemini API key — get one at [aistudio.google.com](https://aistudio.google.com) |

---

## .gitignore recommendations

Make sure these are ignored:
```
.env
upload/
Chroma_DB/
__pycache__/
venv/
```
