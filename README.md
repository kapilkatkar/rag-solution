##  What is RAG?

**RAG (Retrieval-Augmented Generation)** enhances large language models (LLMs) with external knowledge. Instead of relying solely on model memory, it retrieves relevant context from your document store before generating a response.

---

## Features

- Vector search over embedded documents using **ChromaDB**
- Fast, minimal API backend via **FastAPI**
- Powered by **OpenAI GPT-3.5/GPT-4** via LangChain
- Supports loading `.txt`, `.pdf`, and `.md` files
- API secured with environment variables
- Deployable via Render, Railway, or Docker

---

## Tech Stack

- **Backend:** Python, FastAPI, LangChain
- **Vector Store:** ChromaDB
- **Embeddings:** OpenAI Embeddings (`text-embedding-ada-002`)
- **LLM:** GPT-3.5 / GPT-4 (via OpenAI API)
- **Frontend (optional):** React or static HTML

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/kapilkatkar/rag-solution.git
cd rag-solution
pip install -r requirements.txt
OPENAI_API_KEY=sk-...
uvicorn app.main:app --reload

