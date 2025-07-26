from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.rag_chain import get_rag_chain
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://kapilk-portfolio.netlify.app/", 
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://kapilk-portfolio.netlify.app/" 
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qa_chain = get_rag_chain()

class Query(BaseModel):
    question: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/rag-v1")
async def ask(query: Query):
    result = qa_chain.invoke(query.question)
    return {
        "answer": result["result"],
        "sources": [doc.metadata.get("source", "") for doc in result["source_documents"]]
    }
