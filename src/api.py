"""FastAPI app: RAG over TGC article chunks."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from src.ingest.embedder import Embedder
from src.rag.answer import ask as answer_ask
from src.rag.retriever import Retriever

load_dotenv()

# Project root (parent of src/)
ROOT = Path(__file__).resolve().parent.parent
STATIC_INDEX = ROOT / "static" / "index.html"

app = FastAPI(
    title="TGC RAG API",
    description="Ask questions over The Gospel Coalition articles (RAG).",
    version="0.1.0",
)


def _get_rag():
    chroma_path = os.getenv("CHROMA_PATH", "./data/chroma")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not set",
        )
    embedder = Embedder(api_key=api_key)
    retriever = Retriever(chroma_path=chroma_path, embedder=embedder)
    return retriever, api_key


class AskRequest(BaseModel):
    query: str
    n_chunks: int = 5


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the chat UI."""
    if STATIC_INDEX.exists():
        return FileResponse(STATIC_INDEX)
    return HTMLResponse(
        "<p>TGC RAG API. <a href='/docs'>Docs</a> | <a href='/health'>Health</a></p>",
        status_code=200,
    )


@app.get("/api")
def api_info():
    return {
        "message": "TGC RAG API",
        "docs": "/docs",
        "health": "/health",
        "ask": "POST /ask with {\"query\": \"Your question?\"}",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Return an answer and source chunks for the given question."""
    try:
        retriever, api_key = _get_rag()
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    n = max(1, min(request.n_chunks, 20))
    chunks = retriever.retrieve(request.query, n=n)
    answer_text = answer_ask(request.query, chunks, api_key=api_key)
    sources = [
        {
            "title": c["title"],
            "author": c["author"],
            "source_url": c["source_url"],
            "snippet": c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
        }
        for c in chunks
    ]
    return AskResponse(answer=answer_text, sources=sources)
