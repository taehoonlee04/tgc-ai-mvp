"""FastAPI app: RAG over TGC article chunks."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, NamedTuple

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, field_validator

from openai import APIError, AuthenticationError, RateLimitError

from src.config import Settings
from src.ingest.embedder import Embedder
from src.ingest.index_builder import COLLECTION_NAME
from src.rag.answer import ask_async as answer_ask_async
from src.rag.retriever import Retriever

# Project root (parent of src/)
ROOT = Path(__file__).resolve().parent.parent
STATIC_INDEX = ROOT / "static" / "index.html"


class RAGDeps(NamedTuple):
    retriever: Retriever
    api_key: str


def _create_rag(settings: Settings) -> RAGDeps | None:
    """Create RAG deps or None if invalid."""
    if not settings.openai_api_key:
        return None
    try:
        embedder = Embedder(api_key=settings.openai_api_key)
        retriever = Retriever(chroma_path=settings.chroma_path, embedder=embedder)
        return RAGDeps(retriever=retriever, api_key=settings.openai_api_key)
    except FileNotFoundError:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create and cache RAG deps at startup."""
    settings = get_settings()
    app.state.rag = _create_rag(settings)
    yield
    app.state.rag = None


app = FastAPI(
    title="TGC RAG API",
    description="Ask questions over The Gospel Coalition articles (RAG).",
    version="0.1.0",
    lifespan=lifespan,
)


@lru_cache
def get_settings() -> Settings:
    """Cached per FastAPI best practice: avoid repeated .env reads."""
    return Settings()


def get_rag(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> RAGDeps:
    """Use RAG from app.state (lifespan) if available; else create on demand."""
    if hasattr(request.app.state, "rag") and request.app.state.rag is not None:
        return request.app.state.rag
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    try:
        rag = _create_rag(settings)
        if rag is None:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
        return rag
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


class Source(BaseModel):
    title: str
    author: str
    source_url: str
    snippet: str


class AskRequest(BaseModel):
    query: str
    n_chunks: int = 5
    where: dict | None = None
    where_document: dict | None = None

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        s = v.strip() if v else ""
        if not s:
            raise ValueError("query cannot be empty")
        return s

    @field_validator("n_chunks")
    @classmethod
    def clamp_n_chunks(cls, v: int) -> int:
        return max(1, min(20, v))


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def _check_chroma_health(chroma_path: str) -> dict:
    """Blocking: verify ChromaDB exists and is readable."""
    import chromadb
    path = Path(chroma_path)
    if not path.exists():
        return {"status": "unhealthy", "reason": "ChromaDB path not found"}
    try:
        client = chromadb.PersistentClient(path=str(path))
        coll = client.get_collection(COLLECTION_NAME)
        count = coll.count()
        return {"status": "ok", "chroma_chunks": count}
    except Exception as e:
        return {"status": "unhealthy", "reason": str(e)}


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
async def health(settings: Annotated[Settings, Depends(get_settings)]):
    """Health check: verifies ChromaDB is reachable."""
    result = await asyncio.to_thread(_check_chroma_health, settings.chroma_path)
    if result["status"] != "ok":
        raise HTTPException(status_code=503, detail=result)
    return result


@app.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequest,
    rag: Annotated[RAGDeps, Depends(get_rag)],
):
    """Return an answer and source chunks for the given question."""
    try:
        chunks = await rag.retriever.retrieve_async(
            body.query,
            n=body.n_chunks,
            where=body.where,
            where_document=body.where_document,
        )
        answer_text = await answer_ask_async(
            body.query, chunks, api_key=rag.api_key
        )
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid or missing OpenAI API key")
    except RateLimitError:
        raise HTTPException(status_code=429, detail="OpenAI rate limit exceeded; retry later")
    except APIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI API error: {getattr(e, 'message', str(e))}",
        )
    sources = [
        Source(
            title=c["title"],
            author=c["author"],
            source_url=c["source_url"],
            snippet=c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
        )
        for c in chunks
    ]
    return AskResponse(answer=answer_text, sources=sources)
