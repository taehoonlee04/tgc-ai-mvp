"""Retrieve relevant chunks from ChromaDB for a query."""

from __future__ import annotations

import asyncio
from pathlib import Path

import chromadb

from src.ingest.embedder import Embedder
from src.ingest.index_builder import COLLECTION_NAME


def _get_collection(chroma_path: str):
    path = Path(chroma_path)
    if not path.exists():
        raise FileNotFoundError(f"ChromaDB not found at {chroma_path}. Run ingest first.")
    client = chromadb.PersistentClient(path=str(path))
    return client.get_collection(COLLECTION_NAME)


def _query_with_embedding(
    collection,
    emb: list[float],
    n: int,
    where: dict | None = None,
    where_document: dict | None = None,
) -> list[dict]:
    count = collection.count()
    if count == 0:
        return []
    kwargs: dict = {
        "query_embeddings": [emb],
        "n_results": min(n, count),
        "include": ["documents", "metadatas"],
    }
    if where is not None:
        kwargs["where"] = where
    if where_document is not None:
        kwargs["where_document"] = where_document
    results = collection.query(**kwargs)
    out = []
    for doc, meta in zip(
        results["documents"][0],
        results["metadatas"][0],
    ):
        out.append({
            "text": doc,
            "title": meta.get("title", ""),
            "author": meta.get("author", ""),
            "section": meta.get("section", ""),
            "date": meta.get("date", ""),
            "source_url": meta.get("source_url", ""),
        })
    return out


class Retriever:
    """Fetch top-k chunks for a query using the tgc-articles collection."""

    def __init__(self, chroma_path: str, embedder: Embedder):
        self._collection = _get_collection(chroma_path)
        self._embedder = embedder

    def retrieve(
        self,
        query: str,
        n: int = 5,
        *,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[dict]:
        """
        Return top-n chunks for the query. Each item has keys:
        text, title, author, section, date, source_url.

        Optional ChromaDB filters:
        - where: metadata filter, e.g. {"section": "article"}
        - where_document: full-text filter, e.g. {"$contains": "search string"}
        """
        emb = self._embedder.embed(query)
        return _query_with_embedding(
            self._collection, emb, n, where=where, where_document=where_document
        )

    async def retrieve_async(
        self,
        query: str,
        n: int = 5,
        *,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[dict]:
        """Async: embed query and retrieve top-n chunks. Same output as retrieve()."""
        emb = await self._embedder.embed_async(query)
        return await asyncio.to_thread(
            _query_with_embedding,
            self._collection,
            emb,
            n,
            where,
            where_document,
        )
