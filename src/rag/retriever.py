"""Retrieve relevant chunks from ChromaDB for a query."""

from __future__ import annotations

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


class Retriever:
    """Fetch top-k chunks for a query using the tgc-articles collection."""

    def __init__(self, chroma_path: str, embedder: Embedder):
        self._collection = _get_collection(chroma_path)
        self._embedder = embedder

    def retrieve(self, query: str, n: int = 5) -> list[dict]:
        """
        Return top-n chunks for the query. Each item has keys:
        text, title, author, section, date, source_url.
        """
        count = self._collection.count()
        if count == 0:
            return []
        emb = self._embedder.embed(query)
        results = self._collection.query(
            query_embeddings=[emb],
            n_results=min(n, count),
            include=["documents", "metadatas"],
        )
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
