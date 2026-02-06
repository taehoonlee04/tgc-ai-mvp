"""ChromaDB collection management and document upsert."""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb

from .chunker import Chunk

COLLECTION_NAME = "tgc-articles"


def _chunk_id(source_url: str, chunk_index: int) -> str:
    """Generate unique ID for a chunk."""
    h = hashlib.sha256(source_url.encode()).hexdigest()[:12]
    return f"{h}_{chunk_index}"


class IndexBuilder:
    """Build and populate ChromaDB collection with article chunks."""

    def __init__(self, chroma_path: str):
        path = Path(chroma_path)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Add chunks with pre-computed embeddings to the collection.

        Replaces existing collection on each full run (clear then add).
        """
        collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "TGC article chunks"},
        )

        ids = [_chunk_id(c.source_url, c.chunk_index) for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "title": c.title,
                "author": c.author,
                "section": c.section,
                "date": c.date,
                "source_url": c.source_url,
            }
            for c in chunks
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def clear_and_add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Clear collection and add all chunks (full replace)."""
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.add_chunks(chunks, embeddings)
