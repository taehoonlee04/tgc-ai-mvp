"""ChromaDB collection management and document upsert."""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb

from .chunker import Chunk

COLLECTION_NAME = "tgc-articles"
# ChromaDB has a max add() batch size (~5461); stay under it
CHROMA_ADD_BATCH_SIZE = 5000


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
        Adds in batches to stay under ChromaDB's max batch size.
        """
        collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "TGC article chunks"},
        )

        for start in range(0, len(chunks), CHROMA_ADD_BATCH_SIZE):
            batch = chunks[start : start + CHROMA_ADD_BATCH_SIZE]
            batch_embs = embeddings[start : start + CHROMA_ADD_BATCH_SIZE]
            ids = [_chunk_id(c.source_url, c.chunk_index) for c in batch]
            documents = [c.text for c in batch]
            metadatas = [
                {
                    "title": c.title,
                    "author": c.author,
                    "section": c.section,
                    "date": c.date,
                    "source_url": c.source_url,
                }
                for c in batch
            ]
            collection.add(
                ids=ids,
                embeddings=batch_embs,
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
