"""RAG: retrieve chunks from ChromaDB and generate answers with OpenAI."""

from .answer import ask
from .retriever import Retriever

__all__ = ["Retriever", "ask"]
