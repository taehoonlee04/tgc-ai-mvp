"""OpenAI embeddings wrapper with retry."""

from __future__ import annotations

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSIONS = 1536


class Embedder:
    """Generate embeddings via OpenAI API with retry logic."""

    def __init__(self, api_key: str):
        self._client = OpenAI(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True,
    )
    def embed(self, text: str) -> list[float]:
        """Generate embedding for text. Truncates to ~8K chars if needed."""
        if len(text) > 8000:
            text = text[:8000]
        resp = self._client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return resp.data[0].embedding
