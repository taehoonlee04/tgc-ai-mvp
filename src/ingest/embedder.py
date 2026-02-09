"""OpenAI embeddings wrapper with retry."""

from __future__ import annotations

from openai import AsyncOpenAI, OpenAI
from openai import RateLimitError as OpenAIRateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
BATCH_SIZE = 100  # texts per API call (OpenAI accepts many; 100 is conservative)
RATE_LIMIT_WAIT_SEC = 65  # wait for TPM window to reset


def _wait_for_embed_retry(retry_state):
    """Wait ~65s on rate limit so TPM resets; exponential backoff for other errors."""
    if retry_state.outcome is not None and retry_state.outcome.failed:
        exc = retry_state.outcome.exception()
        if isinstance(exc, OpenAIRateLimitError) or getattr(exc, "status_code", None) == 429:
            return RATE_LIMIT_WAIT_SEC
    return wait_exponential(multiplier=1, min=2, max=30)(retry_state)


class Embedder:
    """Generate embeddings via OpenAI API with retry logic."""

    def __init__(self, api_key: str):
        self._client = OpenAI(api_key=api_key)
        self._async_client = AsyncOpenAI(api_key=api_key)

    @retry(
        stop=stop_after_attempt(15),
        wait=_wait_for_embed_retry,
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OpenAIRateLimitError)),
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

    @retry(
        stop=stop_after_attempt(20),
        wait=_wait_for_embed_retry,
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OpenAIRateLimitError)),
        reraise=True,
    )
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in one API call. Truncates each to ~8K chars."""
        if not texts:
            return []
        inputs = [t[:8000] if len(t) > 8000 else t for t in texts]
        resp = self._client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=inputs,
        )
        # API returns in same order as input
        by_index = {d.index: d.embedding for d in resp.data}
        return [by_index[i] for i in range(len(inputs))]

    @retry(
        stop=stop_after_attempt(15),
        wait=_wait_for_embed_retry,
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OpenAIRateLimitError)),
        reraise=True,
    )
    async def embed_async(self, text: str) -> list[float]:
        """Async: generate embedding for text. Truncates to ~8K chars if needed."""
        if len(text) > 8000:
            text = text[:8000]
        resp = await self._async_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return resp.data[0].embedding
