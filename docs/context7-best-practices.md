# Context7-Informed Best Practices for TGC RAG MVP

Recommendations based on official docs for ChromaDB, FastAPI, and OpenAI via Context7.

---

## 1. FastAPI: Cache Settings with `@lru_cache`

**Context7 source:** FastAPI Settings Dependency Injection recommends `@lru_cache` on `get_settings()` so Settings is instantiated once and reused across requests, avoiding repeated file I/O when reading `.env`.

**Current:** `get_settings()` returns `Settings()` on every call—no caching.

**Change:**
```python
from functools import lru_cache

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

**Why:** Prevents repeated `.env` reads and Pydantic validation on each request.

---

## 2. FastAPI: Use Lifespan for Resource Management

**Context7 source:** FastAPI recommends the `lifespan` parameter instead of deprecated `on_startup`/`on_shutdown`. Use `@asynccontextmanager` to load resources at startup and clean up at shutdown.

**Current:** `get_rag()` creates Embedder + Retriever (and ChromaDB client) on each request via Depends. ChromaDB client is not explicitly shared or cleaned up.

**Change:** Use a lifespan context manager to create and cache the Retriever (and optionally ChromaDB client) at startup, and release at shutdown. Dependencies can read from `app.state` instead of constructing per-request.

**Why:** Reduces repeated client creation, makes lifecycle explicit, and aligns with FastAPI’s recommended pattern.

---

## 3. ChromaDB: Use `where` and `where_document` for Filtered Queries

**Context7 source:** ChromaDB supports `where` (metadata filters) and `where_document` (full-text search) in `collection.query()`.

**Current:** `retriever.retrieve()` only uses `query_embeddings` and `n_results`.

**Change:** Add optional `where` and `where_document` parameters to `retrieve()` and pass them through to `collection.query()`.

**Why:** Enables filtering by section, author, date, or document content without changing the core RAG flow.

---

## 4. OpenAI: Structured Error Handling

**Context7 source:** Use typed exceptions: `AuthenticationError`, `RateLimitError`, `APIError`. Handle each with specific logic (e.g., exponential backoff for rate limits).

**Current:** Embedder already retries on `RateLimitError`. The RAG answer path (`src/rag/answer.py`) does not catch or map OpenAI errors to HTTP responses.

**Change:** Wrap `answer_ask()` in try/except for `AuthenticationError`, `RateLimitError`, `APIError` and return appropriate HTTP status codes (401, 429, 502/503) with clear messages.

**Why:** Avoids exposing raw stack traces and gives clients actionable error information.

---

## 5. OpenAI: Embeddings API Accepts Array Input

**Context7 source:** Embeddings API `input` can be a string or array of texts. Use array input for batch embedding.

**Current:** `embed_batch()` already sends `input=inputs` (list of strings). ✓

**Why:** No change needed; implementation matches documented behavior.

---

## 6. FastAPI: Run Blocking Calls in Thread Pool

**Context7 source:** For sync/blocking code (e.g., synchronous SDKs), run in a thread pool to avoid blocking the event loop.

**Current:** `/ask` uses `asyncio.to_thread(_run_rag_sync, ...)`. ✓

**Why:** No change needed; pattern is correct.

---

## 7. Optional: Async OpenAI Client

**Context7 source:** OpenAI provides `AsyncOpenAI` for async chat completions and embeddings.

**Current:** Using sync `OpenAI` client in `Embedder` and `answer.py`, with `asyncio.to_thread` to avoid blocking.

**Change (optional):** Use `AsyncOpenAI` in the API layer so RAG runs fully async without a thread pool. Embedder and answer would need async variants.

**Why:** Reduces thread pool usage and can improve concurrency under load.

---

## Summary: Priority Order

| Priority | Change | Effort | Impact |
|----------|--------|--------|--------|
| 1 | Add `@lru_cache` to `get_settings()` | Trivial | Avoids repeated Settings init |
| 2 | Add OpenAI error handling in RAG answer path | Low | Better client-facing errors |
| 3 | Add optional `where`/`where_document` to retriever | Low | Enables filtered RAG queries |
| 4 | Lifespan for Retriever/ChromaDB client | Medium | Cleaner resource lifecycle |
| 5 | Use AsyncOpenAI for API layer | Medium | Native async, fewer threads |
