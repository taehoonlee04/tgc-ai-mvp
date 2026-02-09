# TGC RAG MVP

Vector storage and chatbot over The Gospel Coalition articles. Ingests content via sitemaps, chunks and embeds with OpenAI, stores in ChromaDB.

## Setup

Use **Python 3.11 or 3.12** for ChromaDB compatibility (3.14 is not yet supported).

**If you don't have Python 3.12:** install it, then create the venv:

- **Homebrew (macOS):** Fix permissions if needed (`sudo chown -R $(whoami) /opt/homebrew`), then:
  ```bash
  brew install python@3.12
  /opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
  ```
- **pyenv:** `pyenv install 3.12` then `pyenv local 3.12`, then `python -m venv .venv`

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env
# Edit .env with OPENAI_API_KEY
```

## Ingest

Sitemaps are fetched in parallel (no per-request delay). Article pages are still rate-limited (1.5s between requests).

```bash
# Quick test (few articles, fast)
python scripts/run_ingest.py --limit 10

# Ingest more (e.g. 200 articles; still caps sitemap fetches)
python scripts/run_ingest.py --limit 200

# Full ingest (all sitemaps, all articles – parallel fetch with 5 workers by default)
python scripts/run_ingest.py
# Optional: --workers 5 (parallel article fetch), --sitemap-limit N, --dry-run
# Use --workers 1 for sequential + 1.5s delay between requests
```

## Data

ChromaDB persists to `./data/chroma` (gitignored).

## Inspect Vector DB

```bash
# View collection stats and sample documents
python scripts/inspect_db.py

# Test similarity search
python scripts/inspect_db.py --query "What does the Bible say about faith?" --limit 5
```

## RAG API & UI

Run the API and chat UI:

```bash
uvicorn src.api:app --reload
# Or: python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

- **GET /** — chat UI (ask questions in the browser)
- **GET /health** — health check
- **GET /api** — API info (JSON)
- **POST /ask** — body: `{"query": "Your question?", "n_chunks": 5}` (default 5, max 20). Returns `{"answer": "...", "sources": [{ "title", "author", "source_url", "snippet" }]}`

Example:

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"query": "What does the Bible say about faith?"}'
```

Docs: http://localhost:8000/docs

## Tests

```bash
pip install pytest
pytest tests/ -v
```
