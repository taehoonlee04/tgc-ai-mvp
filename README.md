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
python scripts/run_ingest.py
# Optional: --limit 10 (cap URLs; also caps sitemap fetches for speed), --sitemap-limit N, --dry-run
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

## Tests

```bash
pip install pytest
pytest tests/ -v
```
