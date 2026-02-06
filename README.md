# TGC RAG MVP

Vector storage and chatbot over The Gospel Coalition articles. Ingests content via sitemaps, chunks and embeds with OpenAI, stores in ChromaDB.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env
# Edit .env with OPENAI_API_KEY
```

## Ingest

```bash
python scripts/run_ingest.py
# Optional: --limit 10 (cap URLs for testing), --dry-run (parse only, no embed/index)
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
