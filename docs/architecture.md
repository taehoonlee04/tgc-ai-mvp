# TGC RAG MVP – Architecture

## Current system (what you have)

```mermaid
flowchart TB
    subgraph INGEST["Ingest pipeline (run_ingest.py)"]
        SITEMAP[Sitemaps] --> FETCH[Fetch URLs]
        FETCH --> CRAWL[Crawl + parse articles]
        CRAWL --> CHUNK[Chunk articles]
        CHUNK --> EMBED[Embed chunks - OpenAI]
        EMBED --> CHROMA[(ChromaDB)]
    end

    subgraph RAG["RAG at query time"]
        USER[User] --> UI[Chat UI]
        UI --> API[FastAPI /ask]
        API --> RETRIEVE[Retriever]
        RETRIEVE --> EMBED_Q[Embed query - OpenAI]
        EMBED_Q --> CHROMA
        CHROMA --> TOP_K[Top-k chunks]
        TOP_K --> LLM[OpenAI Chat - answer]
        LLM --> API
        API --> UI
        UI --> USER
    end

    CHROMA -.->|"persisted\n./data/chroma"| DISK[(Disk)]
```

## What’s in each box

| Piece | What it does |
|-------|----------------|
| **Ingest** | Sitemaps → article URLs → fetch HTML → parse → chunk → embed → write to ChromaDB. Ctrl+C saves partial progress. |
| **ChromaDB** | Vector store: chunk text + embedding + metadata (title, author, URL). Lives in `./data/chroma`. |
| **Chat UI** | Single page at `/`: type question → call `POST /ask` → show answer + sources. |
| **FastAPI** | Serves UI and `POST /ask`. Loads ChromaDB + embedder, runs retriever + answer. |
| **Retriever** | Embeds the user query, runs similarity search in ChromaDB, returns top-k chunks. |
| **Answer** | Takes query + chunks, builds prompt, calls OpenAI chat, returns one answer. |

## What else you could add (optional)

```mermaid
flowchart LR
    subgraph NOW["You have now"]
        A[Ingest] --> B[(Chroma)]
        B --> C[RAG API + UI]
    end

    subgraph LATER["Possible next steps"]
        D[Auth / API key]
        E[Conversation history]
        F[Streaming answers]
        G[Re-ingest / refresh]
        H[Deploy to cloud]
    end

    C -.-> D
    C -.-> E
    C -.-> F
    A -.-> G
    C -.-> H
```

- **Auth** – Protect `/ask` or the UI (e.g. API key header, login).
- **Conversation history** – Keep multi-turn context (e.g. store messages, pass last N to the LLM).
- **Streaming** – Stream the model response token-by-token in the UI (SSE or WebSocket).
- **Re-ingest / refresh** – Cron or script to re-run ingest for new/updated articles.
- **Deploy** – Run FastAPI + ChromaDB on a server (e.g. Railway, Fly.io) so others can use the UI.

---

**TL;DR:** Ingest fills ChromaDB with article chunks. The chat UI calls the API; the API retrieves chunks from ChromaDB and asks OpenAI to answer. Everything after “we have Chroma + chat UI” is optional polish and scale.
