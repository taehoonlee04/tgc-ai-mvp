#!/usr/bin/env python3
"""Ingest TGC articles: sitemap -> crawl -> parse -> chunk -> embed -> ChromaDB."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from dotenv import load_dotenv

from src.scraper.sitemap import fetch_all_urls
from src.scraper.parser import fetch_and_parse_article
from src.ingest.chunker import chunk_article
from src.ingest.embedder import Embedder

DEFAULT_WORKERS = 20
EMBED_BATCH_SIZE = 100

# Shared state for Ctrl+C handler (saves partial progress)
_state: dict = {}


def _exit_immediately(signum: int, frame: object) -> None:
    """Second Ctrl+C: exit without saving."""
    print("\nExiting without save.", flush=True)
    sys.exit(0)


def _save_partial_and_exit(signum: int, frame: object) -> None:
    """On Ctrl+C: chunk, embed, and write whatever articles we have, then exit. Second Ctrl+C exits immediately."""
    if _state.get("saving"):
        _exit_immediately(signum, frame)
        return
    _state["saving"] = True
    signal.signal(signal.SIGINT, _exit_immediately)  # second Ctrl+C = exit now
    print("\n\nInterrupted. Saving partial progress... (Ctrl+C again to skip save)", flush=True)
    articles = _state.get("articles", [])
    api_key = _state.get("api_key")
    chroma_path = _state.get("chroma_path")
    dry_run = _state.get("dry_run", False)
    if not articles:
        print("No articles to save.", flush=True)
        sys.exit(0)
    if dry_run or not api_key:
        print("Dry run or no API key; cannot embed. Exiting without save.", flush=True)
        sys.exit(0)
    try:
        all_chunks = []
        for article in articles:
            all_chunks.extend(chunk_article(article))
        print(f"  Chunked {len(articles)} articles into {len(all_chunks)} chunks.", flush=True)
        embedder = Embedder(api_key=api_key)
        embeddings = []
        for start in range(0, len(all_chunks), EMBED_BATCH_SIZE):
            batch = all_chunks[start : start + EMBED_BATCH_SIZE]
            batch_embs = embedder.embed_batch([c.text for c in batch])
            embeddings.extend(batch_embs)
            print(f"  Embedded {min(start + EMBED_BATCH_SIZE, len(all_chunks))}/{len(all_chunks)}...", end="\r", flush=True)
        print()
        from src.ingest.index_builder import IndexBuilder
        builder = IndexBuilder(chroma_path)
        builder.clear_and_add(all_chunks, embeddings)
        print(f"Saved {len(articles)} articles, {len(all_chunks)} chunks to ChromaDB.", flush=True)
    except Exception as e:
        print(f"Error saving: {e}", flush=True)
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest TGC articles into vector storage")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of URLs to process (for testing)",
    )
    parser.add_argument(
        "--sitemap-limit",
        type=int,
        default=None,
        help="Cap number of sitemap files to fetch (for testing; avoids fetching 1000+ on WordPress sites). Defaults to 15 when --limit is set.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel workers for article fetch (default %(default)s). Use 1 for sequential + 1.5s delay.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse only, skip embedding and indexing",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("TGC_BASE_URL", "https://www.thegospelcoalition.org")
    chroma_path = os.getenv("CHROMA_PATH", "./data/chroma")

    if not api_key and not args.dry_run:
        print("Error: OPENAI_API_KEY not set. Set in .env or use --dry-run.")
        sys.exit(1)

    sitemap_limit = args.sitemap_limit
    if sitemap_limit is None and args.limit is not None:
        # Fetch enough sitemaps to reach the limit (15 was too low; TGC needs many)
        sitemap_limit = min(400, max(80, args.limit // 25))
    print("Fetching sitemap URLs...", flush=True)
    t0 = time.perf_counter()
    urls = fetch_all_urls(base_url, verbose=True, max_sitemap_files=sitemap_limit)
    print(f"Found {len(urls)} content URLs ({time.perf_counter() - t0:.1f}s)", flush=True)

    if args.limit:
        urls = urls[: args.limit]
        print(f"Limited to {len(urls)} URLs", flush=True)

    workers = max(1, args.workers)
    rate_limit = workers == 1
    print(f"Crawling and parsing articles ({workers} worker{'s' if workers > 1 else ''})...", flush=True)
    articles: list = []
    _state["articles"] = articles
    _state["api_key"] = api_key
    _state["chroma_path"] = chroma_path
    _state["dry_run"] = args.dry_run
    signal.signal(signal.SIGINT, _save_partial_and_exit)
    n = len(urls)

    def _fetch(url: str):
        return fetch_and_parse_article(url, session=None, rate_limit=rate_limit)

    if workers == 1:
        session = requests.Session()
        session.headers["User-Agent"] = "TGC-MVP-Scraper/1.0"
        for i, url in enumerate(urls):
            print(f"  [{i + 1}/{n}] {url[:60]}...", end="\r", flush=True)
            article = fetch_and_parse_article(url, session=session, rate_limit=True)
            if article:
                articles.append(article)
            if (i + 1) % 50 == 0 and n > 50:
                print(f"  Processed {i + 1}/{n} URLs, {len(articles)} articles parsed", flush=True)
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fetch, url): url for url in urls}
            for future in as_completed(futures):
                done += 1
                if done % 50 == 0 or done == n:
                    print(f"  Processed {done}/{n} URLs, {len(articles)} articles parsed", flush=True)
                article = future.result()
                if article:
                    articles.append(article)

    print()  # newline after progress
    print(f"Parsed {len(articles)} articles", flush=True)

    if not articles:
        print("No articles to index. Exiting.")
        sys.exit(0)

    print("Chunking articles...", flush=True)
    all_chunks: list = []
    for article in articles:
        all_chunks.extend(chunk_article(article))
    print(f"Created {len(all_chunks)} chunks", flush=True)

    if args.dry_run:
        print("Dry run: skipping embed and index.", flush=True)
        return

    print(f"Embedding chunks (batches of {EMBED_BATCH_SIZE})...", flush=True)
    embedder = Embedder(api_key=api_key)
    embeddings: list[list[float]] = []
    for start in range(0, len(all_chunks), EMBED_BATCH_SIZE):
        batch = all_chunks[start : start + EMBED_BATCH_SIZE]
        batch_texts = [c.text for c in batch]
        batch_embs = embedder.embed_batch(batch_texts)
        embeddings.extend(batch_embs)
        print(f"  Embedded {min(start + EMBED_BATCH_SIZE, len(all_chunks))}/{len(all_chunks)} chunks...", end="\r", flush=True)
    print(f"  Embedded {len(all_chunks)} chunks.", flush=True)

    print("Writing to ChromaDB...", flush=True)
    from src.ingest.index_builder import IndexBuilder

    builder = IndexBuilder(chroma_path)
    builder.clear_and_add(all_chunks, embeddings)

    print(f"Done. Indexed {len(articles)} articles, {len(all_chunks)} chunks.", flush=True)


if __name__ == "__main__":
    main()
