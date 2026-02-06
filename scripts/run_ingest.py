#!/usr/bin/env python3
"""Ingest TGC articles: sitemap -> crawl -> parse -> chunk -> embed -> ChromaDB."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from dotenv import load_dotenv

from src.scraper.sitemap import fetch_all_urls
from src.scraper.parser import fetch_and_parse_article
from src.ingest.chunker import chunk_article
from src.ingest.embedder import Embedder


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
        sitemap_limit = 15  # enough to get content URLs when testing with --limit
    print("Fetching sitemap URLs...", flush=True)
    t0 = time.perf_counter()
    urls = fetch_all_urls(base_url, verbose=True, max_sitemap_files=sitemap_limit)
    print(f"Found {len(urls)} content URLs ({time.perf_counter() - t0:.1f}s)", flush=True)

    if args.limit:
        urls = urls[: args.limit]
        print(f"Limited to {len(urls)} URLs", flush=True)

    print("Crawling and parsing articles...", flush=True)
    articles: list = []
    session = requests.Session()
    session.headers["User-Agent"] = "TGC-MVP-Scraper/1.0"
    n = len(urls)

    for i, url in enumerate(urls):
        print(f"  [{i + 1}/{n}] {url[:60]}...", end="\r", flush=True)
        article = fetch_and_parse_article(url, session=session)
        if article:
            articles.append(article)
        if (i + 1) % 50 == 0 and n > 50:
            print(f"  Processed {i + 1}/{n} URLs, {len(articles)} articles parsed", flush=True)

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

    print("Embedding chunks...", flush=True)
    embedder = Embedder(api_key=api_key)
    embeddings: list[list[float]] = []
    for i, chunk in enumerate(all_chunks):
        print(f"  Embedding {i + 1}/{len(all_chunks)}...", end="\r", flush=True)
        emb = embedder.embed(chunk.text)
        embeddings.append(emb)
    print(f"  Embedded {len(all_chunks)} chunks.", flush=True)

    print("Writing to ChromaDB...", flush=True)
    from src.ingest.index_builder import IndexBuilder

    builder = IndexBuilder(chroma_path)
    builder.clear_and_add(all_chunks, embeddings)

    print(f"Done. Indexed {len(articles)} articles, {len(all_chunks)} chunks.", flush=True)


if __name__ == "__main__":
    main()
