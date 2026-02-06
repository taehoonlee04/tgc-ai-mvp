#!/usr/bin/env python3
"""Ingest TGC articles: sitemap -> crawl -> parse -> chunk -> embed -> ChromaDB."""

from __future__ import annotations

import argparse
import os
import sys
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

    print("Fetching sitemap URLs...")
    urls = fetch_all_urls(base_url, verbose=True)
    print(f"Found {len(urls)} content URLs")

    if args.limit:
        urls = urls[: args.limit]
        print(f"Limited to {len(urls)} URLs")

    print("Crawling and parsing articles...")
    articles: list = []
    session = requests.Session()
    session.headers["User-Agent"] = "TGC-MVP-Scraper/1.0"

    for i, url in enumerate(urls):
        article = fetch_and_parse_article(url, session=session)
        if article:
            articles.append(article)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(urls)} URLs, {len(articles)} articles parsed")

    print(f"Parsed {len(articles)} articles")

    if not articles:
        print("No articles to index. Exiting.")
        sys.exit(0)

    print("Chunking articles...")
    all_chunks: list = []
    for article in articles:
        all_chunks.extend(chunk_article(article))
    print(f"Created {len(all_chunks)} chunks")

    if args.dry_run:
        print("Dry run: skipping embed and index.")
        return

    print("Embedding chunks...")
    embedder = Embedder(api_key=api_key)
    embeddings: list[list[float]] = []
    for i, chunk in enumerate(all_chunks):
        emb = embedder.embed(chunk.text)
        embeddings.append(emb)
        if (i + 1) % 100 == 0:
            print(f"  Embedded {i + 1}/{len(all_chunks)}")

    print("Writing to ChromaDB...")
    from src.ingest.index_builder import IndexBuilder

    builder = IndexBuilder(chroma_path)
    builder.clear_and_add(all_chunks, embeddings)

    print(f"Done. Indexed {len(articles)} articles, {len(all_chunks)} chunks.")


if __name__ == "__main__":
    main()
