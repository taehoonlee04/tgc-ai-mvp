#!/usr/bin/env python3
"""Inspect ChromaDB collection: count, sample documents, test queries."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import chromadb


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect TGC ChromaDB collection")
    parser.add_argument(
        "--query",
        type=str,
        help="Test query to search for similar chunks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return for queries",
    )
    args = parser.parse_args()

    load_dotenv()
    chroma_path = os.getenv("CHROMA_PATH", "./data/chroma")

    if not Path(chroma_path).exists():
        print(f"ChromaDB not found at {chroma_path}")
        print("Run: python scripts/run_ingest.py")
        sys.exit(1)

    client = chromadb.PersistentClient(path=chroma_path)
    
    try:
        collection = client.get_collection("tgc-articles")
    except Exception as e:
        print(f"Collection 'tgc-articles' not found: {e}")
        print("Run: python scripts/run_ingest.py")
        sys.exit(1)

    # Show collection stats
    count = collection.count()
    print(f"Collection: tgc-articles")
    print(f"Total chunks: {count}")

    if count == 0:
        print("Collection is empty. Run ingest first.")
        sys.exit(0)

    # Show sample documents
    print("\n--- Sample Documents ---")
    sample = collection.get(limit=3, include=["documents", "metadatas"])
    for i, (doc, meta) in enumerate(zip(sample["documents"], sample["metadatas"]), 1):
        print(f"\n{i}. {meta['title']} ({meta['section']})")
        print(f"   Author: {meta['author']}, Date: {meta['date']}")
        print(f"   URL: {meta['source_url']}")
        print(f"   Content: {doc[:150]}...")

    # Test query if provided
    if args.query:
        print(f"\n--- Query: '{args.query}' ---")
        
        # Need embedder for query
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not set. Cannot perform query.")
            sys.exit(1)

        from src.ingest.embedder import Embedder
        embedder = Embedder(api_key=api_key)
        
        query_embedding = embedder.embed(args.query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=args.limit,
            include=["documents", "metadatas", "distances"],
        )

        for i, (doc, meta, dist) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            1,
        ):
            print(f"\n{i}. {meta['title']} (distance: {dist:.4f})")
            print(f"   Section: {meta['section']}, Author: {meta['author']}")
            print(f"   URL: {meta['source_url']}")
            print(f"   Content: {doc[:200]}...")


if __name__ == "__main__":
    main()
