"""Tests for chunker module."""

from src.ingest.chunker import chunk_article
from src.scraper.parser import Article


def test_chunk_short_article():
    """Short articles become a single chunk."""
    article = Article(
        url="https://example.com/article/foo",
        title="Test",
        author="Author",
        section="Section",
        date="2025-01-01",
        content="Short content. " * 20,
    )
    chunks = chunk_article(article)
    assert len(chunks) == 1
    assert chunks[0].title == "Test"
    assert chunks[0].source_url == article.url


def test_chunk_long_article():
    """Long articles are split into multiple chunks."""
    content = "Sentence one. " * 500
    article = Article(
        url="https://example.com/article/long",
        title="Long",
        author="A",
        section="S",
        date="",
        content=content,
    )
    chunks = chunk_article(article)
    assert len(chunks) >= 2
    assert all(c.title == "Long" for c in chunks)
