"""Tests for parser module."""

from src.scraper.parser import parse_article


def test_parse_article_with_og_meta():
    """Extract title from og:title."""
    body_text = "Body content here. " * 20  # Exceeds 100 char minimum
    html = f"""
    <html>
    <head><meta property="og:title" content="My Title" /></head>
    <body><article><p>{body_text}</p></article></body>
    </html>
    """
    result = parse_article(html, "https://example.com/article/foo")
    assert result is not None
    assert result.title == "My Title"
    assert len(result.content) > 100


def test_parse_article_empty_body_returns_none():
    """Empty or too-short body returns None."""
    html = "<html><body><article><p>Hi</p></article></body></html>"
    result = parse_article(html, "https://example.com/foo")
    assert result is None
