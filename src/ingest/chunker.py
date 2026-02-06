"""Split articles into chunks for embedding, with overlap and metadata."""

from __future__ import annotations

from dataclasses import dataclass

from ..scraper.parser import Article

CHUNK_CHARS = 2400  # ~600 tokens
OVERLAP_CHARS = 400  # ~100 tokens
SLIDE_CHARS = CHUNK_CHARS - OVERLAP_CHARS  # 2000
MIN_ARTICLE_CHARS = 800  # Below this, keep as single chunk


@dataclass
class Chunk:
    """A chunk of text with metadata for retrieval."""

    text: str
    title: str
    author: str
    section: str
    date: str
    source_url: str
    chunk_index: int


def _find_break(text: str, start: int, end: int) -> int:
    """Find a good sentence boundary between start and end."""
    segment = text[start:end]
    last_period = segment.rfind(". ")
    last_newline = segment.rfind("\n\n")
    break_rel = max(last_period, last_newline)
    if break_rel > CHUNK_CHARS // 2:
        return start + break_rel + 1
    return end


def chunk_article(article: Article) -> list[Chunk]:
    """
    Split article content into chunks with overlap.

    For short articles (< MIN_ARTICLE_CHARS), returns a single chunk.
    Each chunk carries article metadata for retrieval context.
    """
    content = article.content.strip()
    if not content:
        return []

    metadata = {
        "title": article.title,
        "author": article.author,
        "section": article.section,
        "date": article.date,
        "source_url": article.url,
    }

    if len(content) < MIN_ARTICLE_CHARS:
        return [
            Chunk(
                text=content,
                chunk_index=0,
                **metadata,
            )
        ]

    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(content):
        end = min(start + CHUNK_CHARS, len(content))
        break_at = _find_break(content, start, end)
        text = content[start:break_at].strip()
        if text:
            chunks.append(
                Chunk(
                    text=text,
                    chunk_index=idx,
                    **metadata,
                )
            )
            idx += 1
        start = break_at - OVERLAP_CHARS if break_at < len(content) else len(content)

    return chunks
