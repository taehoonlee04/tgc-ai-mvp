"""Generate an answer from a query and retrieved chunks using OpenAI."""

from __future__ import annotations

from openai import AsyncOpenAI, OpenAI

SYSTEM_PROMPT = """You answer questions based only on the provided excerpts from The Gospel Coalition (TGC) articles. If the excerpts do not contain enough information, say so. Keep answers concise and cite the articles (by title or author) when relevant."""


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] From \"{c['title']}\" by {c['author']}:\n{c['text']}"
        )
    return "\n\n".join(parts)


def ask(
    query: str,
    chunks: list[dict],
    api_key: str,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Use OpenAI chat to answer the query given the retrieved chunks.
    chunks: list of dicts with at least "text", "title", "author".
    """
    if not chunks:
        return "I don't have any relevant articles to answer that. Try rephrasing or run the ingest to add more content."
    context = _build_context(chunks)
    user_content = f"""Use the following excerpts from TGC articles to answer the question.

Excerpts:
{context}

Question: {query}"""
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=500,
    )
    return resp.choices[0].message.content or ""


async def ask_async(
    query: str,
    chunks: list[dict],
    api_key: str,
    model: str = "gpt-4o-mini",
) -> str:
    """Async: use OpenAI chat to answer the query given the retrieved chunks."""
    if not chunks:
        return "I don't have any relevant articles to answer that. Try rephrasing or run the ingest to add more content."
    context = _build_context(chunks)
    user_content = f"""Use the following excerpts from TGC articles to answer the question.

Excerpts:
{context}

Question: {query}"""
    client = AsyncOpenAI(api_key=api_key)
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=500,
    )
    return resp.choices[0].message.content or ""
