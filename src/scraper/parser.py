"""Fetch and parse TGC article pages to extract title, author, section, date, body."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = "TGC-MVP-Scraper/1.0"
REQUEST_DELAY_SEC = 1.5
TIMEOUT_SEC = 30


@dataclass
class Article:
    """Parsed article with metadata and content."""

    url: str
    title: str
    author: str
    section: str
    date: str
    content: str


def _strip_html(html: str) -> str:
    """Extract plain text from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    return " ".join(soup.get_text(separator=" ").split())


def _get_meta_content(soup: BeautifulSoup, prop: str) -> str | None:
    """Get content from meta tag with property (e.g. og:title)."""
    tag = soup.find("meta", attrs={"property": prop})
    if tag and tag.get("content"):
        return tag.get("content", "").strip()
    return None


def _get_meta_name(soup: BeautifulSoup, name: str) -> str | None:
    """Get content from meta tag with name."""
    tag = soup.find("meta", attrs={"name": name})
    if tag and tag.get("content"):
        return tag.get("content", "").strip()
    return None


def _get_json_ld(soup: BeautifulSoup) -> dict | list | None:
    """Extract first JSON-LD script with Article/NewsArticle schema."""
    for script in soup.find_all("script", type="application/ld+json"):
        if script.string:
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get("@type") in ("Article", "NewsArticle"):
                            return item
                elif isinstance(data, dict) and data.get("@type") in ("Article", "NewsArticle"):
                    return data
            except json.JSONDecodeError:
                continue
    return None


def _extract_author_from_json_ld(ld: dict) -> str:
    """Extract author name from JSON-LD."""
    author = ld.get("author")
    if isinstance(author, dict):
        return author.get("name", "")
    if isinstance(author, str):
        return author
    if isinstance(author, list) and author:
        first = author[0]
        if isinstance(first, dict):
            return first.get("name", "")
        if isinstance(first, str):
            return first
    return ""


def _extract_section_from_url(url: str) -> str:
    """Derive section from URL path (e.g. /article/foo -> article)."""
    parsed = urlparse(url)
    parts = [p for p in parsed.path.strip("/").split("/") if p and p != "article" and not p.isdigit()]
    if parts:
        return parts[0].replace("-", " ").title()
    return ""


def parse_article(html: str, url: str) -> Article | None:
    """
    Parse article HTML and extract title, author, section, date, body.

    Returns None if content is empty or page is not a valid article.
    """
    soup = BeautifulSoup(html, "html.parser")
    ld = _get_json_ld(soup)

    # Title: og:title, h1, title
    title = (
        _get_meta_content(soup, "og:title")
        or (ld.get("headline") if isinstance(ld, dict) else None)
        or (soup.find("h1").get_text(strip=True) if soup.find("h1") else None)
        or (soup.find("title").get_text(strip=True) if soup.find("title") else None)
        or ""
    )

    # Author: article:author, JSON-LD, .author, .byline
    author = (
        _get_meta_content(soup, "article:author")
        or (ld and _extract_author_from_json_ld(ld))
        or ""
    )
    if not author:
        for sel in (".author", ".byline", "[rel=author]", ".post-author"):
            el = soup.select_one(sel)
            if el:
                author = el.get_text(strip=True)
                break

    # Section: article:section, breadcrumb, URL
    section = (
        _get_meta_content(soup, "article:section")
        or _extract_section_from_url(url)
        or ""
    )
    if not section:
        nav = soup.select_one("nav[aria-label=breadcrumb], .breadcrumb, [class*=breadcrumb]")
        if nav:
            links = nav.find_all("a")
            if len(links) >= 2:
                section = links[-2].get_text(strip=True)

    # Date: article:published_time, JSON-LD datePublished
    date = (
        _get_meta_content(soup, "article:published_time")
        or (ld.get("datePublished") if isinstance(ld, dict) else None)
        or ""
    )
    if isinstance(date, str) and len(date) > 10:
        date = date[:10]  # YYYY-MM-DD

    # Body: main content area
    body = ""
    for sel in ("article", ".post-content", ".entry-content", ".article-body", "[class*=content]", "main"):
        el = soup.select_one(sel)
        if el:
            # Remove script, style, nav
            for tag in el.find_all(["script", "style", "nav"]):
                tag.decompose()
            body = _strip_html(str(el))
            if len(body) > 200:
                break

    if not body and soup.find("article"):
        body = _strip_html(str(soup.find("article")))

    if not body or len(body) < 100:
        return None

    return Article(
        url=url,
        title=title or "Untitled",
        author=author or "Unknown",
        section=section or "General",
        date=date or "",
        content=body,
    )


_last_fetch_time = 0.0


def fetch_and_parse_article(
    url: str,
    session: requests.Session | None = None,
    *,
    rate_limit: bool = True,
) -> Article | None:
    """
    Fetch URL and parse as article.
    When rate_limit=True (default), waits REQUEST_DELAY_SEC between requests.
    Set rate_limit=False when using a bounded worker pool to limit concurrency instead.
    """
    global _last_fetch_time
    sess = session or requests.Session()
    if not hasattr(sess, "_tgc_headers_set"):
        sess.headers["User-Agent"] = USER_AGENT
        sess._tgc_headers_set = True

    if rate_limit:
        elapsed = time.monotonic() - _last_fetch_time
        if elapsed < REQUEST_DELAY_SEC:
            time.sleep(REQUEST_DELAY_SEC - elapsed)
        _last_fetch_time = time.monotonic()

    try:
        resp = sess.get(url, timeout=TIMEOUT_SEC)
        resp.raise_for_status()
        return parse_article(resp.text, url)
    except requests.RequestException:
        return None
