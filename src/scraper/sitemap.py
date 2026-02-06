"""Fetch and parse sitemaps to discover TGC article URLs."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse

import requests

USER_AGENT = "TGC-MVP-Scraper/1.0"
SITEMAP_PATHS = ["/sitemap.xml", "/wp-sitemap.xml", "/sitemap_index.xml"]
REQUEST_DELAY_SEC = 1.5

# URL path prefixes to include (content pages)
INCLUDE_PREFIXES = (
    "/article/",
    "/articles/",
    "/essays/",
    "/essay/",
    "/blogs/",
    "/blog/",
    "/commentary/",
    "/topics/",
)

# URL path prefixes to exclude
EXCLUDE_PREFIXES = (
    "/churches/",
    "/store/",
    "/donate/",
    "/courses/",
    "/course/",
    "/auth",
    "/login",
    "/register",
    "/page/",  # pagination
    "/feed/",
    "/tag/",
    "/author/",
    "/?",
)


def _fetch_xml(url: str, session: requests.Session) -> str | None:
    """Fetch URL and return response text, or None on failure."""
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException:
        return None


def _parse_sitemap_urls(xml_text: str, base_url: str) -> list[str]:
    """Extract <loc> URLs from sitemap XML. Handles both sitemap index and URL list."""
    urls: list[str] = []
    root = ET.fromstring(xml_text)

    # Handle namespace (sitemaps often use xmlns)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = root.findall(".//sm:loc", ns)
    if not locs:
        locs = root.findall(".//loc")

    for loc in locs:
        if loc.text:
            urls.append(loc.text.strip())

    return urls


def _is_sitemap_index(xml_text: str) -> bool:
    """Check if XML is a sitemap index (contains <sitemap> elements)."""
    root = ET.fromstring(xml_text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    sitemaps = root.findall(".//sm:sitemap", ns)
    if not sitemaps:
        sitemaps = root.findall(".//sitemap")
    return len(sitemaps) > 0


def _filter_content_urls(urls: list[str]) -> list[str]:
    """Filter to content URLs only, excluding non-article pages."""
    result: list[str] = []
    seen: set[str] = set()

    for url in urls:
        parsed = urlparse(url)
        path = parsed.path.lower()

        # Must match at least one include prefix
        if not any(path.startswith(p) or path == p.rstrip("/") for p in INCLUDE_PREFIXES):
            continue

        # Must not match any exclude prefix
        if any(ex in path for ex in EXCLUDE_PREFIXES):
            continue

        # Deduplicate
        key = url.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        result.append(url)

    return result


def fetch_all_urls(base_url: str, *, verbose: bool = False) -> list[str]:
    """
    Discover all article URLs from TGC sitemaps.

    Tries sitemap paths in order until one succeeds. Recursively follows
    sitemap index links. Returns filtered list of content URLs.
    """
    base_url = base_url.rstrip("/")
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    # Find working sitemap
    sitemap_url: str | None = None
    xml_text: str | None = None
    for path in SITEMAP_PATHS:
        url = urljoin(base_url, path)
        if verbose:
            print(f"  Trying {path}...")
        xml_text = _fetch_xml(url, session)
        if xml_text:
            sitemap_url = url
            if verbose:
                print(f"  Found sitemap at {path}")
            time.sleep(REQUEST_DELAY_SEC)
            break

    if not sitemap_url or not xml_text:
        return []

    # Collect all URLs (may need to follow child sitemaps)
    all_urls: list[str] = []
    to_process: list[str] = [sitemap_url]
    processed: set[str] = set()
    total = len(to_process)

    while to_process:
        current = to_process.pop(0)
        if current in processed:
            continue
        processed.add(current)

        if verbose:
            done = len(processed)
            print(f"  Fetching sitemap {done}/{total + len(to_process) - 1}...", end="\r")

        xml_text = _fetch_xml(current, session)
        if not xml_text:
            continue

        time.sleep(REQUEST_DELAY_SEC)

        urls = _parse_sitemap_urls(xml_text, base_url)

        if _is_sitemap_index(xml_text):
            # Child sitemaps - add to queue
            to_process.extend(urls)
            total = max(total, len(to_process) + len(processed))
        else:
            # URL list - collect
            all_urls.extend(urls)

    if verbose:
        print()  # newline after progress

    return _filter_content_urls(all_urls)
