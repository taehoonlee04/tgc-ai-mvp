"""Fetch and parse sitemaps to discover TGC article URLs."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse

import requests

USER_AGENT = "TGC-MVP-Scraper/1.0"
SITEMAP_PATHS = [
    "/wp-sitemap.xml",
    "/wp-sitemap-index.xml",
    "/sitemap.xml",
    "/sitemap_index.xml",
]
# No delay for sitemap fetches; we use a bounded worker pool instead. Article crawl keeps 1.5s in parser.
SITEMAP_WORKERS = 10

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


def _fetch_xml(url: str, session: requests.Session, verbose: bool = False) -> str | None:
    """Fetch URL and return response text, or None on failure."""
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        if verbose:
            print(f"    Failed to fetch {url}: {e}")
        return None


def _fetch_one_sitemap(url: str) -> tuple[str, str | None]:
    """Fetch one sitemap URL (creates its own session; safe for thread pool)."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return (url, _fetch_xml(url, session, verbose=False))


def _parse_sitemap_urls(xml_text: str, base_url: str) -> list[str]:
    """Extract <loc> URLs from sitemap XML. Handles both sitemap index and URL list."""
    urls: list[str] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

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
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return False
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


def fetch_all_urls(
    base_url: str,
    *,
    verbose: bool = False,
    max_sitemap_files: int | None = None,
) -> list[str]:
    """
    Discover all article URLs from TGC sitemaps.

    Tries sitemap paths in order until one succeeds. Recursively follows
    sitemap index links. Returns filtered list of content URLs.

    If max_sitemap_files is set, stop after fetching that many sitemap files
    (so with huge WordPress sites we don't fetch 1000+ sitemaps for a small --limit).
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
            print(f"  Trying {path}...", flush=True)
        xml_text = _fetch_xml(url, session, verbose=verbose)
        if xml_text:
            sitemap_url = url
            if verbose:
                print(f"  Found sitemap at {path}")
            break

    if not sitemap_url or not xml_text:
        return []

    # Parse root: if URL list we're done; if index we fetch children in parallel
    all_urls: list[str] = []
    urls_from_root = _parse_sitemap_urls(xml_text, base_url)
    if not _is_sitemap_index(xml_text):
        return _filter_content_urls(urls_from_root)

    to_fetch: list[str] = urls_from_root
    if max_sitemap_files is not None:
        to_fetch = urls_from_root[: max(1, max_sitemap_files - 1)]
    total_fetched = 1  # root already fetched
    processed: set[str] = set()

    with ThreadPoolExecutor(max_workers=SITEMAP_WORKERS) as executor:
        while to_fetch:
            if max_sitemap_files is not None and total_fetched >= max_sitemap_files:
                if verbose:
                    print(f"  Stopping after {max_sitemap_files} sitemaps (--sitemap-limit).", flush=True)
                break

            batch_size = min(
                SITEMAP_WORKERS * 2,  # fetch in chunks
                len(to_fetch),
                (max_sitemap_files - total_fetched) if max_sitemap_files else len(to_fetch),
            )
            batch = to_fetch[:batch_size]
            to_fetch = to_fetch[batch_size:]

            if verbose:
                total_display = max_sitemap_files if max_sitemap_files is not None else total_fetched + len(to_fetch) + len(batch)
                print(f"  Fetching sitemaps {total_fetched + 1}-{total_fetched + len(batch)}/{total_display}...", end="\r", flush=True)

            futures = {executor.submit(_fetch_one_sitemap, url): url for url in batch}
            for future in as_completed(futures):
                url, xml = future.result()
                if url in processed:
                    continue
                processed.add(url)
                total_fetched += 1
                if xml is None:
                    continue
                urls = _parse_sitemap_urls(xml, base_url)
                if _is_sitemap_index(xml):
                    remaining = None
                    if max_sitemap_files is not None:
                        remaining = max(0, max_sitemap_files - total_fetched - len(to_fetch))
                    if remaining is None or remaining > 0:
                        to_fetch.extend(urls[:remaining] if remaining is not None else urls)
                else:
                    all_urls.extend(urls)

    if verbose:
        print()  # newline after progress

    return _filter_content_urls(all_urls)
