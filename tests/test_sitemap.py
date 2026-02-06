"""Tests for sitemap module."""

from src.scraper.sitemap import (
    _parse_sitemap_urls,
    _is_sitemap_index,
    _filter_content_urls,
)


def test_parse_sitemap_urls():
    """Extract URLs from sitemap XML."""
    xml = """<?xml version="1.0"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://example.com/article/foo</loc></url>
        <url><loc>https://example.com/article/bar</loc></url>
    </urlset>
    """
    urls = _parse_sitemap_urls(xml, "https://example.com")
    assert len(urls) == 2
    assert "article/foo" in urls[0]
    assert "article/bar" in urls[1]


def test_is_sitemap_index():
    """Detect sitemap index vs URL list."""
    index_xml = """<?xml version="1.0"?>
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <sitemap><loc>https://example.com/sitemap-1.xml</loc></sitemap>
    </sitemapindex>
    """
    assert _is_sitemap_index(index_xml) is True

    url_xml = """<?xml version="1.0"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://example.com/article/foo</loc></url>
    </urlset>
    """
    assert _is_sitemap_index(url_xml) is False


def test_filter_content_urls():
    """Filter to content URLs only."""
    urls = [
        "https://tgc.org/article/resurrection/",
        "https://tgc.org/churches/",
        "https://tgc.org/article/another/",
        "https://tgc.org/store/item",
    ]
    filtered = _filter_content_urls(urls)
    assert len(filtered) == 2
    assert any("resurrection" in u for u in filtered)
    assert any("another" in u for u in filtered)
    assert not any("churches" in u for u in filtered)
    assert not any("store" in u for u in filtered)
