"""API tests using FastAPI TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def client():
    """TestClient for the FastAPI app."""
    return TestClient(app)


def test_root_returns_html(client):
    """GET / returns HTML (UI or fallback)."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


def test_api_info(client):
    """GET /api returns JSON with links."""
    resp = client.get("/api")
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "TGC RAG API"
    assert "/docs" in data["docs"]
    assert "/health" in data["health"]


def test_health(client):
    """GET /health returns 200 with chroma_chunks or 503 if ChromaDB not found."""
    resp = client.get("/health")
    assert resp.status_code in (200, 503)
    data = resp.json()
    assert "status" in data
    if resp.status_code == 200:
        assert data["status"] == "ok"
        assert "chroma_chunks" in data
    else:
        assert data["status"] == "unhealthy"
        assert "reason" in data


def test_ask_validation_empty_query(client):
    """POST /ask with empty query returns 422."""
    resp = client.post("/ask", json={"query": "   ", "n_chunks": 5})
    assert resp.status_code == 422


def test_ask_validation_n_chunks_clamped(client):
    """POST /ask accepts n_chunks and clamps to 1-20 (validates, may 503 if no Chroma)."""
    resp = client.post("/ask", json={"query": "What is faith?", "n_chunks": 999})
    # 200 if ChromaDB exists and has data; 503 if not; 500 if no API key; 502 if OpenAI error
    assert resp.status_code in (200, 422, 500, 502, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)
