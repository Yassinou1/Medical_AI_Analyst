"""Tests for the FastAPI application endpoints."""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_check():
    """Health endpoint returns OK status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
