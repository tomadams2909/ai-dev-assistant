import pytest
from httpx import AsyncClient, ASGITransport
from api import app


@pytest.mark.asyncio
async def test_health_endpoint():
    """App boots and health check responds."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "ollama" in data
    assert isinstance(data["ollama"], bool)


@pytest.mark.asyncio
async def test_models_endpoint():
    """Models endpoint returns provider and models dict."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "provider" in data
    assert "models" in data


@pytest.mark.asyncio
async def test_stream_requires_post():
    """/stream rejects GET requests."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/stream")
    assert response.status_code == 405


@pytest.mark.asyncio
async def test_review_rejects_missing_body():
    """/review returns 422 when required fields are missing."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/review", json={})
    assert response.status_code == 422
