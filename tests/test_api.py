import pytest
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport
from api import app
from config import CODE_MODEL


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


@pytest.mark.asyncio
async def test_query_rejects_missing_body():
    """/query returns 422 when required fields are missing."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/query", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_rejects_invalid_model():
    """/query returns 400 when an unknown model is specified."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/query", json={
            "question":     "what is this?",
            "project_name": "test",
            "session_id":   "abc",
            "model":        "not-a-real-model",
        })
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_query_happy_path_with_mock():
    """/query returns answer and session_id when orchestrator succeeds."""
    from memory import new_session
    fake_session = new_session("test-project")

    with patch("api.query", return_value=("mocked answer", fake_session)), \
         patch("retriever.retrieve", return_value=[]):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/query", json={
                "question":     "what does main do?",
                "project_name": "test-project",
                "session_id":   fake_session.session_id,
                "model":        CODE_MODEL,
            })

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "mocked answer"
    assert "session_id" in data


@pytest.mark.asyncio
async def test_stream_rejects_invalid_model():
    """/stream returns 400 when an unknown model is specified."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/stream", json={
            "question":     "hello",
            "project_name": "test",
            "session_id":   "abc",
            "model":        "gpt-4-turbo",
        })
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_projects_endpoint_returns_list():
    """/projects returns a list."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/projects")
    assert response.status_code == 200
    assert isinstance(response.json()["projects"], list)


@pytest.mark.asyncio
async def test_usage_endpoint_returns_data():
    """/usage returns usage statistics."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/usage")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_stream_rejects_claude_without_api_key():
    """/stream returns 503 when Claude is requested but ANTHROPIC_API_KEY is absent."""
    from config import CLAUDE_MODEL
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/stream", json={
            "question":     "hello",
            "project_name": "test",
            "session_id":   "abc",
            "model":        CLAUDE_MODEL,
        })
    assert response.status_code == 503
