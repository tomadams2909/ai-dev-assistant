# tests/test_streaming.py
"""
Tests for the streaming pipeline:
  - _strip_think_stream  (orchestrator unit tests)
  - /stream SSE endpoint (FastAPI integration tests)
  - /query endpoint      (untouched — regression check)
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# ── Helpers ───────────────────────────────────────────────────────

def _collect(gen) -> str:
    """Drain a string iterator and join the tokens."""
    return "".join(gen)


def _sse_events(raw: bytes) -> list[dict]:
    """Parse raw SSE bytes into a list of decoded JSON event dicts."""
    events = []
    for block in raw.decode().split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            events.append(json.loads(block[6:]))
    return events


# ── _strip_think_stream ───────────────────────────────────────────

class TestStripThinkStream:
    """Unit tests for the streaming think-tag stripper."""

    # Import here so tests fail fast if the function is missing/renamed
    from orchestrator import _strip_think_stream as _strip

    def test_no_think_tags_passes_through(self):
        tokens = ["Hello", ", ", "world", "!"]
        assert _collect(self._strip(iter(tokens))) == "Hello, world!"

    def test_think_block_at_start_is_stripped(self):
        tokens = ["<think>chain of thought</think>", "The answer is 42."]
        assert _collect(self._strip(iter(tokens))) == "The answer is 42."

    def test_think_block_split_across_tokens(self):
        tokens = ["<thi", "nk>", "reasoning", "</thi", "nk>", "Answer."]
        assert _collect(self._strip(iter(tokens))) == "Answer."

    def test_think_block_with_leading_newline_stripped(self):
        tokens = ["<think>thoughts</think>\n\nClean answer."]
        assert _collect(self._strip(iter(tokens))) == "Clean answer."

    def test_empty_stream(self):
        assert _collect(self._strip(iter([]))) == ""

    def test_only_think_block_yields_nothing(self):
        tokens = ["<think>internal monologue</think>"]
        assert _collect(self._strip(iter(tokens))) == ""

    def test_no_think_tag_single_token(self):
        tokens = ["Just a response."]
        assert _collect(self._strip(iter(tokens))) == "Just a response."

    def test_think_open_tag_split_at_boundary(self):
        # Tag boundary spans two tokens: "<" and "think>…</think>rest"
        tokens = ["<", "think>skip</think>keep"]
        assert _collect(self._strip(iter(tokens))) == "keep"

    def test_content_before_think_tag_is_preserved(self):
        # If a response starts with real text then has a think block, preserve the text.
        # (Unusual for deepseek-r1 but the stripper must handle it.)
        tokens = ["Preamble. ", "<think>hidden</think>", "Suffix."]
        assert _collect(self._strip(iter(tokens))) == "Preamble. Suffix."


# ── /stream endpoint ──────────────────────────────────────────────

# Shared mock chunks returned by the fake Ollama stream
FAKE_TOKENS = ["Hello", ", ", "world", "!"]
FAKE_CHUNKS = [{"message": {"content": t}} for t in FAKE_TOKENS]

FAKE_SOURCES = [
    {
        "text": "def foo(): pass",
        "filepath": "src/foo.py",
        "start_line": 1,
        "end_line": 1,
        "score": 0.95,
    }
]


def _make_app():
    """Build a fresh TestClient with all external I/O patched."""
    import api  # import after patches are in place

    return TestClient(api.app, raise_server_exceptions=False)


@pytest.fixture()
def client():
    """TestClient with Ollama, ChromaDB, and disk I/O mocked out."""
    with (
        # Prevent OllamaProvider._verify_connection from failing
        patch("ollama.list", return_value={}),
        # Fake streaming chat response
        patch("ollama.chat", return_value=iter(FAKE_CHUNKS)),
        # Fake embedding (retriever path)
        patch("ollama.embeddings", return_value={"embedding": [0.1] * 768}),
        # Fake ChromaDB collection
        patch("chromadb.PersistentClient") as mock_chroma,
        # Don't write sessions to disk
        patch("memory.save_session"),
        # Don't read sessions from disk (always return None → create new)
        patch("memory.load_session", return_value=None),
    ):
        mock_col = MagicMock()
        mock_col.count.return_value = 1
        mock_col.query.return_value = {
            "documents": [["def foo(): pass"]],
            "metadatas": [[{"filepath": "src/foo.py", "start_line": 1, "end_line": 1}]],
            "distances": [[0.05]],
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_col

        from api import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


class TestStreamEndpoint:

    def test_stream_returns_200(self, client):
        res = client.post("/stream", json={
            "question": "what does foo do?",
            "project_name": "test-project",
        })
        assert res.status_code == 200

    def test_stream_content_type_is_sse(self, client):
        res = client.post("/stream", json={
            "question": "what does foo do?",
            "project_name": "test-project",
        })
        assert "text/event-stream" in res.headers["content-type"]

    def test_stream_emits_token_events(self, client):
        res = client.post("/stream", json={
            "question": "what does foo do?",
            "project_name": "test-project",
        })
        events = _sse_events(res.content)
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) > 0
        combined = "".join(e["content"] for e in token_events)
        assert combined == "Hello, world!"

    def test_stream_emits_done_event(self, client):
        res = client.post("/stream", json={
            "question": "what does foo do?",
            "project_name": "test-project",
        })
        events  = _sse_events(res.content)
        done_ev = [e for e in events if e["type"] == "done"]
        assert len(done_ev) == 1
        assert "session_id" in done_ev[0]
        assert "sources" in done_ev[0]

    def test_stream_done_event_has_sources(self, client):
        res = client.post("/stream", json={
            "question": "what does foo do?",
            "project_name": "test-project",
        })
        events  = _sse_events(res.content)
        done_ev = next(e for e in events if e["type"] == "done")
        assert isinstance(done_ev["sources"], list)
        if done_ev["sources"]:
            src = done_ev["sources"][0]
            assert "filepath" in src
            assert "start_line" in src
            assert "score" in src

    def test_stream_done_event_last(self, client):
        res    = client.post("/stream", json={
            "question": "what does foo do?",
            "project_name": "test-project",
        })
        events = _sse_events(res.content)
        assert events[-1]["type"] == "done"

    def test_stream_rejects_unknown_model(self, client):
        res = client.post("/stream", json={
            "question": "x",
            "project_name": "test-project",
            "model": "gpt-99-turbo",
        })
        assert res.status_code == 400

    def test_stream_strips_think_tags(self, client):
        """deepseek-r1 think blocks must not appear in token events."""
        think_chunks = [
            {"message": {"content": "<think>internal reasoning</think>"}},
            {"message": {"content": "Clean answer."}},
        ]
        with patch("ollama.chat", return_value=iter(think_chunks)):
            res = client.post("/stream", json={
                "question": "explain foo",
                "project_name": "test-project",
            })
        events = _sse_events(res.content)
        combined = "".join(e["content"] for e in events if e["type"] == "token")
        assert "<think>" not in combined
        assert "internal reasoning" not in combined
        assert "Clean answer." in combined


# ── /query endpoint (regression) ─────────────────────────────────

class TestQueryEndpointUntouched:
    """Confirm the original /query endpoint still works after the changes."""

    def test_query_returns_200(self, client):
        with patch("ollama.chat", return_value={"message": {"content": "It does X."}}):
            res = client.post("/query", json={
                "question": "what does foo do?",
                "project_name": "test-project",
            })
        assert res.status_code == 200

    def test_query_response_shape(self, client):
        with patch("ollama.chat", return_value={"message": {"content": "It does X."}}):
            res = client.post("/query", json={
                "question": "what does foo do?",
                "project_name": "test-project",
            })
        data = res.json()
        assert "answer" in data
        assert "session_id" in data
        assert "model_used" in data
        assert "sources" in data

    def test_query_strips_think_tags(self, client):
        raw = "<think>hidden</think>Real answer."
        with patch("ollama.chat", return_value={"message": {"content": raw}}):
            res = client.post("/query", json={
                "question": "explain foo",
                "project_name": "test-project",
            })
        assert "<think>" not in res.json()["answer"]
        assert "Real answer." in res.json()["answer"]
