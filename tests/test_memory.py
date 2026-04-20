import json
import pytest
from pathlib import Path
from unittest.mock import patch
from memory import new_session, Session, trim_history, build_messages, save_session, load_session
from config import MAX_HISTORY_MESSAGES


def test_new_session_returns_session():
    """new_session returns a valid Session object."""
    session = new_session("test-project")
    assert isinstance(session, Session)
    assert session.session_id is not None
    assert isinstance(session.history, list)


def test_session_history_trims_correctly():
    """History must not exceed MAX_HISTORY_MESSAGES after trimming."""
    session = new_session("test-project")
    for i in range(MAX_HISTORY_MESSAGES + 10):
        session.history.append({"role": "user", "content": f"message {i}"})
        session.history.append({"role": "assistant", "content": f"response {i}"})

    trim_history(session)
    assert len(session.history) <= MAX_HISTORY_MESSAGES


def test_session_has_required_fields():
    """Session must have all required fields."""
    session = new_session("test-project")
    assert hasattr(session, "session_id")
    assert hasattr(session, "history")
    assert hasattr(session, "project_name")


def test_trim_moves_dropped_messages_to_summary():
    """Messages dropped from history must appear in summary, not be silently lost."""
    session = new_session("test-project")
    for i in range(MAX_HISTORY_MESSAGES + 4):
        session.history.append({"role": "user",      "content": f"question {i}"})
        session.history.append({"role": "assistant",  "content": f"answer {i}"})

    trim_history(session)

    assert len(session.history) <= MAX_HISTORY_MESSAGES
    assert "question 0" in session.summary
    assert "answer 0" in session.summary


def test_trim_does_not_store_duplicate_summary():
    """Calling trim_history twice must not duplicate content in summary."""
    session = new_session("test-project")
    for i in range(MAX_HISTORY_MESSAGES + 4):
        session.history.append({"role": "user",      "content": f"unique-first-{i}"})
        session.history.append({"role": "assistant",  "content": f"reply-first-{i}"})

    trim_history(session)

    # Add more messages and trim again
    for i in range(MAX_HISTORY_MESSAGES):
        session.history.append({"role": "user",      "content": f"unique-second-{i}"})
        session.history.append({"role": "assistant",  "content": f"reply-second-{i}"})

    trim_history(session)

    # The earliest dropped message must appear exactly once — not duplicated by the second trim
    assert session.summary.count("unique-first-0") == 1


def test_build_messages_no_summary_returns_history():
    """build_messages with no summary must return only the history messages."""
    session = new_session("test-project")
    session.history = [
        {"role": "user",      "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    messages = build_messages(session)
    assert messages == session.history


def test_build_messages_with_summary_prepends_synthetic_exchange():
    """build_messages must prepend a synthetic user/assistant pair when summary exists."""
    session = new_session("test-project")
    session.summary = "User: old question\nAssistant: old answer"
    session.history = [{"role": "user", "content": "new question"}]

    messages = build_messages(session)

    assert len(messages) == 3  # synthetic user + synthetic assistant + history
    assert messages[0]["role"] == "user"
    assert "old question" in messages[0]["content"]
    assert messages[1]["role"] == "assistant"
    assert messages[2]["content"] == "new question"


def test_build_messages_empty_session_returns_empty_list():
    """build_messages on a brand new session returns an empty list."""
    session = new_session("test-project")
    assert build_messages(session) == []


def test_save_and_load_session_roundtrip(tmp_path):
    """save_session then load_session must return an equivalent session."""
    session = new_session("roundtrip-project")
    session.history = [{"role": "user", "content": "test question"}]
    session.summary = "some summary"

    with patch("memory.SESSION_STORE", tmp_path):
        save_session(session)
        loaded = load_session(session.session_id)

    assert loaded is not None
    assert loaded.session_id == session.session_id
    assert loaded.project_name == "roundtrip-project"
    assert loaded.history == session.history
    assert loaded.summary == session.summary


def test_load_session_returns_none_for_missing_file(tmp_path):
    """load_session must return None when the session file does not exist."""
    with patch("memory.SESSION_STORE", tmp_path):
        result = load_session("nonexistent-session-id")
    assert result is None


def test_load_session_returns_none_for_corrupt_json(tmp_path):
    """load_session must return None and not raise when the file contains invalid JSON."""
    corrupt_file = tmp_path / "bad-session.json"
    corrupt_file.write_text("{this is not valid json", encoding="utf-8")

    with patch("memory.SESSION_STORE", tmp_path):
        result = load_session("bad-session")

    assert result is None
