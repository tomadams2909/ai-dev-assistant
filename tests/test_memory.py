import pytest
from memory import new_session, Session, trim_history
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
