import pytest
from unittest.mock import patch, MagicMock
from memory import new_session
from orchestrator import query


def _make_provider(response: str = "test answer") -> MagicMock:
    provider = MagicMock()
    provider.HAS_NATIVE_SEARCH = False
    provider.chat.return_value = response
    return provider


def test_query_injects_chunks_into_prompt_not_history():
    """
    Code chunks must appear in the message sent to the model but must NOT
    be stored in session.history — only the clean question and answer go there.
    """
    fake_chunks = [
        {
            "text": "def secret_function(): pass",
            "filepath": "app.py",
            "start_line": 1,
            "end_line": 5,
            "score": 0.99,
        }
    ]
    provider = _make_provider()

    with patch("orchestrator.retrieve", return_value=fake_chunks), \
         patch("orchestrator.get_provider", return_value=provider), \
         patch("orchestrator.save_session"):

        session = new_session("test-project")
        answer, session = query("what does secret_function do?", session)

    # The model must have received the chunk text in the user message
    call_messages = provider.chat.call_args[0][1]  # positional arg: messages list
    full_user_content = " ".join(m["content"] for m in call_messages if m["role"] == "user")
    assert "secret_function" in full_user_content

    # session.history must contain the clean question but NOT the chunk source code
    history_content = " ".join(m["content"] for m in session.history)
    assert "what does secret_function do?" in history_content
    assert "def secret_function(): pass" not in history_content


def test_query_stores_answer_in_history():
    """Answer returned by the model must be recorded in session history."""
    provider = _make_provider(response="It initialises the database.")

    with patch("orchestrator.retrieve", return_value=[]), \
         patch("orchestrator.get_provider", return_value=provider), \
         patch("orchestrator.save_session"):

        session = new_session("test-project")
        answer, session = query("what does init_db do?", session)

    assert answer == "It initialises the database."
    assistant_messages = [m for m in session.history if m["role"] == "assistant"]
    assert any("initialises the database" in m["content"] for m in assistant_messages)
