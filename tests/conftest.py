import pytest
import os


@pytest.fixture(autouse=True)
def no_real_api_calls(monkeypatch):
    """
    Prevent any real API calls during tests.
    Tests that need API behaviour must mock explicitly.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
