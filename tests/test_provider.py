import pytest
from models.provider import ModelProvider, OllamaProvider, ClaudeProvider, GeminiProvider
from unittest.mock import patch


def test_provider_interface():
    """All providers must implement the required interface."""
    required = ["chat", "chat_stream", "embed", "HAS_NATIVE_SEARCH"]
    for attr in required:
        assert hasattr(ModelProvider, attr), f"ModelProvider missing: {attr}"


def test_native_search_flags():
    """Native search flags must be set correctly per provider."""
    assert OllamaProvider.HAS_NATIVE_SEARCH is False
    assert ClaudeProvider.HAS_NATIVE_SEARCH is True
    assert GeminiProvider.HAS_NATIVE_SEARCH is True


def test_ollama_provider_requires_connection():
    """OllamaProvider raises RuntimeError when Ollama is not running."""
    with patch("models.provider.ollama.list", side_effect=Exception("connection refused")):
        with pytest.raises(RuntimeError, match="Ollama"):
            OllamaProvider()


def test_claude_provider_requires_api_key():
    """ClaudeProvider raises RuntimeError when API key is missing."""
    import os
    os.environ.pop("ANTHROPIC_API_KEY", None)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ClaudeProvider()
