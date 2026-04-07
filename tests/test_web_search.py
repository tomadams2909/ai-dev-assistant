from unittest.mock import patch, MagicMock
from tools.web_search import search


def test_web_search_returns_string():
    """search() always returns a string, never raises."""
    mock_results = [
        {"title": "Test", "href": "https://example.com", "body": "test body content"}
    ]
    with patch("tools.web_search.DDGS") as mock_ddgs:
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_instance.text.return_value = mock_results
        mock_ddgs.return_value = mock_instance

        result = search("test query")
    assert isinstance(result, str)
    assert "Test" in result


def test_web_search_handles_failure_gracefully():
    """search() returns empty string on any exception — never blocks a response."""
    with patch("tools.web_search.DDGS", side_effect=Exception("network error")):
        result = search("test query")
    assert result == ""
