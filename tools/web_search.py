"""
DuckDuckGo web search fallback for providers without native search capability.
Used by: Groq (Llama 3.3 70B), local Ollama models.
Not used by: Claude (native Anthropic search), Gemini (Google grounding).
"""
# tools/web_search.py
import logging
try:
    from ddgs import DDGS          # new package name (ddgs >= 1.0)
except ImportError:
    from duckduckgo_search import DDGS  # fallback for older installs

logger = logging.getLogger(__name__)


def fetch_web_results(query: str, max_results: int = 3) -> str:
    """
    Fetch DuckDuckGo search results for query and return a formatted string.

    Returns an empty string if the search fails or returns no results.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            logger.warning("web_search: query returned no results for: %r", query)
            return ""
        logger.info("web_search: got %d results for: %r", len(results), query)
        formatted = "Web search results:\n"
        for r in results:
            formatted += f"\n{r['title']}\n{r['href']}\n{r['body'][:300]}\n"
        return formatted
    except Exception as exc:
        logger.error("web_search: search failed for %r — %s: %s", query, type(exc).__name__, exc)
        return ""


# Alias used by orchestrator for clarity
search = fetch_web_results
