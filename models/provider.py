# models/provider.py
import os
from abc import ABC, abstractmethod
from typing import Iterator
import ollama
from config import CODE_MODEL, REASONING_MODEL, EMBEDDING_MODEL, PROVIDER, CLAUDE_MODEL, GROQ_MODEL, GEMINI_MODEL


# ── Base class ────────────────────────────────────────────────────
class ModelProvider(ABC):
    """Abstract base — all model providers must implement these."""

    # Set True in providers that handle web search natively (Claude, Gemini).
    # The orchestrator uses this to decide whether to prepend DuckDuckGo results.
    HAS_NATIVE_SEARCH: bool = False

    @abstractmethod
    def chat(self, system: str, messages: list[dict], web_search: bool = False) -> str:
        """Send a conversation and return the response string."""
        pass

    @abstractmethod
    def chat_stream(self, system: str, messages: list[dict], web_search: bool = False) -> Iterator[str]:
        """Stream response tokens one by one."""
        pass

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Convert text to an embedding vector."""
        pass


# ── Ollama (local) ────────────────────────────────────────────────
class OllamaProvider(ModelProvider):
    """
    Fully local model provider via Ollama.
    Nothing leaves your machine.
    """

    def __init__(self, chat_model: str = CODE_MODEL, embedding_model: str = EMBEDDING_MODEL):
        self.chat_model      = chat_model
        self.embedding_model = embedding_model
        self._verify_connection()

    def _verify_connection(self):
        """Check Ollama is running before anything else."""
        try:
            ollama.list()
        except Exception:
            raise RuntimeError(
                "Cannot connect to Ollama.\n"
                "Make sure Ollama is running — check your system tray or run 'ollama serve' in a terminal."
            )

    def chat(self, system: str, messages: list[dict], web_search: bool = False) -> str:
        response = ollama.chat(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system},
                *messages
            ]
        )
        return response["message"]["content"]

    def chat_stream(self, system: str, messages: list[dict], web_search: bool = False) -> Iterator[str]:
        stream = ollama.chat(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system},
                *messages
            ],
            stream=True,
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token

    def embed(self, text: str) -> list[float]:
        response = ollama.embeddings(
            model=self.embedding_model,
            prompt=text
        )
        return response["embedding"]


# ── Anthropic Claude (God Mode) ───────────────────────────────────
class ClaudeProvider(ModelProvider):
    """
    Anthropic Claude Sonnet via the Anthropic API.
    Requires ANTHROPIC_API_KEY in environment.
    Embeddings always fall back to local Ollama — Claude has no embedding endpoint.
    Native web search via web_search_20260209 tool — Claude decides autonomously whether to search.
    """

    HAS_NATIVE_SEARCH = True

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Add ANTHROPIC_API_KEY=your-key-here to your .env file to enable God Mode."
            )
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model  = CLAUDE_MODEL

    def chat(self, system: str, messages: list[dict], web_search: bool = False) -> str:
        import logging
        from usage_tracker import track_usage
        tools = [{"type": "web_search_20260209", "name": "web_search"}] if web_search else None
        response = self._client.messages.create(
            model=self._model,
            max_tokens=8096,
            system=system,
            messages=messages,
            tools=tools,
        )
        track_usage("claude", response.usage.input_tokens, response.usage.output_tokens)
        if web_search:
            search_requests = getattr(
                getattr(response.usage, "server_tool_use", None),
                "web_search_requests", 0
            )
            logging.getLogger(__name__).info(
                "claude: web_search_requests=%s", search_requests
            )
        # Find the text block — may not be first if web search was used
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""

    def chat_stream(self, system: str, messages: list[dict], web_search: bool = False) -> Iterator[str]:
        from usage_tracker import track_usage
        tools = [{"type": "web_search_20260209", "name": "web_search"}] if web_search else None
        final_message = None
        with self._client.messages.stream(
            model=self._model,
            max_tokens=8096,
            system=system,
            messages=messages,
            tools=tools,
        ) as stream:
            for text in stream.text_stream:
                yield text
            final_message = stream.get_final_message()
        if final_message is not None:
            track_usage(
                "claude",
                final_message.usage.input_tokens,
                final_message.usage.output_tokens,
            )

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "Claude has no embedding endpoint. "
            "Embeddings always use local Ollama regardless of God Mode."
        )


# ── OpenAI-compatible base (reusable for any OpenAI-format API) ───
class OpenAICompatibleProvider(ModelProvider):
    """
    Abstract base for providers using the OpenAI API wire format.
    Groq, Together AI, Fireworks, and many others are drop-in compatible.

    To add a new OpenAI-compatible provider, subclass this and call super().__init__()
    with the appropriate api_key, base_url, chat_model, and provider_name.
    No other code needs to change.

    Embeddings always raise NotImplementedError — embeddings use local Ollama.
    """

    def __init__(self, api_key: str, base_url: str, chat_model: str, provider_name: str):
        self.chat_model    = chat_model
        self.provider_name = provider_name
        self.base_url      = base_url
        if not api_key:
            raise RuntimeError(f"{provider_name} API key not set in .env")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, system: str, messages: list[dict], web_search: bool = False) -> str:
        from usage_tracker import track_usage
        openai_messages = [{"role": "system", "content": system}, *messages]
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=openai_messages,
        )
        usage = response.usage
        if usage:
            track_usage(self.provider_name, usage.prompt_tokens, usage.completion_tokens)
        return response.choices[0].message.content

    def chat_stream(self, system: str, messages: list[dict], web_search: bool = False) -> Iterator[str]:
        from usage_tracker import track_usage
        openai_messages = [{"role": "system", "content": system}, *messages]
        stream = self.client.chat.completions.create(
            model=self.chat_model,
            messages=openai_messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        input_tokens  = 0
        output_tokens = 0
        for chunk in stream:
            if chunk.usage:
                input_tokens  = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
        if input_tokens or output_tokens:
            track_usage(self.provider_name, input_tokens, output_tokens)

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            f"{self.provider_name} embeddings are not supported. "
            "Embeddings always use local Ollama."
        )


# ── Groq (Llama 3.3 70B — free tier, 500+ tokens/sec) ────────────
class GroqProvider(OpenAICompatibleProvider):
    """
    Groq cloud inference via OpenAI-compatible API.
    Requires GROQ_API_KEY in environment.
    Free tier: no cost, higher rate limits than most free APIs.
    10x more parameters than local models — 500+ tokens/second inference.
    """

    def __init__(self):
        super().__init__(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            chat_model=GROQ_MODEL,
            provider_name="groq",
        )


# ── Google Gemini 2.5 Flash (1M token context window) ────────────
class GeminiProvider(ModelProvider):
    """
    Google Gemini 2.5 Flash via the google-genai SDK.
    Requires GEMINI_API_KEY in environment.
    Key capability: 1M token context window — handles entire codebases in one request.
    Cannot extend OpenAICompatibleProvider — Google uses a different API format.
    Native web search via Google grounding — enabled via GenerateContentConfig.
    Embeddings always raise NotImplementedError — embeddings use local Ollama.
    """

    HAS_NATIVE_SEARCH = True

    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in .env")
        from google import genai
        self.client        = genai.Client(api_key=api_key)
        self.provider_name = "gemini"

    def chat(self, system: str, messages: list[dict], web_search: bool = False) -> str:
        from usage_tracker import track_usage
        from google.genai import types as genai_types
        contents = _to_gemini_history(system, messages)
        config = None
        if web_search:
            config = genai_types.GenerateContentConfig(
                tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())]
            )
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=config,
        )
        meta = getattr(response, "usage_metadata", None)
        if meta:
            track_usage(
                "gemini",
                getattr(meta, "prompt_token_count", 0),
                getattr(meta, "candidates_token_count", 0),
            )
        return response.text

    def chat_stream(self, system: str, messages: list[dict], web_search: bool = False) -> Iterator[str]:
        from usage_tracker import track_usage
        from google.genai import types as genai_types
        contents      = _to_gemini_history(system, messages)
        input_tokens  = 0
        output_tokens = 0
        config = None
        if web_search:
            config = genai_types.GenerateContentConfig(
                tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())]
            )
        for chunk in self.client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=contents,
            config=config,
        ):
            text = getattr(chunk, "text", None)
            if text:
                yield text
            meta = getattr(chunk, "usage_metadata", None)
            if meta:
                input_tokens  = getattr(meta, "prompt_token_count",     0) or input_tokens
                output_tokens = getattr(meta, "candidates_token_count", 0) or output_tokens
        if input_tokens or output_tokens:
            track_usage("gemini", input_tokens, output_tokens)

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "Gemini embeddings are not used by REX. "
            "Embeddings always use local Ollama."
        )


def _to_gemini_history(system: str, messages: list[dict]) -> list[dict]:
    """
    Convert OpenAI-style messages to Gemini content format.
    Gemini uses 'user'/'model' roles (not 'user'/'assistant').
    System prompt is prepended to the first user message.
    """
    result = []
    for i, msg in enumerate(messages):
        role    = "model" if msg["role"] == "assistant" else "user"
        content = msg["content"]
        if i == 0 and role == "user":
            content = f"{system}\n\n{content}"
        result.append({"role": role, "parts": [{"text": content}]})
    return result


# ── Provider factory ──────────────────────────────────────────────
def get_provider(name: str = PROVIDER, model: str = CODE_MODEL) -> ModelProvider:
    """
    Get a model provider by name or by model string.

    Model-string routing (takes priority over name):
      CLAUDE_MODEL  → ClaudeProvider
      GROQ_MODEL    → GroqProvider
      GEMINI_MODEL  → GeminiProvider

    All other models are routed through Ollama.

    Usage:
        get_provider()                          # default local Ollama
        get_provider(model=REASONING_MODEL)     # local Ollama deepseek-r1
        get_provider(model=CLAUDE_MODEL)        # Claude Sonnet (God Mode)
        get_provider(model=GROQ_MODEL)          # Groq Llama 3.3 70B
        get_provider(model=GEMINI_MODEL)        # Gemini 1.5 Flash

    RuntimeError is raised with a clear message if a required API key is absent.
    """
    # Route by model string — lets the orchestrator stay provider-agnostic
    if model == CLAUDE_MODEL:
        return ClaudeProvider()
    if model == GROQ_MODEL:
        return GroqProvider()
    if model == GEMINI_MODEL:
        return GeminiProvider()

    # Named provider fallback (for direct calls / future use)
    named = {
        "ollama": lambda: OllamaProvider(chat_model=model),
        "claude": ClaudeProvider,
        "groq":   GroqProvider,
        "gemini": GeminiProvider,
    }

    if name not in named:
        raise ValueError(f"Unknown provider '{name}'. Choose from: {list(named.keys())}")

    return named[name]()
