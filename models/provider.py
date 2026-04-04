# models/provider.py
import os
from abc import ABC, abstractmethod
from typing import Iterator
import ollama
from config import CODE_MODEL, REASONING_MODEL, EMBEDDING_MODEL, PROVIDER, CLAUDE_MODEL


# ── Base class ────────────────────────────────────────────────────
class ModelProvider(ABC):
    """Abstract base — all model providers must implement these."""

    @abstractmethod
    def chat(self, system: str, messages: list[dict]) -> str:
        """Send a conversation and return the response string."""
        pass

    @abstractmethod
    def chat_stream(self, system: str, messages: list[dict]) -> Iterator[str]:
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

    def chat(self, system: str, messages: list[dict]) -> str:
        response = ollama.chat(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system},
                *messages
            ]
        )
        return response["message"]["content"]

    def chat_stream(self, system: str, messages: list[dict]) -> Iterator[str]:
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
    """

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

    def chat(self, system: str, messages: list[dict]) -> str:
        from usage_tracker import track_usage
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=messages,
        )
        track_usage(response.usage.input_tokens, response.usage.output_tokens)
        return response.content[0].text

    def chat_stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        from usage_tracker import track_usage
        final_message = None
        with self._client.messages.stream(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
            final_message = stream.get_final_message()
        if final_message is not None:
            track_usage(
                final_message.usage.input_tokens,
                final_message.usage.output_tokens,
            )

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "Claude has no embedding endpoint. "
            "Embeddings always use local Ollama regardless of God Mode."
        )


# ── Future providers ──────────────────────────────────────────────
class OpenAIProvider(ModelProvider):
    """OpenAI via API — placeholder for Phase 5."""

    def chat(self, system: str, messages: list[dict]) -> str:
        raise NotImplementedError("OpenAI provider not yet configured.")

    def chat_stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        raise NotImplementedError("OpenAI provider not yet configured.")
        yield  # make static analysis happy

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("OpenAI provider not yet configured.")


# ── Provider factory ──────────────────────────────────────────────
def get_provider(name: str = PROVIDER, model: str = CODE_MODEL) -> ModelProvider:
    """
    Get a model provider by name.

    When model == CLAUDE_MODEL the request is automatically routed to
    ClaudeProvider regardless of the name argument.  This lets the
    orchestrator stay unaware of external providers — it just passes
    the model string through and this factory does the routing.

    Usage:
        get_provider()                          # default local Ollama
        get_provider(model=REASONING_MODEL)     # local Ollama deepseek-r1
        get_provider(model=CLAUDE_MODEL)        # Claude Sonnet (God Mode)
    """
    # Auto-route to Claude when the Claude model is requested
    if model == CLAUDE_MODEL:
        # Raises RuntimeError with a clear message if API key is absent
        return ClaudeProvider()

    providers = {
        "ollama": lambda: OllamaProvider(chat_model=model),
        "claude": lambda: ClaudeProvider(),
        "openai": lambda: OpenAIProvider(),
    }

    if name not in providers:
        raise ValueError(f"Unknown provider '{name}'. Choose from: {list(providers.keys())}")

    return providers[name]()
