# models/ollama_provider.py
from abc import ABC, abstractmethod
import ollama
from config import CODE_MODEL, REASONING_MODEL, EMBEDDING_MODEL, PROVIDER


# ── Base class ────────────────────────────────────────────────────
class ModelProvider(ABC):
    """Abstract base — all model providers must implement these."""

    @abstractmethod
    def chat(self, system: str, messages: list[dict]) -> str:
        """Send a conversation and return the response string."""
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

    def embed(self, text: str) -> list[float]:
        response = ollama.embeddings(
            model=self.embedding_model,
            prompt=text
        )
        return response["embedding"]


# ── Future providers (not yet implemented) ────────────────────────
class ClaudeProvider(ModelProvider):
    """Anthropic Claude via API — plug in later for heavier tasks."""

    def chat(self, system: str, messages: list[dict]) -> str:
        raise NotImplementedError("Claude provider not yet configured.")

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Claude provider not yet configured.")


class OpenAIProvider(ModelProvider):
    """OpenAI via API — plug in later as alternative."""

    def chat(self, system: str, messages: list[dict]) -> str:
        raise NotImplementedError("OpenAI provider not yet configured.")

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("OpenAI provider not yet configured.")


# ── Provider factory ──────────────────────────────────────────────
def get_provider(name: str = PROVIDER, model: str = CODE_MODEL) -> ModelProvider:
    """
    Get a model provider by name.
    Pass model=REASONING_MODEL to switch to llama3 for broader tasks.

    Usage:
        provider = get_provider()                            # codellama default
        provider = get_provider(model=REASONING_MODEL)      # llama3
    """
    providers = {
        "ollama": lambda: OllamaProvider(chat_model=model),
        "claude": ClaudeProvider,
        "openai": OpenAIProvider,
    }

    if name not in providers:
        raise ValueError(f"Unknown provider '{name}'. Choose from: {list(providers.keys())}")

    return providers[name]()