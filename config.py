from pathlib import Path

# ── Local provider models ──────────────────────────────────────
# Local models (qwen2.5-coder, qwen3.5:9b) — privacy, zero cost, offline capability
PROVIDER         = "ollama"
EMBEDDING_MODEL  = "nomic-embed-text"
CODE_MODEL       = "qwen2.5-coder:7b"   # used for code explanation and analysis
REASONING_MODEL  = "qwen3.5:9b"     # used for broader reasoning and conversation

# ── External provider models ──────────────────────────────────────
# Claude Sonnet 4.6  — highest quality reasoning for complex review and analysis tasks
CLAUDE_MODEL     = "claude-sonnet-4-6"

# Groq: free cloud tier, Llama 3.3 70B (10x more parameters than local), 500+ tokens/sec
GROQ_MODEL       = "llama-3.3-70b-versatile"

# Gemini: unique 1M token context window — handles entire codebases in one request
GEMINI_MODEL     = "gemini-2.5-flash"

# ── Project index store ───────────────────────────────────────────
REX_HOME     = Path.home() / ".rex"
VECTOR_STORE = REX_HOME / "vector_store"
REX_HOME.mkdir(exist_ok=True)
VECTOR_STORE.mkdir(exist_ok=True)

# ── Memory / session settings ─────────────────────────────────────
MAX_HISTORY_MESSAGES  = 20   # rolling window kept in session.history
SESSION_STORE         = REX_HOME / "sessions"
SESSION_STORE.mkdir(exist_ok=True)

# ── Ingestion settings ────────────────────────────────────────────
CHUNK_SIZE       = 60     # lines per chunk when indexing source files
CHUNK_OVERLAP    = 10     # lines of overlap between consecutive chunks
MAX_FILE_TOKENS  = 25000  # hard limit for single-file review (est. tokens)

# ── Special project names ─────────────────────────────────────────
CHAT_MODE = "__chat__"  # sentinel used when no project is indexed — general chat session

# ── File access rules ─────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".py", ".js", ".ts", ".sql", ".md", ".yaml", ".json", ".txt", ".html"}
EXCLUDED_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "dist", "build",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "coverage",
    ".next", ".nuxt", "target", ".gradle"
}
EXCLUDED_FILES     = {".env", ".env.local", "secrets.yaml", "secrets.json"}

def is_allowed(path: Path) -> bool:
    path = path.resolve()
    if any(part in EXCLUDED_DIRS for part in path.parts):
        return False
    if path.name in EXCLUDED_FILES:
        return False
    return path.suffix in ALLOWED_EXTENSIONS