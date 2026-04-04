from pathlib import Path

# ── Model settings ────────────────────────────────────────────────
PROVIDER         = "ollama"
EMBEDDING_MODEL  = "nomic-embed-text"
CODE_MODEL       = "qwen2.5-coder:7b"   # used for code explanation and analysis
REASONING_MODEL  = "deepseek-r1:7b"     # used for broader reasoning and conversation

# ── External provider settings ────────────────────────────────────
CLAUDE_MODEL      = "claude-sonnet-4-6"
GOD_MODE_PROVIDER = "claude"

# ── Project index store ───────────────────────────────────────────
REX_HOME     = Path.home() / ".rex"
VECTOR_STORE = REX_HOME / "vector_store"
REX_HOME.mkdir(exist_ok=True)
VECTOR_STORE.mkdir(exist_ok=True)

# ── Memory / session settings ─────────────────────────────────────
MAX_HISTORY_MESSAGES  = 20   # rolling window kept in session.history
SUMMARY_KEEP_MESSAGES = 10    # messages retained when compressing to summary
SESSION_STORE         = REX_HOME / "sessions"
SESSION_STORE.mkdir(exist_ok=True)

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