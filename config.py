from pathlib import Path

# ── Model settings ────────────────────────────────────────────────
PROVIDER         = "ollama"
EMBEDDING_MODEL  = "nomic-embed-text"
CODE_MODEL       = "codellama"   # used for code explanation and analysis
REASONING_MODEL  = "llama3"      # used for broader reasoning and conversation

# Default — orchestrator uses this unless told otherwise
CHAT_MODEL = CODE_MODEL

# ── Project index store ───────────────────────────────────────────
REX_HOME     = Path.home() / ".rex"
VECTOR_STORE = REX_HOME / "vector_store"
REX_HOME.mkdir(exist_ok=True)
VECTOR_STORE.mkdir(exist_ok=True)

# ── File access rules ─────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".py", ".js", ".ts", ".sql", ".md", ".yaml", ".json", ".txt", ".html", ".css"}
EXCLUDED_DIRS      = {".git", "__pycache__", "node_modules", ".venv", "dist", "build"}
EXCLUDED_FILES     = {".env", ".env.local", "secrets.yaml", "secrets.json"}

def is_allowed(path: Path) -> bool:
    path = path.resolve()
    if any(part in EXCLUDED_DIRS for part in path.parts):
        return False
    if path.name in EXCLUDED_FILES:
        return False
    return path.suffix in ALLOWED_EXTENSIONS