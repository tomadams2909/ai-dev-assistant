from pathlib import Path

# ── Model settings ────────────────────────────────────────────────
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL      = "codellama"

# ── Project index store ───────────────────────────────────────────
# All projects get their own index stored here permanently
JARVIS_HOME = Path.home() / ".jarvis"
VECTOR_STORE = JARVIS_HOME / "vector_store"
JARVIS_HOME.mkdir(exist_ok=True)
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