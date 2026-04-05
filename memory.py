# memory.py
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import SESSION_STORE, MAX_HISTORY_MESSAGES, SUMMARY_KEEP_MESSAGES


# ── Session data model ────────────────────────────────────────────
@dataclass
class Session:
    session_id:   str
    project_name: str
    created_at:   str
    updated_at:   str
    history:      list[dict]  # rolling window of clean Q&A — NO code chunks
    full_log:     list[dict]  # complete untruncated Q&A log
    summary:      str         # plain-text compression of trimmed history


def new_session(project_name: str, session_id: str = None) -> Session:
    now = datetime.now(timezone.utc).isoformat()
    return Session(
        session_id=session_id or str(uuid.uuid4()),
        project_name=project_name,
        created_at=now,
        updated_at=now,
        history=[],
        full_log=[],
        summary="",
    )


# ── History management ────────────────────────────────────────────
def trim_history(session: Session) -> None:
    """
    Keep only the most recent MAX_HISTORY_MESSAGES messages in session.history.
    Dropped messages are compressed into session.summary as plain text.

    IMPORTANT: history must only ever contain clean Q&A, never code chunks.
    Code chunks are injected fresh each turn and must not accumulate here.
    """
    if len(session.history) <= MAX_HISTORY_MESSAGES:
        return

    drop_count = len(session.history) - MAX_HISTORY_MESSAGES
    dropped    = session.history[:drop_count]
    session.history = session.history[drop_count:]

    # Format dropped messages as plain text and prepend any existing summary
    lines = []
    if session.summary:
        lines.append(session.summary)
        lines.append("")  # blank separator

    for msg in dropped:
        label = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{label}: {msg['content']}")

    session.summary = "\n".join(lines)


def build_messages(session: Session) -> list[dict]:
    """
    Build the messages list to pass to the model.

    If a summary exists it is prepended as a synthetic user/assistant exchange
    so the model has older context without consuming excessive tokens.
    The current turn's context-augmented user message is NOT added here —
    the orchestrator appends that immediately before calling the provider.
    """
    messages: list[dict] = []

    if session.summary:
        messages.append({
            "role": "user",
            "content": "Summary of our earlier conversation:\n" + session.summary,
        })
        messages.append({
            "role": "assistant",
            "content": "Understood. I have the context from our earlier conversation.",
        })

    messages.extend(session.history)
    return messages


# ── Persistence ───────────────────────────────────────────────────
def save_session(session: Session) -> None:
    """Write full session to ~/.rex/sessions/{session_id}.json."""
    session.updated_at = datetime.now(timezone.utc).isoformat()
    path = SESSION_STORE / f"{session.session_id}.json"
    data = {
        "session_id":   session.session_id,
        "project_name": session.project_name,
        "created_at":   session.created_at,
        "updated_at":   session.updated_at,
        "history":      session.history,
        "full_log":     session.full_log,
        "summary":      session.summary,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_session(session_id: str) -> Optional[Session]:
    """Load session from disk. Returns None if not found."""
    path = SESSION_STORE / f"{session_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Session(**data)


def list_sessions(project_name: str = None) -> list[dict]:
    """
    Return session metadata dicts (no history), sorted newest-updated first.
    Optionally filtered by project_name.
    """
    sessions = []
    for path in SESSION_STORE.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if project_name and data.get("project_name") != project_name:
            continue
        full_log   = data.get("full_log", [])
        first_user = next((m["content"] for m in full_log if m.get("role") == "user"), "")
        sessions.append({
            "session_id":    data["session_id"],
            "project_name":  data["project_name"],
            "created_at":    data["created_at"],
            "updated_at":    data["updated_at"],
            "message_count": len(full_log),
            "preview":       first_user[:80],
        })

    sessions.sort(key=lambda s: s["updated_at"], reverse=True)
    return sessions


def delete_session(session_id: str) -> bool:
    """Remove session JSON from disk. Returns True if file existed."""
    path = SESSION_STORE / f"{session_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False
