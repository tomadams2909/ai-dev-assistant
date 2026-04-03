# api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path
from orchestrator import query
from ingest import ingest
from config import CODE_MODEL, REASONING_MODEL, PROVIDER
from memory import Session, new_session, load_session, list_sessions, delete_session

app = FastAPI(title="REX — Repository Engineering eXpert")

# ── CORS ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend static files ───────────────────────────────────
BASE_DIR = Path(__file__).parent
app.mount("/app", StaticFiles(directory=BASE_DIR / "frontend", html=True), name="frontend")

# ── Session store ─────────────────────────────────────────────────
# In-memory cache; sessions are always persisted to disk so they
# survive server restarts — if not found here, we fall back to disk.
sessions: dict[str, Session] = {}


# ── Request / Response models ─────────────────────────────────────
class QueryRequest(BaseModel):
    question:     str
    project_name: str
    session_id:   str = "default"
    model:        str = CODE_MODEL
    n_results:    int = 5

class QueryResponse(BaseModel):
    answer:     str
    session_id: str
    model_used: str
    sources:    list[dict]

class IngestRequest(BaseModel):
    project_path: str

class IngestResponse(BaseModel):
    success:      bool
    message:      str
    project_name: str


# ── Routes ────────────────────────────────────────────────────────
@app.get("/")
def root():
    return RedirectResponse(url="/app")


@app.get("/health")
def health():
    """Simple health check — useful for confirming the server is up."""
    return {"status": "ok"}


@app.get("/models")
def list_models():
    """Return available models so the frontend can populate a selector."""
    return {
        "provider": PROVIDER,
        "models": {
            "code":      CODE_MODEL,
            "reasoning": REASONING_MODEL,
        }
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest_project(request: IngestRequest):
    """
    Index a project directory into the vector store.
    Run this once per project before querying.
    """
    try:
        ingest(request.project_path)
        project_name = Path(request.project_path).resolve().name
        return IngestResponse(
            success=True,
            message=f"Successfully indexed '{project_name}'",
            project_name=project_name
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query_project(request: QueryRequest):
    """
    Ask a question about an indexed project.
    Maintains conversation history per session_id across restarts.
    """
    allowed_models = {CODE_MODEL, REASONING_MODEL}
    if request.model not in allowed_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from: {allowed_models}"
        )

    # Load session: memory cache → disk → create new
    session = sessions.get(request.session_id)
    if session is None:
        session = load_session(request.session_id)
    if session is None:
        session = new_session(request.project_name, session_id=request.session_id)

    try:
        answer, session = query(
            question=request.question,
            session=session,
            n_results=request.n_results,
            model=request.model,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sessions[session.session_id] = session

    # Retrieve sources for citation (embeddings are cheap; keeps orchestrator clean)
    from retriever import retrieve
    sources = retrieve(request.question, request.project_name, request.n_results)

    return QueryResponse(
        answer=answer,
        session_id=session.session_id,
        model_used=request.model,
        sources=[{
            "filepath":   s["filepath"],
            "start_line": s["start_line"],
            "end_line":   s["end_line"],
            "score":      s["score"],
        } for s in sources]
    )


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clear conversation history for a session — frontend 'New Chat' button."""
    sessions.pop(session_id, None)
    delete_session(session_id)
    return {"cleared": session_id}


@app.get("/sessions")
def get_sessions(project_name: str = None):
    """List all sessions, optionally filtered by project_name."""
    return list_sessions(project_name)


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Return full session data including history and full log."""
    session = load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {
        "session_id":   session.session_id,
        "project_name": session.project_name,
        "created_at":   session.created_at,
        "updated_at":   session.updated_at,
        "history":      session.history,
        "full_log":     session.full_log,
        "summary":      session.summary,
    }


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
