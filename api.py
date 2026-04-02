# api.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from orchestrator import query
from ingest import ingest
from config import CODE_MODEL, REASONING_MODEL, PROVIDER

app = FastAPI(title="REX — Repository Engineering eXpert")

# ── CORS ──────────────────────────────────────────────────────────
# Allows the browser frontend to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend static files ───────────────────────────────────
app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")

# ── Session store ─────────────────────────────────────────────────
# Holds conversation history per session in memory
# Simple dict for now — good enough for a local single-user tool
sessions: dict[str, list[dict]] = {}


# ── Request / Response models ─────────────────────────────────────
class QueryRequest(BaseModel):
    question:     str
    project_name: str
    session_id:   str  = "default"
    model:        str  = CODE_MODEL
    n_results:    int  = 5

class QueryResponse(BaseModel):
    answer:     str
    session_id: str
    model_used: str
    sources:    list[dict]

class IngestRequest(BaseModel):
    project_path: str

class IngestResponse(BaseModel):
    success: bool
    message: str
    project_name: str


# ── Routes ────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "REX is running", "version": "0.1.0"}


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
        from pathlib import Path
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
    Maintains conversation history per session_id.
    """
    # Validate model choice
    allowed_models = {CODE_MODEL, REASONING_MODEL}
    if request.model not in allowed_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from: {allowed_models}"
        )

    # Load or create session history
    history = sessions.get(request.session_id, [])

    try:
        answer, updated_history = query(
            question=request.question,
            project_name=request.project_name,
            history=history,
            n_results=request.n_results,
            model=request.model
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Save updated history back to session
    sessions[request.session_id] = updated_history

    # Pull sources from last user message for citation
    from retriever import retrieve
    sources = retrieve(request.question, request.project_name, request.n_results)

    return QueryResponse(
        answer=answer,
        session_id=request.session_id,
        model_used=request.model,
        sources=[{
            "filepath":   s["filepath"],
            "start_line": s["start_line"],
            "end_line":   s["end_line"],
            "score":      s["score"]
        } for s in sources]
    )


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clear conversation history for a session — frontend 'New Chat' button."""
    sessions.pop(session_id, None)
    return {"cleared": session_id}


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)