# api.py
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path
from orchestrator import query, query_stream, review_file
from ingest import ingest
from config import CODE_MODEL, REASONING_MODEL, PROVIDER, VECTOR_STORE
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

class StreamRequest(BaseModel):
    question:     str
    project_name: str
    session_id:   str = "default"
    model:        str = CODE_MODEL
    n_results:    int = 5

class ReviewRequest(BaseModel):
    filepath:     str
    project_name: str
    session_id:   str = "default"
    model:        str = CODE_MODEL


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


@app.get("/projects")
def list_projects():
    """Return a sorted list of all previously ingested project names."""
    if not VECTOR_STORE.exists():
        return {"projects": []}
    projects = sorted(p.name for p in VECTOR_STORE.iterdir() if p.is_dir())
    return {"projects": projects}


@app.post("/ingest", response_model=IngestResponse)
def ingest_project(request: IngestRequest):
    """
    Index a project directory into the vector store.
    Run this once per project before querying.
    """
    try:
        ingest(request.project_path)
        project_root = Path(request.project_path).resolve()
        project_name = project_root.name

        # Persist project root so review_file can recover the absolute path later
        meta_path = VECTOR_STORE / project_name / "_rex_meta.json"
        meta_path.write_text(
            json.dumps({"project_root": str(project_root)}),
            encoding="utf-8",
        )

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


@app.post("/stream")
def stream_query(request: StreamRequest):
    """
    Ask a question and receive the answer as a Server-Sent Events stream.

    Each SSE message carries a JSON payload:
      {"type": "token",  "content": "<text>"}      — one streamed token
      {"type": "done",   "session_id": "...",
                         "sources":   [...]}        — stream finished
      {"type": "error",  "detail": "<msg>"}         — something went wrong

    The existing /query endpoint is unchanged.
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

    def event_generator():
        try:
            for token in query_stream(
                question=request.question,
                session=session,
                n_results=request.n_results,
                model=request.model,
            ):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            # Stream complete — session was saved inside query_stream
            sessions[session.session_id] = session

            from retriever import retrieve
            raw_sources = retrieve(request.question, request.project_name, request.n_results)
            sources = [
                {
                    "filepath":   s["filepath"],
                    "start_line": s["start_line"],
                    "end_line":   s["end_line"],
                    "score":      s["score"],
                }
                for s in raw_sources
            ]
            yield f"data: {json.dumps({'type': 'done', 'session_id': session.session_id, 'sources': sources})}\n\n"

        except FileNotFoundError as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/review", response_model=QueryResponse)
def review_file_endpoint(request: ReviewRequest):
    """
    Load a full file into context and return a structured code review.
    Does not use RAG — the entire file is passed directly to the model.
    """
    allowed_models = {CODE_MODEL, REASONING_MODEL}
    if request.model not in allowed_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from: {allowed_models}"
        )

    session = sessions.get(request.session_id)
    if session is None:
        session = load_session(request.session_id)
    if session is None:
        session = new_session(request.project_name, session_id=request.session_id)

    try:
        answer, session = review_file(
            filepath=request.filepath,
            project_name=request.project_name,
            session=session,
            model=request.model,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sessions[session.session_id] = session

    return QueryResponse(
        answer=answer,
        session_id=session.session_id,
        model_used=request.model,
        sources=[],
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
