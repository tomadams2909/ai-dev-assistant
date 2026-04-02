# ingest.py
import sys
import ollama
import chromadb
from pathlib import Path
from config import (
    EMBEDDING_MODEL,
    VECTOR_STORE,
    ALLOWED_EXTENSIONS,
    EXCLUDED_DIRS,
    EXCLUDED_FILES,
    is_allowed
)

# ── Chunking ──────────────────────────────────────────────────────
def chunk_file(filepath: Path, project_root: Path) -> list[dict]:
    """Split a file into overlapping line-based chunks."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"  Skipping {filepath.name}: {e}")
        return []

    lines = content.splitlines()
    if not lines:
        return []

    chunks    = []
    chunk_size = 60   # lines per chunk
    overlap    = 10   # lines shared between consecutive chunks

    for i in range(0, len(lines), chunk_size - overlap):
        chunk_lines = lines[i : i + chunk_size]
        chunk_text  = "\n".join(chunk_lines).strip()
        if not chunk_text:
            continue

        chunks.append({
            "text":       chunk_text,
            "filepath":   str(filepath.relative_to(project_root)),
            "start_line": i + 1,
            "end_line":   i + len(chunk_lines),
        })

    return chunks


# ── Embedding ─────────────────────────────────────────────────────
def embed(text: str) -> list[float]:
    """Convert text to a vector using Ollama locally."""
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]


# ── File scanner ──────────────────────────────────────────────────
def scan_files(project_root: Path) -> list[Path]:
    """Return all allowed files under project_root."""
    return [
        p for p in project_root.rglob("*")
        if p.is_file() and is_allowed(p)
    ]


# ── Main ingestion ────────────────────────────────────────────────
def ingest(project_path: str):
    project_root = Path(project_path).resolve()

    if not project_root.exists():
        print(f"Path does not exist: {project_root}")
        sys.exit(1)

    # Each project gets its own ChromaDB collection named after the folder
    project_name = project_root.name
    print(f"\nIngesting project: {project_name}")
    print(f"Root: {project_root}")

    client     = chromadb.PersistentClient(path=str(VECTOR_STORE / project_name))
    collection = client.get_or_create_collection(project_name)

    files = scan_files(project_root)
    print(f"Found {len(files)} files to index\n")

    total_chunks = 0

    for filepath in files:
        chunks = chunk_file(filepath, project_root)
        if not chunks:
            continue

        print(f"  Indexing {filepath.relative_to(project_root)} → {len(chunks)} chunks")

        for chunk in chunks:
            doc_id = f"{chunk['filepath']}:{chunk['start_line']}"

            collection.upsert(
                ids        = [doc_id],
                embeddings = [embed(chunk["text"])],
                documents  = [chunk["text"]],
                metadatas  = [{
                    "filepath":   chunk["filepath"],
                    "start_line": chunk["start_line"],
                    "end_line":   chunk["end_line"],
                }]
            )
            total_chunks += 1

    print(f"\nDone. {total_chunks} chunks indexed for '{project_name}'")
    print(f"Stored at: {VECTOR_STORE / project_name}")


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path-to-project>")
        print("Example: python ingest.py C:\\tom\\repos\\insurance-platform")
        sys.exit(1)

    ingest(sys.argv[1])