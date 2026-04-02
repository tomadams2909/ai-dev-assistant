# retriever.py
import ollama
import chromadb
from pathlib import Path
from config import EMBEDDING_MODEL, VECTOR_STORE


# ── Load a project's collection ───────────────────────────────────
def load_collection(project_name: str):
    """Connect to the ChromaDB index for a given project."""
    store_path = VECTOR_STORE / project_name

    if not store_path.exists():
        raise FileNotFoundError(
            f"No index found for '{project_name}'.\n"
            f"Run: python ingest.py <path-to-{project_name}>"
        )

    client = chromadb.PersistentClient(path=str(store_path))
    return client.get_or_create_collection(project_name)


# ── Semantic search ───────────────────────────────────────────────
def retrieve(query: str, project_name: str, n_results: int = 5) -> list[dict]:
    """
    Embed the query and find the most semantically similar
    chunks in the project index.

    Returns a list of dicts with text, filepath, and line numbers.
    """
    # Convert the question into a vector
    query_embedding = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=query
    )["embedding"]

    collection = load_collection(project_name)

    # Guard against requesting more results than chunks stored
    count = collection.count()
    if count == 0:
        raise ValueError(f"Index for '{project_name}' is empty. Re-run ingest.py.")
    n_results = min(n_results, count)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        chunks.append({
            "text":       doc,
            "filepath":   meta["filepath"],
            "start_line": meta["start_line"],
            "end_line":   meta["end_line"],
            "score":      round(1 - results["distances"][0][i], 3)  # cosine similarity
        })

    return chunks


# ── Debug helper ──────────────────────────────────────────────────
def retrieve_and_print(query: str, project_name: str, n_results: int = 5):
    """Useful for testing retrieval quality from the terminal."""
    print(f"\nQuery: {query}")
    print(f"Project: {project_name}\n")

    chunks = retrieve(query, project_name, n_results)

    for i, chunk in enumerate(chunks):
        print(f"── Result {i + 1} (score: {chunk['score']}) ──────────────")
        print(f"   File: {chunk['filepath']}")
        print(f"   Lines: {chunk['start_line']}–{chunk['end_line']}")
        print(f"   Preview: {chunk['text'][:120].strip()}...")
        print()


# ── Entry point (for testing) ─────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python retriever.py <project-name> <query>")
        print('Example: python retriever.py insurance-platform "where is authentication handled"')
        sys.exit(1)

    project = sys.argv[1]
    query   = " ".join(sys.argv[2:])

    retrieve_and_print(query, project)