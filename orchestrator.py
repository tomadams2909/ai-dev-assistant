# orchestrator.py
import json
import re
from pathlib import Path
from typing import Iterator
from retriever import retrieve
from config import CODE_MODEL, VECTOR_STORE
from models.provider import get_provider
from memory import Session, build_messages, trim_history, save_session

# ── System prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are REX (Repository Engineering eXpert), an expert developer assistant with access to the user's codebase.

Your behaviour:
- Answer questions using ONLY the code context provided to you
- Always cite the filename and line number your answer comes from
- If the answer isn't in the provided context, say so clearly — do not guess
- When explaining code, be concise and precise
- If asked to suggest a change, explain what to change and why, but do not apply it yet
- Always wrap code in fenced code blocks with an explicit language tag. Example: \`\`\`python. Never use a bare \`\`\` fence with no language.

You are not a general chatbot. You are a codebase assistant."""


# ── Context builder ───────────────────────────────────────────────
def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block."""
    if not chunks:
        return "No relevant code found in the index."

    context = ""
    for chunk in chunks:
        context += f"\n--- {chunk['filepath']} (lines {chunk['start_line']}–{chunk['end_line']}) ---\n"
        context += chunk["text"] + "\n"
    return context


# ── Think-tag stripper for streaming ─────────────────────────────
def _strip_think_stream(token_stream: Iterator[str]) -> Iterator[str]:
    """
    Filter <think>…</think> blocks out of a streaming token iterator.

    DeepSeek-r1 emits chain-of-thought inside <think> tags before the
    answer.  We buffer until we know whether the stream starts with
    <think>, discard everything inside that block, then yield the rest
    token-by-token with no extra latency.
    """
    OPEN  = "<think>"
    CLOSE = "</think>"

    buf      = ""
    in_think = False
    decided  = False

    for token in token_stream:
        if not decided:
            buf += token
            if buf.startswith(OPEN):
                in_think = True
                decided  = True
                buf      = buf[len(OPEN):]
            elif len(buf) >= len(OPEN) or not OPEN.startswith(buf):
                # Buffer is long enough — no think tag is coming
                decided  = True
                in_think = False
                yield buf
                buf = ""
            # else: still accumulating — could still be start of <think>
        elif in_think:
            buf += token
            idx = buf.find(CLOSE)
            if idx != -1:
                in_think = False
                buf = buf[idx + len(CLOSE):].lstrip("\n ")
                if buf:
                    yield buf
                    buf = ""
        else:
            yield token

    # Flush anything left after the loop (only if we're not mid-think)
    if buf and not in_think:
        yield buf


# ── Main query function ───────────────────────────────────────────
def query(
    question: str,
    session: Session,
    n_results: int = 5,
    model: str = CODE_MODEL,
) -> tuple[str, Session]:
    """
    Ask a question about a project using RAG + session memory.

    Retrieved code chunks are injected into the current turn's prompt only —
    they are never stored in session.history, keeping the context window lean
    across long sessions.

    Args:
        question:  The user's question
        session:   Session object (carries history, summary, project name)
        n_results: How many chunks to retrieve from the vector store
        model:     Ollama model to use for inference

    Returns:
        answer:  The model's response (think tags stripped)
        session: Updated session (history trimmed, persisted to disk)
    """
    chunks  = retrieve(question, session.project_name, n_results)
    context = build_context(chunks)

    # Full user message for the model: context + question.
    # This is NOT what gets stored in history — only the clean question is.
    user_message_with_context = f"Codebase context:\n{context}\n\nQuestion: {question}"

    # Build messages from session memory, then append the context-rich turn
    messages = build_messages(session)
    messages.append({"role": "user", "content": user_message_with_context})

    raw_answer = get_provider(model=model).chat(SYSTEM_PROMPT, messages)

    # deepseek-r1 wraps chain-of-thought in <think>...</think> — strip it
    answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()

    # Store only the clean Q&A — never code chunks
    session.history.append({"role": "user",      "content": question})
    session.history.append({"role": "assistant",  "content": answer})
    session.full_log.append({"role": "user",      "content": question})
    session.full_log.append({"role": "assistant",  "content": answer})

    trim_history(session)
    save_session(session)

    return answer, session


# ── Streaming query ───────────────────────────────────────────────
def query_stream(
    question: str,
    session: Session,
    n_results: int = 5,
    model: str = CODE_MODEL,
) -> Iterator[str]:
    """
    Streaming variant of query().

    Yields clean response tokens one by one as they arrive from Ollama.
    <think> blocks are stripped in real time before any token reaches
    the caller.

    Session history is updated and persisted to disk after the last
    token has been yielded (i.e. after the caller exhausts this
    generator).
    """
    chunks  = retrieve(question, session.project_name, n_results)
    context = build_context(chunks)

    user_message_with_context = f"Codebase context:\n{context}\n\nQuestion: {question}"

    messages = build_messages(session)
    messages.append({"role": "user", "content": user_message_with_context})

    raw_stream = get_provider(model=model).chat_stream(SYSTEM_PROMPT, messages)

    accumulated: list[str] = []
    for token in _strip_think_stream(raw_stream):
        accumulated.append(token)
        yield token

    # Persist session after stream completes
    answer = "".join(accumulated)
    session.history.append({"role": "user",      "content": question})
    session.history.append({"role": "assistant",  "content": answer})
    session.full_log.append({"role": "user",      "content": question})
    session.full_log.append({"role": "assistant",  "content": answer})
    trim_history(session)
    save_session(session)


# ── File review ───────────────────────────────────────────────────
def review_file(
    filepath: str,
    project_name: str,
    session: Session,
    model: str = CODE_MODEL,
) -> tuple[str, Session]:
    """
    Load an entire file into context and run a structured code review.

    Does NOT use RAG retrieval — the full file is passed directly to the model.
    Only a clean summary is stored in session history, never the raw file contents.

    Raises:
        FileNotFoundError: Project metadata or target file not found.
        ValueError:        File exceeds MAX_FILE_TOKENS or path traversal detected.
    """
    from tools.file_reader import read_full_file, estimate_tokens, MAX_FILE_TOKENS

    # Recover the original project root from metadata saved at ingest time
    meta_path = VECTOR_STORE / project_name / "_rex_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No ingest metadata found for project '{project_name}'. "
            f"Re-index the project via the sidebar to enable file review."
        )

    meta         = json.loads(meta_path.read_text(encoding="utf-8"))
    project_root = meta["project_root"]

    file_contents   = read_full_file(filepath, project_root)
    token_estimate  = estimate_tokens(file_contents)

    if token_estimate > MAX_FILE_TOKENS:
        raise ValueError(
            f"'{filepath}' is too large for local review "
            f"(~{token_estimate:,} estimated tokens; limit is {MAX_FILE_TOKENS:,}). "
            f"Use God Mode for large files."
        )

    review_prompt = (
        f"Please review the following file and provide a structured analysis.\n\n"
        f"File: {filepath}\n\n"
        f"```\n{file_contents}\n```\n\n"
        f"Provide your analysis in these four sections — "
        f"cite line numbers wherever possible:\n\n"
        f"## Bugs\n"
        f"Any bugs, logic errors, or incorrect behaviour.\n\n"
        f"## Security Issues\n"
        f"Any vulnerabilities such as injection, path traversal, secrets exposure, etc.\n\n"
        f"## Code Quality\n"
        f"Maintainability, readability, or structural concerns.\n\n"
        f"## Improvement Suggestions\n"
        f"Concrete suggestions to improve the code.\n\n"
        f"If a section has no findings, write \"None identified.\""
    )

    messages = build_messages(session)
    messages.append({"role": "user", "content": review_prompt})

    raw_answer = get_provider(model=model).chat(SYSTEM_PROMPT, messages)
    answer     = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()

    # Store only a clean label — never the full file contents
    clean_label = f"[File review] {filepath}"
    session.history.append({"role": "user",      "content": clean_label})
    session.history.append({"role": "assistant",  "content": answer})
    session.full_log.append({"role": "user",      "content": clean_label})
    session.full_log.append({"role": "assistant",  "content": answer})

    trim_history(session)
    save_session(session)

    return answer, session


# ── CLI entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from memory import new_session

    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py <project-name>")
        print("Example: python orchestrator.py insurance-platform")
        sys.exit(1)

    project_name = sys.argv[1]
    session      = new_session(project_name)

    print(f"\nREX ready. Querying project: {project_name}")
    print("Type 'exit' to quit, 'clear' to reset conversation history.\n")

    while True:
        try:
            question = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() == "exit":
            print("Goodbye.")
            break
        if question.lower() == "clear":
            session = new_session(project_name)
            print("History cleared.\n")
            continue

        print("\nREX: thinking...\n")
        answer, session = query(question, session)
        print(f"REX: {answer}\n")
