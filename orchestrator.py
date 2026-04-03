# orchestrator.py
import re
from retriever import retrieve
from config import CODE_MODEL
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
