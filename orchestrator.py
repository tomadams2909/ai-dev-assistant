import ollama
from retriever import retrieve
from config import CODE_MODEL
from models.provider import get_provider

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
    project_name: str,
    history: list[dict] = None,
    n_results: int = 5
) -> tuple[str, list[dict]]:
    """
    Ask a question about a project.

    Args:
        question:     The user's question
        project_name: Which project index to search
        history:      Conversation history for multi-turn context
        n_results:    How many chunks to retrieve

    Returns:
        answer:       The AI's response string
        history:      Updated conversation history
    """
    if history is None:
        history = []

    chunks       = retrieve(question, project_name, n_results)
    context      = build_context(chunks)
    user_message = f"""Codebase context:
{context}

Question: {question}"""

    history.append({"role": "user", "content": user_message})

    response = ollama.chat(
        model=CODE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *history
        ]
    )

    answer = response["message"]["content"]
    history.append({"role": "assistant", "content": answer})

    return answer, history


# ── CLI entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py <project-name>")
        print("Example: python orchestrator.py insurance-platform")
        sys.exit(1)

    project_name = sys.argv[1]
    history      = []

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
            history = []
            print("History cleared.\n")
            continue

        print("\nREX: thinking...\n")
        answer, history = query(question, project_name, history)
        print(f"REX: {answer}\n")