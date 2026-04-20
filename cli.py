# cli.py — interactive terminal interface for REX
import sys
from orchestrator import query
from memory import new_session


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python cli.py <project-name>")
        print("Example: python cli.py insurance-platform")
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


if __name__ == "__main__":
    main()
