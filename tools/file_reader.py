# tools/file_reader.py
from pathlib import Path

MAX_FILE_TOKENS = 25000


def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return len(text) // 4


def read_full_file(filepath: str, project_root: str) -> str:
    """
    Read the full contents of a file, resolved safely against project_root.

    Raises:
        ValueError:       If the resolved path escapes the project root (traversal attempt).
        FileNotFoundError: If the file does not exist.
        ValueError:       If the path points to a directory, not a file.
    """
    root   = Path(project_root).resolve()
    target = (root / filepath).resolve()

    # Block path traversal — target must be inside root
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError(
            f"Access denied: '{filepath}' resolves outside the project root. "
            f"Only files within the indexed project may be reviewed."
        )

    if not target.exists():
        raise FileNotFoundError(
            f"File not found: '{filepath}'. "
            f"Check the path is relative to the project root and the file exists."
        )

    if not target.is_file():
        raise ValueError(f"'{filepath}' is a directory, not a file.")

    return target.read_text(encoding="utf-8", errors="ignore")
