import pytest
from pathlib import Path
from ingest import chunk_file, scan_files
from config import CHUNK_SIZE, CHUNK_OVERLAP


def test_chunk_file_splits_large_file(tmp_path):
    """A file with more lines than CHUNK_SIZE must produce multiple chunks."""
    f = tmp_path / "big.py"
    f.write_text("\n".join(f"line {i}" for i in range(CHUNK_SIZE + 20)))
    chunks = chunk_file(f, tmp_path)
    assert len(chunks) > 1


def test_chunk_file_empty_file_returns_empty(tmp_path):
    """An empty file must produce no chunks."""
    f = tmp_path / "empty.py"
    f.write_text("")
    assert chunk_file(f, tmp_path) == []


def test_chunk_file_small_file_returns_one_chunk(tmp_path):
    """A file smaller than CHUNK_SIZE must produce exactly one chunk."""
    f = tmp_path / "small.py"
    f.write_text("\n".join(f"line {i}" for i in range(10)))
    chunks = chunk_file(f, tmp_path)
    assert len(chunks) == 1


def test_chunk_file_filepath_is_relative_to_root(tmp_path):
    """chunk filepath must be relative to project_root, not absolute."""
    sub = tmp_path / "src"
    sub.mkdir()
    f = sub / "module.py"
    f.write_text("x = 1\n")
    chunks = chunk_file(f, tmp_path)
    assert len(chunks) == 1
    assert chunks[0]["filepath"] == str(Path("src") / "module.py")
    assert not Path(chunks[0]["filepath"]).is_absolute()


def test_chunk_file_overlap_produces_correct_start_lines(tmp_path):
    """Second chunk must start before the first chunk ends (overlap)."""
    total_lines = CHUNK_SIZE + 5
    f = tmp_path / "overlap.py"
    f.write_text("\n".join(f"line {i}" for i in range(total_lines)))
    chunks = chunk_file(f, tmp_path)
    assert len(chunks) >= 2
    first_end = chunks[0]["end_line"]
    second_start = chunks[1]["start_line"]
    assert second_start <= first_end  # overlap means second chunk starts before first ends


def test_chunk_file_chunk_contains_correct_content(tmp_path):
    """First chunk must contain the first line of the file."""
    f = tmp_path / "content.py"
    f.write_text("def hello(): pass\n" + "\n".join(f"# line {i}" for i in range(5)))
    chunks = chunk_file(f, tmp_path)
    assert "def hello(): pass" in chunks[0]["text"]


def test_chunk_file_unreadable_returns_empty(tmp_path):
    """chunk_file must return [] and not raise when a file cannot be read."""
    f = tmp_path / "ghost.py"
    # Pass a path that doesn't exist — read_text will raise, should be caught
    result = chunk_file(f, tmp_path)
    assert result == []


def test_chunk_file_line_numbers_are_one_indexed(tmp_path):
    """start_line must be 1-indexed (first chunk starts at line 1)."""
    f = tmp_path / "lines.py"
    f.write_text("a = 1\nb = 2\n")
    chunks = chunk_file(f, tmp_path)
    assert chunks[0]["start_line"] == 1


def test_scan_files_excludes_hidden_dirs(tmp_path):
    """Files inside excluded dirs like .git must not be returned."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("data")
    src = tmp_path / "main.py"
    src.write_text("pass")
    files = scan_files(tmp_path)
    paths = [str(f) for f in files]
    assert not any(".git" in p for p in paths)


def test_scan_files_excludes_non_allowed_extensions(tmp_path):
    """Binary or unknown file types must be excluded."""
    (tmp_path / "binary.exe").write_bytes(b"\x00\x01\x02")
    (tmp_path / "script.py").write_text("pass")
    files = scan_files(tmp_path)
    names = [f.name for f in files]
    assert "binary.exe" not in names
    assert "script.py" in names


def test_scan_files_excludes_env_files(tmp_path):
    """Secrets files like .env must never be returned."""
    (tmp_path / ".env").write_text("SECRET=abc")
    (tmp_path / "app.py").write_text("pass")
    files = scan_files(tmp_path)
    names = [f.name for f in files]
    assert ".env" not in names
