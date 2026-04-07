import pytest
from pathlib import Path
from tools.file_reader import read_full_file


def test_path_traversal_blocked(tmp_path):
    """Files outside the project root must be rejected."""
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("secret")

    with pytest.raises((ValueError, PermissionError, Exception)) as exc_info:
        read_full_file("../secret.txt", str(tmp_path))

    assert exc_info.value is not None


def test_valid_file_within_root(tmp_path):
    """Files inside the project root must be readable."""
    test_file = tmp_path / "main.py"
    test_file.write_text("def hello(): pass")

    content = read_full_file("main.py", str(tmp_path))
    assert "hello" in content


def test_nonexistent_file_raises(tmp_path):
    """Missing files must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_full_file("doesnotexist.py", str(tmp_path))
