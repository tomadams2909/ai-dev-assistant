import pytest
from retriever import load_collection


def test_load_collection_raises_for_missing_project():
    """load_collection must raise FileNotFoundError for a project that has never been indexed."""
    with pytest.raises(FileNotFoundError, match="No index found"):
        load_collection("project-that-does-not-exist-xyz")
