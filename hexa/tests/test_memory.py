"""Tests for memory module."""
from __future__ import annotations

import tempfile
import time
import pytest

from hexa.memory import Memory


@pytest.fixture
def memory() -> Memory:
    """Create a temporary memory database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Memory(f"{tmpdir}/test.db")


def test_remember_recall(memory: Memory) -> None:
    """Test basic remember and recall."""
    mid = memory.remember("My exam is on Friday")
    assert mid == 1

    results = memory.recall("exam")
    assert len(results) == 1
    assert results[0]["content"] == "My exam is on Friday"
    assert results[0]["id"] == 1


def test_recall_multiple(memory: Memory) -> None:
    """Test recall with multiple matches."""
    memory.remember("Exam on Friday")
    memory.remember("Meeting on Monday")
    memory.remember("Dentist on Friday")

    results = memory.recall("Friday")
    assert len(results) == 2
    # Most recent first
    assert results[0]["content"] == "Dentist on Friday"
    assert results[1]["content"] == "Exam on Friday"


def test_recall_limit(memory: Memory) -> None:
    """Test recall limit parameter."""
    for i in range(15):
        memory.remember(f"Note {i}")

    results = memory.recall("Note", limit=5)
    assert len(results) == 5


def test_get_all(memory: Memory) -> None:
    """Test get_all returns all memories."""
    memory.remember("First")
    time.sleep(0.01)
    memory.remember("Second")
    time.sleep(0.01)
    memory.remember("Third")

    results = memory.get_all()
    assert len(results) == 3
    assert results[0]["content"] == "Third"
    assert results[1]["content"] == "Second"
    assert results[2]["content"] == "First"


def test_delete(memory: Memory) -> None:
    """Test deleting memories."""
    mid = memory.remember("To delete")
    assert memory.delete(mid) is True
    assert memory.delete(mid) is False  # Already deleted
    results = memory.recall("delete")
    assert len(results) == 0


def test_tags(memory: Memory) -> None:
    """Test tags are stored."""
    mid = memory.remember("Tagged note", tags=["work", "important"])
    results = memory.recall("Tagged")
    assert results[0]["tags"] == ["work", "important"]