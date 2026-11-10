"""Memory tools (remember/recall)."""
from __future__ import annotations

import logging
from hexa.tools import tool
from hexa.memory import get_memory

logger = logging.getLogger(__name__)


@tool
def remember(text: str) -> str:
    """Remember a note or fact for later recall."""
    memory = get_memory()
    memory_id = memory.remember(text)
    logger.info("Remembered [%d]: %s", memory_id, text[:50])
    return f"Remembered (id: {memory_id})"


@tool
def recall(query: str) -> str:
    """Recall notes/facts matching a query."""
    memory = get_memory()
    results = memory.recall(query, limit=5)

    if not results:
        return "Nothing found."

    lines = [f"[{r['id']}] {r['content']}" for r in results]
    return "Found:\n" + "\n".join(lines)