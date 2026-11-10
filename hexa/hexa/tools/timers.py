"""Timer tools."""
from __future__ import annotations

import logging
import threading
import time
from hexa.tools import tool
from hexa.events import emit_event, EventType

logger = logging.getLogger(__name__)


@tool
def set_timer(seconds: int, label: str = "") -> str:
    """Set a timer that speaks when done."""
    if seconds <= 0:
        return "Timer must be positive seconds."

    def timer_done() -> None:
        time.sleep(seconds)
        msg = f"Timer done{f' for {label}' if label else ''}"
        logger.info("Timer fired: %s", msg)
        emit_event(EventType.REPLY, text=msg, source="timer")

    thread = threading.Thread(target=timer_done, daemon=True)
    thread.start()

    label_str = f" for {label}" if label else ""
    return f"Timer set for {seconds} seconds{label_str}"


@tool
def dummy_destructive() -> str:
    """Dummy tool for testing confirm mechanism (permission: confirm)."""
    return "This would be destructive but it's just a test."


@tool
def dummy_blocked() -> str:
    """Dummy tool for testing blocked mechanism (permission: blocked)."""
    return "This tool is blocked."