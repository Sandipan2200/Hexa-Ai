"""Tiny in-process event bus for Hexa."""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Event types emitted by Hexa."""
    STATE_CHANGE = "state_change"
    TRANSCRIPT = "transcript"
    REPLY = "reply"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    AUDIO_LEVEL = "audio_level"
    WAKE_DETECTED = "wake_detected"
    HOTKEY_PRESSED = "hotkey_pressed"
    INTERRUPT = "interrupt"


@dataclass
class Event:
    """Event emitted by Hexa components."""
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


Callback = Callable[[Event], Any]


class EventBus:
    """Simple pub/sub event bus with sync and async support."""

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[Callback]] = defaultdict(list)
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._history: list[Event] = []
        self._max_history = 1000

    def subscribe(self, event_type: EventType, callback: Callback) -> None:
        """Subscribe to an event type."""
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callback) -> None:
        """Unsubscribe from an event type."""
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)

    def emit(self, event: Event) -> None:
        """Emit an event synchronously to all subscribers."""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        for callback in self._subscribers[event.type]:
            try:
                callback(event)
            except Exception as e:
                logger.exception("Error in event callback: %s", e)

        # Also push to async queue for WebSocket forwarding
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    async def emit_async(self, event: Event) -> None:
        """Emit an event asynchronously."""
        self.emit(event)

    async def get_event(self) -> Event:
        """Get the next event from the queue (for WebSocket)."""
        return await self._queue.get()

    def get_history(self, event_type: EventType | None = None, limit: int = 100) -> list[Event]:
        """Get recent event history."""
        if event_type is None:
            return self._history[-limit:]
        return [e for e in self._history if e.type == event_type][-limit:]


# Global event bus instance
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def emit_event(event_type: EventType, **data: Any) -> None:
    """Convenience function to emit an event."""
    get_event_bus().emit(Event(type=event_type, data=data))