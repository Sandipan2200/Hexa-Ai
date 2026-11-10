"""State machine for Hexa voice assistant."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from hexa.events import Event, EventType, get_event_bus

logger = logging.getLogger(__name__)


class State(str, Enum):
    """Hexa state machine states."""
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"


# Valid state transitions
VALID_TRANSITIONS: dict[State, set[State]] = {
    State.IDLE: {State.LISTENING},
    State.LISTENING: {State.TRANSCRIBING, State.IDLE},
    State.TRANSCRIBING: {State.THINKING, State.IDLE},
    State.THINKING: {State.SPEAKING, State.IDLE},
    State.SPEAKING: {State.IDLE, State.LISTENING},  # Can interrupt to listen
}


@dataclass
class StateMachine:
    """Manages Hexa's state transitions and emits events."""
    current_state: State = State.IDLE
    _callbacks: list[Callable[[State, State], None]] = None

    def __post_init__(self) -> None:
        if self._callbacks is None:
            self._callbacks = []

    def add_callback(self, callback: Callable[[State, State], None]) -> None:
        """Add a callback for state changes."""
        self._callbacks.append(callback)

    def can_transition(self, new_state: State) -> bool:
        """Check if transition is valid."""
        return new_state in VALID_TRANSITIONS.get(self.current_state, set())

    def transition(self, new_state: State) -> bool:
        """Transition to a new state, emitting events."""
        if not self.can_transition(new_state):
            logger.warning("Invalid transition: %s -> %s", self.current_state, new_state)
            return False

        old_state = self.current_state
        self.current_state = new_state
        logger.info("State transition: %s -> %s", old_state, new_state)

        # Emit event
        get_event_bus().emit(Event(
            type=EventType.STATE_CHANGE,
            data={"old_state": old_state.value, "new_state": new_state.value}
        ))

        # Call callbacks
        for callback in self._callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.exception("Error in state callback: %s", e)

        return True

    def force_idle(self) -> None:
        """Force transition to IDLE from any state."""
        if self.current_state != State.IDLE:
            self.current_state = State.IDLE
            get_event_bus().emit(Event(
                type=EventType.STATE_CHANGE,
                data={"old_state": "forced", "new_state": State.IDLE.value}
            ))


# Global state machine instance
_state_machine: StateMachine | None = None


def get_state_machine() -> StateMachine:
    """Get the global state machine instance."""
    global _state_machine
    if _state_machine is None:
        _state_machine = StateMachine()
    return _state_machine