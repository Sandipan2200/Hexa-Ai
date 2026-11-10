"""Tests for event bus and state machine."""
from __future__ import annotations

import pytest

from hexa.events import EventBus, Event, EventType
from hexa.state import StateMachine, State


def test_event_bus_subscribe_emit() -> None:
    """Test basic pub/sub."""
    bus = EventBus()
    received = []

    def callback(event: Event) -> None:
        received.append(event)

    bus.subscribe(EventType.STATE_CHANGE, callback)
    bus.emit(Event(type=EventType.STATE_CHANGE, data={"old": "idle", "new": "listening"}))

    assert len(received) == 1
    assert received[0].type == EventType.STATE_CHANGE
    assert received[0].data == {"old": "idle", "new": "listening"}


def test_event_bus_multiple_subscribers() -> None:
    """Test multiple subscribers receive events."""
    bus = EventBus()
    received1 = []
    received2 = []

    bus.subscribe(EventType.TRANSCRIPT, lambda e: received1.append(e))
    bus.subscribe(EventType.TRANSCRIPT, lambda e: received2.append(e))

    bus.emit(Event(type=EventType.TRANSCRIPT, data={"text": "hello"}))

    assert len(received1) == 1
    assert len(received2) == 1


def test_event_bus_unsubscribe() -> None:
    """Test unsubscribing."""
    bus = EventBus()
    received = []

    def callback(event: Event) -> None:
        received.append(event)

    bus.subscribe(EventType.ERROR, callback)
    bus.emit(Event(type=EventType.ERROR, data={}))
    assert len(received) == 1

    bus.unsubscribe(EventType.ERROR, callback)
    bus.emit(Event(type=EventType.ERROR, data={}))
    assert len(received) == 1  # No new events


def test_event_history() -> None:
    """Test event history is maintained."""
    bus = EventBus()

    for i in range(5):
        bus.emit(Event(type=EventType.AUDIO_LEVEL, data={"rms": i}))

    history = bus.get_history(EventType.AUDIO_LEVEL, limit=3)
    assert len(history) == 3
    assert history[-1].data["rms"] == 4


def test_state_machine_initial() -> None:
    """Test initial state is IDLE."""
    sm = StateMachine()
    assert sm.current_state == State.IDLE


def test_state_machine_valid_transitions() -> None:
    """Test valid state transitions."""
    sm = StateMachine()

    # IDLE -> LISTENING
    assert sm.transition(State.LISTENING) is True
    assert sm.current_state == State.LISTENING

    # LISTENING -> TRANSCRIBING
    assert sm.transition(State.TRANSCRIBING) is True
    assert sm.current_state == State.TRANSCRIBING

    # TRANSCRIBING -> THINKING
    assert sm.transition(State.THINKING) is True
    assert sm.current_state == State.THINKING

    # THINKING -> SPEAKING
    assert sm.transition(State.SPEAKING) is True
    assert sm.current_state == State.SPEAKING

    # SPEAKING -> IDLE
    assert sm.transition(State.IDLE) is True
    assert sm.current_state == State.IDLE


def test_state_machine_invalid_transitions() -> None:
    """Test invalid transitions are rejected."""
    sm = StateMachine()

    # IDLE -> TRANSCRIBING (invalid)
    assert sm.transition(State.TRANSCRIBING) is False
    assert sm.current_state == State.IDLE

    # LISTENING -> SPEAKING (invalid)
    sm.transition(State.LISTENING)
    assert sm.transition(State.SPEAKING) is False
    assert sm.current_state == State.LISTENING


def test_state_machine_interrupt() -> None:
    """Test SPEAKING -> LISTENING (interrupt)."""
    sm = StateMachine()
    sm.transition(State.LISTENING)
    sm.transition(State.TRANSCRIBING)
    sm.transition(State.THINKING)
    sm.transition(State.SPEAKING)

    # Can interrupt from SPEAKING to LISTENING
    assert sm.transition(State.LISTENING) is True
    assert sm.current_state == State.LISTENING


def test_state_machine_force_idle() -> None:
    """Test force_idle from any state."""
    sm = StateMachine()
    sm.transition(State.LISTENING)
    sm.transition(State.TRANSCRIBING)
    sm.force_idle()
    assert sm.current_state == State.IDLE


def test_state_machine_callbacks() -> None:
    """Test state change callbacks."""
    sm = StateMachine()
    transitions = []

    def callback(old: State, new: State) -> None:
        transitions.append((old, new))

    sm.add_callback(callback)
    sm.transition(State.LISTENING)
    sm.transition(State.TRANSCRIBING)

    assert len(transitions) == 2
    assert transitions[0] == (State.IDLE, State.LISTENING)
    assert transitions[1] == (State.LISTENING, State.TRANSCRIBING)


def test_state_machine_emits_events() -> None:
    """Test state machine emits events on transition."""
    from hexa.events import get_event_bus
    bus = get_event_bus()
    received = []

    def callback(event: Event) -> None:
        received.append(event)

    bus.subscribe(EventType.STATE_CHANGE, callback)

    sm = StateMachine()
    sm.transition(State.LISTENING)

    assert len(received) == 1
    assert received[0].type == EventType.STATE_CHANGE
    assert received[0].data["old_state"] == "idle"
    assert received[0].data["new_state"] == "listening"