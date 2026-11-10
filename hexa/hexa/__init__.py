"""Hexa - Offline Voice Assistant Core."""
from __future__ import annotations

__version__ = "0.1.0"

# Import tools to register them
from hexa.tools import apps, browser, system, notes, timers  # noqa: F401

from hexa.config import get_config, load_config
from hexa.events import Event, EventType, get_event_bus
from hexa.llm import get_llm
from hexa.memory import get_memory
from hexa.router import get_router
from hexa.state import State, StateMachine, get_state_machine
from hexa.tools import get_registry

__all__ = [
    "get_config",
    "load_config",
    "get_event_bus",
    "Event",
    "EventType",
    "get_llm",
    "get_memory",
    "get_router",
    "State",
    "StateMachine",
    "get_state_machine",
    "get_registry",
]