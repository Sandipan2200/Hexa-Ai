"""Hexa - Offline Voice Assistant Core."""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# STRICT OFFLINE GUARANTEE (set BEFORE any Hugging Face-using library imports).
#
# faster-whisper / transformers / huggingface_hub normally ping
# huggingface.co on model load (e.g. ".../faster-whisper-small.en/revision/main").
# These variables force fully-local cache loading so Hexa never touches the
# network at runtime. All models (Whisper, Silero VAD, openWakeWord, Piper)
# are read from local files. See README.md "Offline operation".
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

__version__ = "0.1.0"

# Import tools to register them
from hexa.tools import apps, browser, system, notes, timers  # noqa: F401

from hexa.audio import AudioStream, get_audio_stream, shutdown_audio_stream
from hexa.config import get_config, load_config
from hexa.events import Event, EventType, get_event_bus
from hexa.llm import get_llm
from hexa.memory import get_memory
from hexa.router import get_router
from hexa.state import State, StateMachine, get_state_machine
from hexa.stt import SpeechRecognizer, STTConfig, VADConfig, SileroVAD, WhisperTranscriber
from hexa.tools import get_registry
from hexa.tts import TTSManager, TTSConfig
from hexa.voice import VoicePipeline, get_voice_pipeline, shutdown_voice_pipeline
from hexa.wake import WakeWordDetector, WakeWordConfig, HotkeyListener, HotkeyConfig

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
    "AudioStream",
    "get_audio_stream",
    "shutdown_audio_stream",
    "SpeechRecognizer",
    "STTConfig",
    "VADConfig",
    "SileroVAD",
    "WhisperTranscriber",
    "TTSManager",
    "TTSConfig",
    "VoicePipeline",
    "get_voice_pipeline",
    "shutdown_voice_pipeline",
    "WakeWordDetector",
    "WakeWordConfig",
    "HotkeyListener",
    "HotkeyConfig",
]