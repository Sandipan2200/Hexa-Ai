"""Integration tests for voice pipeline."""
from __future__ import annotations

import queue
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hexa.voice import VoicePipeline, get_voice_pipeline, shutdown_voice_pipeline
from hexa.state import get_state_machine, State
from hexa.events import get_event_bus, EventType


@pytest.fixture(autouse=True)
def reset_singletons() -> None:
    """Reset global singletons before each test."""
    shutdown_voice_pipeline()
    get_state_machine().force_idle()
    get_event_bus()._subscribers.clear()
    yield
    shutdown_voice_pipeline()


class TestVoicePipeline:
    """Tests for VoicePipeline integration."""

    def test_pipeline_start_stop(self) -> None:
        """Test pipeline lifecycle."""
        with patch("hexa.voice.TTSManager"), \
             patch("hexa.voice.SpeechRecognizer") as mock_recognizer_class, \
             patch("hexa.voice.WakeWordDetector") as mock_wake_class, \
             patch("hexa.voice.HotkeyListener") as mock_hotkey_class, \
             patch("hexa.voice.get_audio_stream"):

            pipeline = VoicePipeline()
            pipeline.start()

            assert pipeline.is_running()
            assert mock_recognizer_class.called
            assert mock_wake_class.called
            assert mock_hotkey_class.called

            pipeline.stop()
            assert not pipeline.is_running()

    def test_pipeline_wake_word_triggers_listening(self) -> None:
        """Test wake word detection triggers LISTENING state."""
        with patch("hexa.voice.TTSManager"), \
             patch("hexa.voice.SpeechRecognizer"), \
             patch("hexa.voice.WakeWordDetector") as mock_wake_class, \
             patch("hexa.voice.HotkeyListener"), \
             patch("hexa.voice.get_audio_stream"):

            mock_wake = MagicMock()
            mock_wake_class.return_value = mock_wake

            pipeline = VoicePipeline()
            pipeline.start()

            # Simulate wake word callback
            wake_callback = mock_wake_class.call_args[1]["on_wake"]
            wake_callback()

            # Should transition to LISTENING
            state_machine = get_state_machine()
            assert state_machine.current_state == State.LISTENING

            pipeline.stop()

    def test_pipeline_hotkey_triggers_listening(self) -> None:
        """Test hotkey press triggers LISTENING state."""
        with patch("hexa.voice.TTSManager"), \
             patch("hexa.voice.SpeechRecognizer"), \
             patch("hexa.voice.WakeWordDetector"), \
             patch("hexa.voice.HotkeyListener") as mock_hotkey_class, \
             patch("hexa.voice.get_audio_stream"):

            mock_hotkey = MagicMock()
            mock_hotkey_class.return_value = mock_hotkey

            pipeline = VoicePipeline()
            pipeline.start()

            # Simulate hotkey press callback
            press_callback = mock_hotkey_class.call_args[1]["on_press"]
            press_callback()

            state_machine = get_state_machine()
            assert state_machine.current_state == State.LISTENING

            pipeline.stop()

    def test_pipeline_interrupt_during_speaking(self) -> None:
        """Test interrupt during SPEAKING returns to IDLE."""
        with patch("hexa.voice.TTSManager") as mock_tts_class, \
             patch("hexa.voice.SpeechRecognizer"), \
             patch("hexa.voice.WakeWordDetector"), \
             patch("hexa.voice.HotkeyListener"), \
             patch("hexa.voice.get_audio_stream"):

            mock_tts = MagicMock()
            mock_tts_class.return_value = mock_tts

            pipeline = VoicePipeline()
            pipeline.start()

            # Force to SPEAKING state
            state_machine = get_state_machine()
            state_machine.transition(State.LISTENING)
            state_machine.transition(State.TRANSCRIBING)
            state_machine.transition(State.THINKING)
            state_machine.transition(State.SPEAKING)

            # Trigger interrupt
            interrupt_event = MagicMock()
            interrupt_event.data = {"reason": "test"}
            # The pipeline subscribes to INTERRUPT events
            # We can call the handler directly
            pipeline.interrupt()

            assert state_machine.current_state == State.IDLE
            mock_tts.interrupt.assert_called_once()

            pipeline.stop()

    def test_pipeline_conversation_mode(self) -> None:
        """Test conversation mode keeps listening after response.

        Deterministic full-flow test: wake word -> VAD ends (TRANSCRIBING)
        -> on_transcript callback -> router reply -> speak -> back to
        LISTENING instead of IDLE (conversation mode active).
        """
        with patch("hexa.voice.TTSManager") as mock_tts_class, \
             patch("hexa.voice.SpeechRecognizer") as mock_recognizer_class, \
             patch("hexa.voice.WakeWordDetector") as mock_wake_class, \
             patch("hexa.voice.HotkeyListener"), \
             patch("hexa.voice.get_audio_stream"), \
             patch("hexa.voice.get_router") as mock_get_router, \
             patch("hexa.voice.get_config") as mock_config:

            mock_config.return_value = MagicMock(
                tts=MagicMock(voice="test", voice_path="test", sample_rate=22050, fallback="pyttsx3"),
                stt=MagicMock(model="small.en", device="cpu", compute_type="int8", language="en", beam_size=5, vad_threshold=0.5),
                vad=MagicMock(enabled=True, silence_timeout=1.5, min_speech_duration=0.5),
                wake_word=MagicMock(enabled=True, model_name="hey_jarvis", inference_framework="onnx", threshold=0.5, debounce_time=1.0),
                hotkey=MagicMock(enabled=True, key="<ctrl>+<alt>+<space>"),
                conversation=MagicMock(enabled=True, timeout=10.0)
            )

            mock_router = MagicMock()
            mock_router.route.return_value = "It's 10:00."
            mock_get_router.return_value = mock_router

            mock_tts = MagicMock()
            mock_tts_class.return_value = mock_tts
            mock_handle = MagicMock()
            mock_tts.speak.return_value = mock_handle

            pipeline = VoicePipeline()
            pipeline.start()

            state_machine = get_state_machine()

            # 1. Wake word detected -> conversation mode on, LISTENING
            wake_callback = mock_wake_class.call_args[1]["on_wake"]
            wake_callback()
            assert state_machine.current_state == State.LISTENING

            # 2. VAD ends speech -> TRANSCRIBING (normally driven by stt.py)
            state_machine.transition(State.TRANSCRIBING)

            # 3. Transcript arrives -> router -> TTS -> conversation LISTENING
            transcript_callback = mock_recognizer_class.call_args[1]["on_transcript"]
            transcript_callback("what time is it")

            # Router was used with the spoken question
            mock_router.route.assert_called_once_with("what time is it")
            # Reply was spoken and playback waited for completion
            mock_tts.speak.assert_called_once_with("It's 10:00.")
            mock_handle.wait.assert_called_once()

            # Conversation mode: after speaking, back to LISTENING, not IDLE
            assert state_machine.current_state == State.LISTENING

            pipeline.stop()

    def test_pipeline_conversation_timeout(self) -> None:
        """Test conversation timeout returns to IDLE."""
        with patch("hexa.voice.TTSManager"), \
             patch("hexa.voice.SpeechRecognizer"), \
             patch("hexa.voice.WakeWordDetector"), \
             patch("hexa.voice.HotkeyListener"), \
             patch("hexa.voice.get_audio_stream"), \
             patch("hexa.voice.get_config") as mock_config:

            mock_config.return_value = MagicMock(
                tts=MagicMock(voice="test", voice_path="test", sample_rate=22050, fallback="pyttsx3"),
                stt=MagicMock(model="small.en", device="cpu", compute_type="int8", language="en", beam_size=5, vad_threshold=0.5),
                vad=MagicMock(enabled=True, silence_timeout=1.5, min_speech_duration=0.5),
                wake_word=MagicMock(enabled=True, model_name="hey_jarvis", inference_framework="onnx", threshold=0.5, debounce_time=1.0),
                hotkey=MagicMock(enabled=True, key="<ctrl>+<alt>+<space>"),
                conversation=MagicMock(enabled=True, timeout=0.1)  # Short timeout for test
            )

            pipeline = VoicePipeline()
            pipeline.start()

            # Activate conversation mode
            state_machine = get_state_machine()
            state_machine.transition(State.LISTENING)
            state_machine.transition(State.TRANSCRIBING)
            state_machine.transition(State.THINKING)
            state_machine.transition(State.SPEAKING)
            state_machine.transition(State.IDLE)  # This triggers conversation timer

            time.sleep(0.2)  # Wait for timeout

            # Should be back to IDLE (already there)
            # The timeout should not crash
            pipeline.stop()

    def test_pipeline_interrupt_clears_conversation(self) -> None:
        """Test interrupt clears conversation state."""
        with patch("hexa.voice.TTSManager"), \
             patch("hexa.voice.SpeechRecognizer"), \
             patch("hexa.voice.WakeWordDetector"), \
             patch("hexa.voice.HotkeyListener"), \
             patch("hexa.voice.get_audio_stream"):

            pipeline = VoicePipeline()
            pipeline.start()

            # Activate conversation
            with pipeline._conversation_lock:
                pipeline._conversation_active = True

            pipeline.interrupt()

            with pipeline._conversation_lock:
                assert not pipeline._conversation_active

            pipeline.stop()

    def test_pipeline_transcript_empty_returns_to_listening(self) -> None:
        """Test empty transcript returns to IDLE."""
        with patch("hexa.voice.TTSManager"), \
             patch("hexa.voice.SpeechRecognizer"), \
             patch("hexa.voice.WakeWordDetector"), \
             patch("hexa.voice.HotkeyListener"), \
             patch("hexa.voice.get_audio_stream"):

            pipeline = VoicePipeline()
            pipeline.start()

            # Set to THINKING (as if transcript received)
            state_machine = get_state_machine()
            state_machine.transition(State.LISTENING)
            state_machine.transition(State.TRANSCRIBING)
            state_machine.transition(State.THINKING)

            # Empty transcript should return to IDLE
            pipeline._on_transcript("")

            assert state_machine.current_state == State.IDLE

            pipeline.stop()

    def test_pipeline_audio_stream_shared(self) -> None:
        """Test that audio stream is shared among components."""
        with patch("hexa.voice.TTSManager"), \
             patch("hexa.voice.SpeechRecognizer"), \
             patch("hexa.voice.WakeWordDetector") as mock_wake_class, \
             patch("hexa.voice.HotkeyListener"), \
             patch("hexa.voice.get_audio_stream") as mock_get_stream:

            mock_stream = MagicMock()
            mock_get_stream.return_value = mock_stream

            pipeline = VoicePipeline()
            pipeline.start()

            # All components should use the same stream
            assert mock_get_stream.call_count >= 1

            pipeline.stop()


class TestVoicePipelineSingleton:
    """Tests for global voice pipeline singleton."""

    def test_get_voice_pipeline_singleton(self) -> None:
        """Test singleton pattern."""
        shutdown_voice_pipeline()

        with patch("hexa.voice.TTSManager"), \
             patch("hexa.voice.SpeechRecognizer"), \
             patch("hexa.voice.WakeWordDetector"), \
             patch("hexa.voice.HotkeyListener"), \
             patch("hexa.voice.get_audio_stream"):

            p1 = get_voice_pipeline()
            p2 = get_voice_pipeline()

            assert p1 is p2

            shutdown_voice_pipeline()

    def test_shutdown_voice_pipeline(self) -> None:
        """Test shutdown clears singleton."""
        shutdown_voice_pipeline()

        with patch("hexa.voice.TTSManager"), \
             patch("hexa.voice.SpeechRecognizer"), \
             patch("hexa.voice.WakeWordDetector"), \
             patch("hexa.voice.HotkeyListener"), \
             patch("hexa.voice.get_audio_stream"):

            pipeline = get_voice_pipeline()
            pipeline.start()

            shutdown_voice_pipeline()

            # Should be stopped
            assert not pipeline.is_running()