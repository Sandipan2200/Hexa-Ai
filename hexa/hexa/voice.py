"""Voice pipeline orchestration for Hexa."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from hexa.audio import get_audio_stream, shutdown_audio_stream
from hexa.config import get_config
from hexa.events import emit_event, EventType, get_event_bus
from hexa.router import get_router
from hexa.state import State, StateMachine, get_state_machine
from hexa.stt import SpeechRecognizer, STTConfig, VADConfig
from hexa.tts import TTSManager, TTSConfig
from hexa.wake import WakeWordDetector, WakeWordConfig, HotkeyListener, HotkeyConfig

logger = logging.getLogger(__name__)


@dataclass
class ConversationConfig:
    """Conversation mode configuration."""
    enabled: bool = True
    timeout: float = 10.0  # seconds to wait for follow-up


class PipelineState(Enum):
    """Internal pipeline states (maps to StateMachine)."""
    IDLE = "idle"
    WAITING_WAKE = "waiting_wake"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    CONVERSATION = "conversation"


class VoicePipeline:
    """
    Orchestrates the complete voice pipeline:
    Wake Word → VAD → STT → Router → TTS → Response
    With conversation mode support.
    """

    def __init__(self) -> None:
        self.config = get_config()
        self._state_machine = get_state_machine()
        self._router = get_router()
        self._event_bus = get_event_bus()

        # Components
        self._wake_detector: Optional[WakeWordDetector] = None
        self._hotkey_listener: Optional[HotkeyListener] = None
        self._recognizer: Optional[SpeechRecognizer] = None
        self._tts: Optional[TTSManager] = None

        # Conversation state
        self._conversation_active = False
        self._conversation_timer: Optional[threading.Timer] = None
        self._conversation_lock = threading.Lock()

        # Pipeline state
        self._running = False
        self._pipeline_state = PipelineState.IDLE
        self._interrupt_requested = False

        # Register state machine callbacks
        self._state_machine.add_callback(self._on_state_change)

    def _on_state_change(self, old: State, new: State) -> None:
        """Handle state machine transitions."""
        logger.info("Pipeline state: %s -> %s", old.value, new.value)

        if new == State.LISTENING:
            self._start_listening()
        elif new == State.TRANSCRIBING:
            pass  # STT handles this
        elif new == State.THINKING:
            pass  # Router handles this
        elif new == State.SPEAKING:
            pass  # TTS handles this
        elif new == State.IDLE:
            self._on_idle()

    def _on_idle(self) -> None:
        """Handle return to IDLE state."""
        with self._conversation_lock:
            if self._conversation_active:
                self._start_conversation_timer()
            else:
                self._cancel_conversation_timer()

    def _start_conversation_timer(self) -> None:
        """Start conversation timeout timer."""
        self._cancel_conversation_timer()
        if self.config.conversation.enabled:
            timeout = self.config.conversation.timeout
            self._conversation_timer = threading.Timer(timeout, self._conversation_timeout)
            self._conversation_timer.daemon = True
            self._conversation_timer.start()
            logger.debug("Conversation timer started: %.1fs", timeout)

    def _cancel_conversation_timer(self) -> None:
        """Cancel conversation timeout timer."""
        if self._conversation_timer:
            self._conversation_timer.cancel()
            self._conversation_timer = None

    def _conversation_timeout(self) -> None:
        """Conversation timeout - return to wake word listening."""
        logger.info("Conversation timeout - returning to wake word listening")
        with self._conversation_lock:
            self._conversation_active = False
        self._state_machine.force_idle()

    def _start_listening(self) -> None:
        """Start speech recognition."""
        if self._recognizer:
            self._recognizer.start()

    def _stop_listening(self) -> None:
        """Stop speech recognition."""
        if self._recognizer:
            self._recognizer.stop()

    def _on_transcript(self, text: str) -> None:
        """Handle transcribed text."""
        if not text:
            logger.debug("Empty transcript, returning to idle")
            self._state_machine.transition(State.IDLE)
            return

        logger.info("User said: %s", text)
        self._state_machine.transition(State.THINKING)

        # Route through router
        try:
            response = self._router.route(text)
            self._state_machine.transition(State.SPEAKING)

            # Speak response
            if self._tts:
                handle = self._tts.speak(response)
                # Wait for playback to complete
                handle.wait()

        except Exception as e:
            logger.exception("Error processing transcript")
            emit_event(EventType.ERROR, source="pipeline", error=str(e))
            self._state_machine.transition(State.IDLE)
            return

        # After speaking, check conversation mode
        with self._conversation_lock:
            if self.config.conversation.enabled and self._conversation_active:
                self._state_machine.transition(State.LISTENING)
            else:
                self._state_machine.transition(State.IDLE)

    def _on_wake(self) -> None:
        """Handle wake word detection."""
        if self._state_machine.current_state == State.SPEAKING:
            # Interrupt current speech
            logger.info("Wake word during speech - interrupting")
            self.interrupt()
            time.sleep(0.1)  # Brief pause

        if self._state_machine.current_state in (State.IDLE, State.LISTENING):
            logger.info("Wake word detected - starting conversation")
            with self._conversation_lock:
                self._conversation_active = True
            self._state_machine.transition(State.LISTENING)
        elif self._state_machine.current_state == State.CONVERSATION:
            # Already in conversation, just continue listening
            self._state_machine.transition(State.LISTENING)

    def _on_hotkey_press(self) -> None:
        """Handle hotkey press (push-to-talk start)."""
        logger.debug("Hotkey pressed")
        if self._state_machine.current_state == State.SPEAKING:
            self.interrupt()
            time.sleep(0.1)

        if self._state_machine.current_state in (State.IDLE, State.LISTENING):
            with self._conversation_lock:
                self._conversation_active = True
            self._state_machine.transition(State.LISTENING)

    def _on_hotkey_release(self) -> None:
        """Handle hotkey release (push-to-talk end)."""
        logger.debug("Hotkey released")
        if self._state_machine.current_state == State.LISTENING:
            # Will transition to TRANSCRIBING via VAD automatically
            pass

    def _on_interrupt(self, reason: str = "") -> None:
        """Handle interrupt event."""
        logger.info("Interrupt: %s", reason)
        self._interrupt_requested = True
        self._tts.interrupt()
        self._stop_listening()
        with self._conversation_lock:
            self._conversation_active = False
        self._cancel_conversation_timer()
        self._state_machine.force_idle()

    def start(self) -> None:
        """Start the voice pipeline."""
        if self._running:
            return

        logger.info("Starting voice pipeline")

        # Initialize components
        self._tts = TTSManager(TTSConfig(
            voice=self.config.tts.voice,
            voice_path=self.config.tts.voice_path,
            sample_rate=self.config.tts.sample_rate,
            fallback=self.config.tts.fallback,
        ))

        self._recognizer = SpeechRecognizer(
            stt_config=STTConfig(
                model=self.config.stt.model,
                device=self.config.stt.device,
                compute_type=self.config.stt.compute_type,
                language=self.config.stt.language,
                beam_size=self.config.stt.beam_size,
                vad_threshold=self.config.stt.vad_threshold,
            ),
            vad_config=VADConfig(
                enabled=self.config.vad.enabled,
                silence_timeout=self.config.vad.silence_timeout,
                min_speech_duration=self.config.vad.min_speech_duration,
            ),
            on_transcript=self._on_transcript
        )

        self._wake_detector = WakeWordDetector(
            config=WakeWordConfig(
                enabled=self.config.wake_word.enabled,
                model_name=self.config.wake_word.model_name,
                inference_framework=self.config.wake_word.inference_framework,
                threshold=self.config.wake_word.threshold,
                debounce_time=self.config.wake_word.debounce_time,
            ),
            on_wake=self._on_wake
        )

        self._hotkey_listener = HotkeyListener(
            config=HotkeyConfig(
                enabled=self.config.hotkey.enabled,
                key=self.config.hotkey.key,
            ),
            on_press=self._on_hotkey_press,
            on_release=self._on_hotkey_release
        )

        # Subscribe to interrupt events
        self._event_bus.subscribe(EventType.INTERRUPT, lambda e: self._on_interrupt(e.data.get("reason", "")))

        # Start components
        self._wake_detector.start()
        self._hotkey_listener.start()

        # Start audio stream
        get_audio_stream()

        self._running = True
        logger.info("Voice pipeline started")

    def stop(self) -> None:
        """Stop the voice pipeline."""
        if not self._running:
            return

        logger.info("Stopping voice pipeline")
        self._running = False

        self._cancel_conversation_timer()
        self._interrupt_requested = True

        if self._wake_detector:
            self._wake_detector.stop()
        if self._hotkey_listener:
            self._hotkey_listener.stop()
        if self._recognizer:
            self._recognizer.stop()
        if self._tts:
            self._tts.interrupt()

        shutdown_audio_stream()
        logger.info("Voice pipeline stopped")

    def interrupt(self) -> None:
        """Public interrupt method."""
        self._on_interrupt("user_interrupt")

    def is_running(self) -> bool:
        return self._running

    def is_speaking(self) -> bool:
        return self._state_machine.current_state == State.SPEAKING


# Global pipeline instance
_pipeline: Optional[VoicePipeline] = None


def get_voice_pipeline() -> VoicePipeline:
    """Get or create the global voice pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = VoicePipeline()
    return _pipeline


def shutdown_voice_pipeline() -> None:
    """Shutdown the global voice pipeline."""
    global _pipeline
    if _pipeline:
        _pipeline.stop()
        _pipeline = None