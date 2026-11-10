"""Text-to-speech with Piper and pyttsx3 fallback for Hexa."""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hexa.audio import play_audio_bytes_async, play_wav_async, PlaybackHandle
from hexa.config import get_config
from hexa.events import emit_event, EventType

logger = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    """TTS configuration."""
    voice: str = "en_GB-alan-medium"
    voice_path: str = "models/en_GB-alan-medium.onnx"
    sample_rate: int = 22050
    fallback: str = "pyttsx3"
    speed: float = 1.0


class PiperTTS:
    """Piper TTS engine."""

    def __init__(self, config: TTSConfig) -> None:
        self.config = config
        self._voice = None
        self._load_voice()

    def _load_voice(self) -> None:
        """Load Piper voice model."""
        try:
            from piper import PiperVoice
            voice_path = Path(self.config.voice_path)
            if not voice_path.is_absolute():
                # Relative to project root
                voice_path = Path(__file__).parent.parent.parent / self.config.voice_path

            if not voice_path.exists():
                raise FileNotFoundError(f"Piper voice not found: {voice_path}")

            self._voice = PiperVoice.load(str(voice_path))
            logger.info("Piper voice loaded: %s", voice_path)
        except Exception as e:
            logger.exception("Failed to load Piper voice")
            emit_event(EventType.ERROR, source="tts", error=f"Piper load failed: {e}")
            raise

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio bytes (int16)."""
        if not self._voice:
            raise RuntimeError("Piper voice not loaded")

        try:
            # Synthesize to bytes
            import io
            import wave

            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.config.sample_rate)

                # Synthesize
                self._voice.synthesize_wav(text, wf)

            buffer.seek(0)
            return buffer.read()
        except Exception as e:
            logger.exception("Piper synthesis failed")
            emit_event(EventType.ERROR, source="tts", error=f"Piper synthesis failed: {e}")
            raise


class Pyttsx3TTS:
    """pyttsx3 fallback TTS engine."""

    def __init__(self, config: TTSConfig) -> None:
        self.config = config
        self._engine = None
        self._init_engine()

    def _init_engine(self) -> None:
        """Initialize pyttsx3 engine."""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', int(200 * self.config.speed))
            # Try to set voice
            voices = self._engine.getProperty('voices')
            for v in voices:
                if 'english' in v.name.lower() or 'en' in v.id.lower():
                    self._engine.setProperty('voice', v.id)
                    break
            logger.info("pyttsx3 engine initialized")
        except Exception as e:
            logger.exception("Failed to initialize pyttsx3")
            emit_event(EventType.ERROR, source="tts", error=f"pyttsx3 init failed: {e}")
            raise

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio bytes via temporary WAV file."""
        if not self._engine:
            raise RuntimeError("pyttsx3 engine not initialized")

        try:
            import tempfile
            import wave

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            self._engine.save_to_file(text, tmp_path)
            self._engine.runAndWait()

            # Read the generated WAV
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()

            os.unlink(tmp_path)
            return audio_bytes
        except Exception as e:
            logger.exception("pyttsx3 synthesis failed")
            emit_event(EventType.ERROR, source="tts", error=f"pyttsx3 synthesis failed: {e}")
            raise


class TTSManager:
    """Manages TTS engines with fallback and interruptible playback."""

    def __init__(self, config: TTSConfig) -> None:
        self.config = config
        self._piper: Optional[PiperTTS] = None
        self._fallback: Optional[Pyttsx3TTS] = None
        self._current_handle: Optional[PlaybackHandle] = None
        self._lock = threading.Lock()
        self._initialize_engines()

    def _initialize_engines(self) -> None:
        """Initialize primary and fallback engines."""
        # Try Piper first
        try:
            self._piper = PiperTTS(self.config)
            logger.info("Primary TTS: Piper")
        except Exception as e:
            logger.warning("Piper unavailable: %s", e)

        # Initialize fallback
        if self.config.fallback == "pyttsx3":
            try:
                self._fallback = Pyttsx3TTS(self.config)
                logger.info("Fallback TTS: pyttsx3")
            except Exception as e:
                logger.warning("pyttsx3 fallback unavailable: %s", e)

    def speak(self, text: str) -> PlaybackHandle:
        """
        Speak text and return handle for interruption.
        Uses primary engine, falls back on failure.
        """
        with self._lock:
            self.interrupt()  # Stop any current playback

            # Try primary
            if self._piper:
                try:
                    audio_bytes = self._piper.synthesize(text)
                    handle = play_audio_bytes_async(audio_bytes, self.config.sample_rate)
                    self._current_handle = handle
                    logger.debug("Spoke via Piper: %s", text[:50])
                    return handle
                except Exception as e:
                    logger.warning("Piper failed, trying fallback: %s", e)

            # Try fallback
            if self._fallback:
                try:
                    audio_bytes = self._fallback.synthesize(text)
                    handle = play_audio_bytes_async(audio_bytes, self.config.sample_rate)
                    self._current_handle = handle
                    logger.debug("Spoke via pyttsx3: %s", text[:50])
                    return handle
                except Exception as e:
                    logger.error("Fallback TTS failed: %s", e)

            raise RuntimeError("No TTS engine available")

    def interrupt(self) -> None:
        """Interrupt current playback immediately."""
        with self._lock:
            if self._current_handle:
                logger.debug("Interrupting TTS playback")
                self._current_handle.interrupt()
                self._current_handle = None
                emit_event(EventType.INTERRUPT, reason="tts_interrupt")

    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        with self._lock:
            return self._current_handle is not None and self._current_handle.is_playing

    def wait(self) -> None:
        """Wait for current playback to complete."""
        with self._lock:
            if self._current_handle:
                self._current_handle.wait()