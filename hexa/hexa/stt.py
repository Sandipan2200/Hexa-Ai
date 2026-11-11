"""Speech-to-text with voice activity detection for Hexa."""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from hexa.audio import get_audio_stream
from hexa.config import get_config
from hexa.events import emit_event, EventType

logger = logging.getLogger(__name__)


@dataclass
class STTConfig:
    """STT configuration."""
    model: str = "small.en"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 5
    vad_threshold: float = 0.5


@dataclass
class VADConfig:
    """VAD configuration."""
    enabled: bool = True
    silence_timeout: float = 1.5
    min_speech_duration: float = 0.5
    speech_pad_ms: int = 200


class SileroVAD:
    """Silero VAD wrapper for voice activity detection."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._model = None
        self._residual = np.zeros(0, dtype=np.float32)  # samples not yet fed to the model
        self._load_model()

    def _load_model(self) -> None:
        """Load Silero VAD model."""
        try:
            # Use silero-vad package
            from silero_vad import load_silero_vad
            self._model = load_silero_vad()
            logger.info("Silero VAD model loaded")
        except Exception as e:
            logger.warning("Failed to load Silero VAD: %s", e)
            # Fallback to openWakeWord's built-in VAD if available
            try:
                from openwakeword.utils import AudioFeatures
                self._model = AudioFeatures()  # Has VAD built-in
                logger.info("Using openWakeWord AudioFeatures for VAD")
            except Exception:
                logger.warning("No VAD available, using energy-based fallback")
                self._model = None

    def get_speech_prob(self, chunk: np.ndarray, sample_rate: int = 16000) -> float:
        """Get speech probability for a chunk.

        Silero is a STATEFUL recurrent model that must see contiguous
        512-sample windows at 16kHz. Feeding it decimated or padded windows
        (e.g. the first 512 samples of every 1280-sample chunk) corrupts its
        internal state - Gate 4: probabilities up to 0.99 on room tone. So
        buffer samples and feed the model contiguous 512-sample windows,
        returning the max probability across this chunk's windows.
        """
        if self._model is None:
            # Energy-based fallback
            if len(chunk) == 0:
                return 0.0
            energy = float(np.mean(chunk.astype(np.float32) ** 2))
            return min(1.0, energy / 1000.0)

        try:
            # silero-vad package
            import torch
            x = np.concatenate([self._residual, chunk.astype(np.float32) / 32768.0])
            n_full = (len(x) // 512) * 512
            self._residual = x[n_full:]
            probs = []
            with torch.no_grad():
                for i in range(0, n_full, 512):
                    tensor = torch.from_numpy(x[i:i + 512])
                    probs.append(float(self._model(tensor, sample_rate).item()))
            return max(probs) if probs else 0.0
        except Exception as e:
            logger.debug("VAD error: %s", e)
            return 0.0

    def reset(self) -> None:
        """Reset the recurrent state and residual buffer (call at utterance
        boundaries so context does not bleed across unrelated segments)."""
        self._residual = np.zeros(0, dtype=np.float32)
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()


class WhisperTranscriber:
    """faster-whisper transcription with VAD integration."""

    def __init__(self, config: STTConfig) -> None:
        self.config = config
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load faster-whisper model from the local HF cache (fully offline)."""
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.config.model,
                device=self.config.device,
                compute_type=self.config.compute_type,
                local_files_only=True,  # never contact huggingface.co at runtime
            )
            logger.info("Whisper model loaded locally: %s", self.config.model)
        except Exception as e:
            hint = (
                f"The cached Whisper model '{self.config.model}' was not found locally. "
                f"Download it once while online, e.g.: "
                f".venv\\Scripts\\python.exe -c \"from faster_whisper import WhisperModel; "
                f"WhisperModel('{self.config.model}')\" "
                f"(then Hexa runs fully offline)."
            )
            logger.error("Failed to load Whisper model: %s. %s", e, hint)
            emit_event(EventType.ERROR, source="stt", error=f"Model load failed: {e}")
            raise RuntimeError(f"Whisper model '{self.config.model}' unavailable: {e}. {hint}") from e

    def transcribe(self, audio_chunks: list[np.ndarray]) -> str:
        """
        Transcribe a list of audio chunks.
        Returns transcribed text or empty string.
        """
        if not audio_chunks:
            return ""

        try:
            # Concatenate chunks
            audio = np.concatenate(audio_chunks)
            # Convert to float32 [-1, 1]
            audio_float = audio.astype(np.float32) / 32768.0

            # Transcribe
            segments, info = self._model.transcribe(
                audio_float,
                language=self.config.language,
                beam_size=self.config.beam_size,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            text = " ".join(seg.text for seg in segments).strip()
            logger.debug("Transcribed: %s (lang=%s, prob=%.2f)", text, info.language, info.language_probability)
            return text

        except Exception as e:
            logger.exception("Transcription failed")
            emit_event(EventType.ERROR, source="stt", error=f"Transcription failed: {e}")
            return ""


class SpeechRecognizer:
    """
    Continuous speech recognition with VAD.
    Listens for speech, collects chunks, transcribes on silence.
    """

    def __init__(
        self,
        stt_config: STTConfig,
        vad_config: VADConfig,
        on_transcript: Callable[[str], None]
    ) -> None:
        self.stt_config = stt_config
        self.vad_config = vad_config
        self._on_transcript = on_transcript
        self._transcriber = WhisperTranscriber(stt_config)
        self._vad = SileroVAD(stt_config.vad_threshold)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._audio_queue: Optional[queue.Queue] = None
        self._audio_stream = None

        # VAD state
        self._speech_chunks: list[np.ndarray] = []
        self._silence_chunks = 0
        self._in_speech = False
        self._chunk_duration_ms = 80  # 1280 samples at 16kHz = 80ms
        self._silence_limit = int(vad_config.silence_timeout * 1000 / self._chunk_duration_ms)
        self._min_speech_chunks = int(vad_config.min_speech_duration * 1000 / self._chunk_duration_ms)

    def start(self) -> None:
        """Start speech recognition."""
        if self._running:
            return

        if not self.vad_config.enabled:
            logger.info("VAD disabled, using simple recording")
            # Could implement fixed-duration recording here
            pass

        self._audio_stream = get_audio_stream()
        self._audio_queue = self._audio_stream.register_consumer(maxsize=100)

        self._running = True
        self._thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self._thread.start()
        logger.info("Speech recognition started")

    def stop(self) -> None:
        """Stop speech recognition."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._audio_queue and self._audio_stream:
            self._audio_stream.unregister_consumer(self._audio_queue)
        logger.info("Speech recognition stopped")

    def _process_speech(self) -> None:
        """Process collected speech chunks."""
        if len(self._speech_chunks) >= self._min_speech_chunks:
            logger.debug("Processing %d speech chunks", len(self._speech_chunks))
            text = self._transcriber.transcribe(self._speech_chunks)
            if text:
                emit_event(EventType.TRANSCRIPT, text=text)
                self._on_transcript(text)
            else:
                logger.debug("Empty transcription")
        else:
            logger.debug("Speech too short (%d chunks), discarding", len(self._speech_chunks))
        self._speech_chunks.clear()
        self._vad.reset()  # utterance boundary: don't bleed recurrent state

    def _recognition_loop(self) -> None:
        """Main recognition loop with VAD."""
        # Use STT config's vad_threshold for VAD decision
        vad_threshold = self.stt_config.vad_threshold

        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.1)
                if chunk is None:
                    continue

                # Get speech probability
                speech_prob = self._vad.get_speech_prob(chunk)

                if speech_prob >= vad_threshold:
                    # Speech detected
                    if not self._in_speech:
                        logger.debug("Speech started")
                        self._in_speech = True
                        self._silence_chunks = 0
                    self._speech_chunks.append(chunk.copy())
                else:
                    # Silence detected
                    if self._in_speech:
                        self._silence_chunks += 1
                        self._speech_chunks.append(chunk.copy())  # Pad with silence
                        if self._silence_chunks >= self._silence_limit:
                            # End of speech
                            logger.debug("Speech ended after %d silence chunks", self._silence_chunks)
                            self._in_speech = False
                            self._process_speech()
                            self._silence_chunks = 0
                    # else: still in silence, ignore

            except queue.Empty:
                continue
            except Exception as e:
                logger.exception("Error in recognition loop")
                emit_event(EventType.ERROR, source="stt", error=str(e))
                time.sleep(0.1)