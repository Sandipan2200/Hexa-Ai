"""Shared audio input stream and playback for Hexa."""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from hexa.events import emit_event, EventType, get_event_bus

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio configuration."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1280  # 80ms at 16kHz
    input_device: Optional[int] = None
    dtype: str = "int16"


class AudioStream:
    """
    Single shared microphone stream at 16kHz mono.
    Distributes chunks to multiple consumers via thread-safe queues.
    """

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._consumers: list[queue.Queue[np.ndarray]] = []
        self._consumer_lock = threading.Lock()
        self._overflow_count = 0
        self._last_chunk_time = 0.0

    def register_consumer(self, maxsize: int = 100) -> queue.Queue[np.ndarray]:
        """Register a new consumer queue for audio chunks."""
        q: queue.Queue[np.ndarray] = queue.Queue(maxsize=maxsize)
        with self._consumer_lock:
            self._consumers.append(q)
        return q

    def unregister_consumer(self, q: queue.Queue[np.ndarray]) -> None:
        """Unregister a consumer queue."""
        with self._consumer_lock:
            if q in self._consumers:
                self._consumers.remove(q)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Sounddevice callback - runs in audio thread."""
        if status:
            logger.warning("Audio callback status: %s", status)

        # Convert to int16 mono
        if indata.dtype != np.int16:
            chunk = (indata * 32767).astype(np.int16)
        else:
            chunk = indata.copy()

        # Ensure mono
        if chunk.ndim > 1 and chunk.shape[1] > 1:
            chunk = chunk[:, 0:1]

        chunk = chunk.flatten()
        now = time.time()

        # Calculate RMS for audio level event
        if len(chunk) > 0:
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            # Emit ~20x/sec (every 80ms chunk)
            if now - self._last_chunk_time >= 0.05:  # throttle to ~20Hz
                emit_event(EventType.AUDIO_LEVEL, rms_mic=rms)
                self._last_chunk_time = now

        # Distribute to all consumers (non-blocking)
        with self._consumer_lock:
            for q in self._consumers:
                try:
                    q.put_nowait(chunk)
                except queue.Full:
                    self._overflow_count += 1
                    if self._overflow_count % 100 == 1:
                        logger.warning("Audio queue overflow (count=%d)", self._overflow_count)

    def start(self) -> None:
        """Start the audio input stream."""
        if self._running:
            return

        try:
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                blocksize=self.config.chunk_size,
                dtype=self.config.dtype,
                device=self.config.input_device,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True
            logger.info("Audio stream started: %dHz, %dch, chunk=%d",
                        self.config.sample_rate, self.config.channels, self.config.chunk_size)
        except Exception as e:
            logger.exception("Failed to start audio stream")
            emit_event(EventType.ERROR, source="audio", error=f"Failed to start mic: {e}")
            raise

    def stop(self) -> None:
        """Stop the audio input stream."""
        if not self._running:
            return

        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning("Error stopping audio stream: %s", e)
            finally:
                self._stream = None
        logger.info("Audio stream stopped")

    def is_running(self) -> bool:
        return self._running


# Global audio stream instance
_audio_stream: Optional[AudioStream] = None


def get_audio_stream(config: Optional[AudioConfig] = None) -> AudioStream:
    """Get or create the global audio stream."""
    global _audio_stream
    if _audio_stream is None:
        if config is None:
            from hexa.config import get_config
            cfg = get_config()
            config = AudioConfig(
                sample_rate=cfg.audio.sample_rate,
                channels=cfg.audio.channels,
                chunk_size=cfg.audio.chunk_size,
                input_device=cfg.audio.input_device,
            )
        _audio_stream = AudioStream(config)
    return _audio_stream


def shutdown_audio_stream() -> None:
    """Shutdown the global audio stream."""
    global _audio_stream
    if _audio_stream:
        _audio_stream.stop()
        _audio_stream = None


# --- Playback ---

def play_wav_file(path: str, blocking: bool = True) -> None:
    """Play a WAV file through the default output device."""
    try:
        data, samplerate = sf.read(path, dtype="float32")
        if data.ndim > 1 and data.shape[1] > 1:
            data = data[:, 0]  # mono
        sd.play(data, samplerate, blocking=blocking)
    except Exception as e:
        logger.exception("Failed to play WAV file: %s", path)
        emit_event(EventType.ERROR, source="tts", error=f"Playback failed: {e}")
        raise


def play_audio_bytes(audio_bytes: bytes, sample_rate: int, blocking: bool = True) -> None:
    """Play raw audio bytes (int16) through the default output device."""
    try:
        data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        sd.play(data, sample_rate, blocking=blocking)
    except Exception as e:
        logger.exception("Failed to play audio bytes")
        emit_event(EventType.ERROR, source="tts", error=f"Playback failed: {e}")
        raise


class PlaybackHandle:
    """Handle for interruptible playback."""

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def interrupt(self) -> None:
        """Stop playback immediately."""
        self._stop_event.set()
        sd.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def wait(self) -> None:
        """Wait for playback to complete."""
        if self._thread:
            self._thread.join()

    @property
    def is_playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


def play_wav_async(path: str) -> PlaybackHandle:
    """Play a WAV file asynchronously, returns handle for interruption."""
    handle = PlaybackHandle()

    def _play() -> None:
        try:
            data, samplerate = sf.read(path, dtype="float32")
            if data.ndim > 1 and data.shape[1] > 1:
                data = data[:, 0]
            # Play in chunks to allow interruption
            chunk_size = samplerate // 10  # 100ms chunks
            for i in range(0, len(data), chunk_size):
                if handle._stop_event.is_set():
                    break
                chunk = data[i:i + chunk_size]
                sd.play(chunk, samplerate, blocking=True)
        except Exception as e:
            logger.exception("Async playback failed")
            emit_event(EventType.ERROR, source="tts", error=f"Async playback failed: {e}")

    handle._thread = threading.Thread(target=_play, daemon=True)
    handle._thread.start()
    return handle


def play_audio_bytes_async(audio_bytes: bytes, sample_rate: int) -> PlaybackHandle:
    """Play raw audio bytes asynchronously, returns handle for interruption."""
    handle = PlaybackHandle()

    def _play() -> None:
        try:
            data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            chunk_size = sample_rate // 10  # 100ms chunks
            for i in range(0, len(data), chunk_size):
                if handle._stop_event.is_set():
                    break
                chunk = data[i:i + chunk_size]
                sd.play(chunk, sample_rate, blocking=True)
        except Exception as e:
            logger.exception("Async playback failed")
            emit_event(EventType.ERROR, source="tts", error=f"Async playback failed: {e}")

    handle._thread = threading.Thread(target=_play, daemon=True)
    handle._thread.start()
    return handle