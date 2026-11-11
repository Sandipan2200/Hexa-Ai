"""Shared audio input stream and playback for Hexa."""
from __future__ import annotations

import io
import logging
import queue
import threading
import time
import wave
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
    use_polling: bool = False  # Use polling thread instead of callback (for WDM-KS devices)


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
        self._poll_thread: Optional[threading.Thread] = None  # polling mode thread

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

        self._process_chunk(indata)

    def _process_chunk(self, chunk: np.ndarray) -> None:
        """Normalize a raw capture block to int16 mono and distribute to consumers."""
        if chunk.dtype != np.int16:
            chunk = (chunk * 32767).astype(np.int16)
        else:
            chunk = chunk.copy()

        # Ensure mono (take channel 0; Intel SST arrays expose ch0 as beamformed)
        if chunk.ndim > 1 and chunk.shape[1] > 1:
            chunk = chunk[:, 0:1]

        chunk = chunk.flatten()
        now = time.time()

        # Calculate RMS for audio level event
        if len(chunk) > 0:
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
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

    def _poll_loop(self) -> None:
        """Polling capture loop for devices whose driver drops PortAudio callbacks
        (e.g. Intel SST arrays exposed only via WDM-KS). Uses sd.rec in short
        blocking windows - proven to deliver real audio on this hardware.

        Captures in 5-chunk windows and splits, since a single sd.rec call per
        80ms chunk is too slow (stream open/close overhead > 80ms)."""
        chunk = self.config.chunk_size
        window_chunks = 10  # capture 10 x 80ms = 800ms per sd.rec call (amortize overhead)
        window_samples = chunk * window_chunks
        chunk_seconds = chunk / self.config.sample_rate
        logger.info("Audio polling loop started (window=%d samples = %.0fms, chunk=%d)",
                    window_samples, window_samples / self.config.sample_rate * 1000, chunk)
        while self._running:
            start = time.perf_counter()
            try:
                # Blocking capture of one window worth of audio
                data = sd.rec(
                    window_samples,
                    samplerate=self.config.sample_rate,
                    channels=self.config.channels,
                    dtype=self.config.dtype,
                    device=self.config.input_device,
                )
                sd.wait()
            except Exception as e:
                logger.exception("Polling capture failed")
                emit_event(EventType.ERROR, source="audio", error=f"Capture failed: {e}")
                time.sleep(0.1)
                continue
            # Split window into chunk-sized pieces and distribute
            for i in range(0, len(data), chunk):
                piece = data[i:i + chunk]
                if len(piece) < chunk:
                    piece = np.pad(piece, (0, chunk - len(piece)), mode="constant")
                self._process_chunk(piece)
            # Pace: keep roughly real-time
            elapsed = time.perf_counter() - start
            delay = window_seconds = window_samples / self.config.sample_rate - elapsed
            if delay > 0:
                time.sleep(delay)

    def start(self) -> None:
        """Start the audio input stream."""
        if self._running:
            return

        try:
            if self.config.use_polling:
                self._running = True
                self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
                self._poll_thread.start()
                logger.info("Audio polling stream started: %dHz, %dch, chunk=%d, device=%s",
                            self.config.sample_rate, self.config.channels,
                            self.config.chunk_size, self.config.input_device)
                return

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
            self._running = False
            logger.exception("Failed to start audio stream")
            emit_event(EventType.ERROR, source="audio", error=f"Failed to start mic: {e}")
            raise

    def stop(self) -> None:
        """Stop the audio input stream."""
        if not self._running:
            return

        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None
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
                use_polling=cfg.audio.use_polling,
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

def _decode_int16_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode raw bytes as int16 PCM.

    Callers pass either raw PCM frames or complete WAV files (Piper/pyttsx3
    produce WAV containers). If the bytes start with a RIFF/WAVE header, the
    header is parsed and stripped - decoding it as audio produced an audible
    click at the start of every utterance (Gate 2 root cause).
    Returns (float32 mono data, sample rate).
    """
    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            rate = wf.getframerate()
            channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        if channels > 1:
            data = data[::channels]
        return data, rate
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0, -1


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
    """Play raw audio bytes (int16 PCM or a WAV file) through the default output."""
    try:
        data, wav_rate = _decode_int16_bytes(audio_bytes)
        if wav_rate > 0:
            sample_rate = wav_rate  # trust the container's rate over the caller's
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
    """Play raw audio bytes (int16 PCM or a WAV file) asynchronously."""
    handle = PlaybackHandle()

    def _play() -> None:
        try:
            data, wav_rate = _decode_int16_bytes(audio_bytes)
            if wav_rate > 0:
                sample_rate = wav_rate  # trust the container's rate over the caller's
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