"""Shared audio input stream and playback for Hexa."""
from __future__ import annotations

import io
import logging
import math
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
    input_device: int | str | None = None
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
        self._capture_device: Optional[int] = None  # resolved at start()
        self._capture_channels: int = config.channels
        self._capture_native_rate: int = config.sample_rate

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

    def _resample_window(self, data: np.ndarray) -> np.ndarray:
        """Downmix a multi-channel native-rate capture window to mono at the
        configured output sample rate (int16). Used by the polling capture path
        (Gate 3: MME pin @44.1kHz -> 16kHz via scipy polyphase resampling)."""
        mono = np.mean(data.astype(np.float32), axis=1)
        g = math.gcd(self.config.sample_rate, self._capture_native_rate)
        up, down = self.config.sample_rate // g, self._capture_native_rate // g
        from scipy.signal import resample_poly  # lazy: only the polling capture path needs it
        # padtype="line": the default zero-pad leaves a filter transient at the
        # window edge - an audible micro-click every 800ms capture window.
        resampled = resample_poly(mono, up, down, padtype="line")
        return np.clip(resampled, -32768, 32767).astype(np.int16)

    def _poll_loop(self) -> None:
        """Polling capture loop for devices whose driver drops PortAudio callbacks.

        Gate 3 root cause: this laptop's Intel SST array only delivers valid audio
        through its MME pin at the NATIVE 44.1kHz - its WDM-KS 16kHz pin serves
        misformatted data (float32 -> NaN, int16 -> garbage). Open ONE pull-mode
        InputStream at the device's native rate and read windows with blocking
        read(): a fresh sd.rec per window re-opens the device every 800ms, and the
        measured open/close overhead loses ~40% of audio. Downmix to mono,
        resample to the configured 16kHz with scipy, split into chunks.
        """
        chunk = self.config.chunk_size
        window_chunks = 10  # 10 x 80ms = 800ms per read (amortize processing)
        native_rate = self._capture_native_rate
        out_rate = self.config.sample_rate
        window_samples = round(window_chunks * chunk / out_rate * native_rate)
        out_window_samples = chunk * window_chunks
        # Capture parameters resolved once at start() time (device index is unstable)
        channels = self._capture_channels
        device = self._capture_device
        logger.info("Audio polling loop started (window=%d @%dHz -> %d @%dHz, %dch, chunk=%d)",
                    window_samples, native_rate, out_window_samples, out_rate, channels, chunk)
        rec_stream = None
        try:
            while self._running:
                try:
                    if rec_stream is None:
                        rec_stream = sd.InputStream(
                            samplerate=native_rate,
                            channels=channels,
                            dtype=self.config.dtype,
                            device=device,
                        )
                        rec_stream.start()
                        logger.info("Polling capture stream opened: native %dHz, %dch, device=%s",
                                    native_rate, channels, device)
                    data, overflowed = rec_stream.read(window_samples)
                    if overflowed:
                        logger.warning("Polling capture overflowed (samples dropped)")
                except Exception as e:
                    logger.exception("Polling capture failed")
                    emit_event(EventType.ERROR, source="audio", error=f"Capture failed: {e}")
                    if rec_stream is not None:
                        try:
                            rec_stream.close()
                        except Exception:
                            logger.exception("Failed to close polling capture stream")
                        rec_stream = None
                    time.sleep(0.1)
                    continue
                try:
                    # Downmix to mono and resample to the configured output rate
                    pcm = self._resample_window(data)
                except Exception:
                    logger.exception("Polling resample failed")
                    emit_event(EventType.ERROR, source="audio", error="Resample failed")
                    continue
                # Pad/trim to exactly the expected window length, then distribute
                if len(pcm) < out_window_samples:
                    pcm = np.pad(pcm, (0, out_window_samples - len(pcm)), mode="constant")
                pcm = pcm[:out_window_samples]
                for i in range(0, len(pcm), chunk):
                    piece = pcm[i:i + chunk]
                    if len(piece) < chunk:
                        piece = np.pad(piece, (0, chunk - len(piece)), mode="constant")
                    self._process_chunk(piece)
        finally:
            if rec_stream is not None:
                try:
                    rec_stream.stop()
                    rec_stream.close()
                except Exception:
                    logger.exception("Failed to close polling capture stream")

    def _resolve_input_device(self) -> tuple[int | None, int, int]:
        """Resolve the configured input device to a live PortAudio index.

        Device indexes are NOT stable across sessions (Bluetooth connect/disconnect
        shifts them - Gate 3 root cause). Resolution order:
          1. int index (only if still valid)
          2. name substring match (config value)
          3. OS default input
        WDM-KS pins are deprioritized: this laptop's Intel SST array serves
        misformatted data through WDM-KS (float32 -> NaN) while the same array's
        MME pin works. Deliberately NO auto-match by native rate - that would
        re-select the poisoned WDM-KS 16kHz pin.
        Returns (device_index_or_None, channels, native_sample_rate).
        """
        try:
            devices = sd.query_devices()
        except Exception:
            logger.exception("Cannot enumerate audio devices")
            return None, self.config.channels, self.config.sample_rate

        def device_tuple(i: int, d) -> tuple[int, int, int]:
            ch = min(self.config.channels, int(d["max_input_channels"]))
            return i, ch, int(d["default_samplerate"])

        spec = self.config.input_device
        if isinstance(spec, int) and 0 <= spec < len(devices) \
                and devices[spec]["max_input_channels"] > 0:
            return device_tuple(spec, devices[spec])
        if isinstance(spec, int):
            logger.warning("Configured input_device index %s is stale; resolving by name/rate", spec)
            spec = None

        def matches(d) -> bool:
            name_ok = spec is None or (isinstance(spec, str) and spec.lower() in d["name"].lower())
            return name_ok and d["max_input_channels"] > 0

        # Prefer non-WDM-KS matches (hostapi 3); fall back to any match.
        for hosts in ((1, 2, 0), None):
            for i, d in enumerate(devices):
                if not matches(d):
                    continue
                if hosts is not None and d["hostapi"] not in hosts:
                    continue
                return device_tuple(i, d)

        # Nothing matched by name - fall back to the OS default input
        if spec is not None:
            logger.warning("No input device matching %r; using OS default", spec)
        return None, self.config.channels, self.config.sample_rate

    def start(self) -> None:
        """Start the audio input stream."""
        if self._running:
            return

        try:
            device, channels, native_rate = self._resolve_input_device()
            self._capture_device = device
            self._capture_channels = channels
            self._capture_native_rate = native_rate
            if self.config.use_polling:
                self._running = True
                self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
                self._poll_thread.start()
                logger.info("Audio polling stream started: %dHz->%dHz, %dch, chunk=%d, device=%s",
                            native_rate, self.config.sample_rate, channels,
                            self.config.chunk_size, device)
                return

            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=channels,
                blocksize=self.config.chunk_size,
                dtype=self.config.dtype,
                device=device,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True
            logger.info("Audio stream started: %dHz, %dch, chunk=%d, device=%s",
                        self.config.sample_rate, channels, self.config.chunk_size, device)
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