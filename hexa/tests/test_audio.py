"""Tests for audio module."""
from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hexa.audio import AudioConfig, AudioStream, PlaybackHandle, get_audio_stream, shutdown_audio_stream


@pytest.fixture
def audio_config() -> AudioConfig:
    return AudioConfig(sample_rate=16000, chunk_size=1280)


def test_audio_stream_start_stop(audio_config: AudioConfig) -> None:
    """Test audio stream lifecycle."""
    with patch("hexa.audio.sd.InputStream") as mock_stream_class:
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        stream = AudioStream(audio_config)
        stream.start()

        assert stream.is_running()
        mock_stream.start.assert_called_once()

        stream.stop()
        assert not stream.is_running()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()


def test_audio_stream_polling_mode(audio_config: AudioConfig) -> None:
    """Test polling mode lifecycle: no InputStream, a thread that captures via sd.rec.

    Polling mode exists because Intel SST mic arrays exposed only via WDM-KS
    deliver no PortAudio callbacks (Gate 1 root cause).
    """
    poll_config = AudioConfig(
        sample_rate=16000, chunk_size=1280, channels=4, input_device=21, use_polling=True
    )
    fake_window = np.zeros((12800, 4), dtype=np.int16)
    with patch("hexa.audio.sd.rec", return_value=fake_window) as mock_rec, \
         patch("hexa.audio.sd.wait") as mock_wait:
        stream = AudioStream(poll_config)
        q = stream.register_consumer()
        stream.start()

        assert stream.is_running()

        # Wait until the polling thread has distributed at least one chunk
        chunk = q.get(timeout=2.0)
        assert len(chunk) == 1280
        mock_rec.assert_called()  # capture actually happens via sd.rec
        mock_wait.assert_called()

        stream.stop()
        assert not stream.is_running()
        # InputStream must never be created in polling mode
        assert stream._stream is None


def test_process_chunk_multichannel_to_mono(audio_config: AudioConfig) -> None:
    """Multi-channel input must be reduced to channel 0 mono chunks."""
    stream = AudioStream(audio_config)
    q = stream.register_consumer()

    # 2-channel block, channel 0 loud, channel 1 noise
    stereo = np.zeros((1280, 2), dtype=np.int16)
    stereo[:, 0] = 10000
    stereo[:, 1] = 12345
    stream._process_chunk(stereo)

    chunk = q.get(timeout=0.1)
    assert chunk.shape == (1280,)
    assert np.all(chunk == 10000)  # channel 0 only, channel 1 discarded


def test_audio_stream_consumer_registration(audio_config: AudioConfig) -> None:
    """Test registering and unregistering consumers."""
    with patch("hexa.audio.sd.InputStream"):
        stream = AudioStream(audio_config)

        q1 = stream.register_consumer()
        q2 = stream.register_consumer(maxsize=50)

        assert len(stream._consumers) == 2

        stream.unregister_consumer(q1)
        assert len(stream._consumers) == 1
        assert q2 in stream._consumers

        stream.unregister_consumer(q2)
        assert len(stream._consumers) == 0


def test_audio_stream_chunk_distribution(audio_config: AudioConfig) -> None:
    """Test audio chunks are distributed to all consumers."""
    with patch("hexa.audio.sd.InputStream"):
        stream = AudioStream(audio_config)
        q1 = stream.register_consumer()
        q2 = stream.register_consumer()

        # Simulate audio callback
        test_chunk = np.ones(1280, dtype=np.int16) * 1000
        stream._audio_callback(test_chunk, 1280, None, None)

        # Both consumers should receive the chunk
        chunk1 = q1.get(timeout=0.1)
        chunk2 = q2.get(timeout=0.1)

        np.testing.assert_array_equal(chunk1, test_chunk)
        np.testing.assert_array_equal(chunk2, test_chunk)


def test_audio_stream_queue_overflow(audio_config: AudioConfig) -> None:
    """Test queue overflow handling."""
    with patch("hexa.audio.sd.InputStream"):
        stream = AudioStream(audio_config)
        q = stream.register_consumer(maxsize=2)

        # Fill queue
        q.put(np.zeros(1280, dtype=np.int16))
        q.put(np.zeros(1280, dtype=np.int16))

        # Next chunk should cause overflow
        test_chunk = np.ones(1280, dtype=np.int16)
        stream._audio_callback(test_chunk, 1280, None, None)

        # Queue should still have the old chunks (non-blocking put drops new)
        assert q.qsize() == 2
        assert stream._overflow_count == 1


def test_audio_stream_rms_calculation(audio_config: AudioConfig) -> None:
    """Test RMS calculation for audio level events."""
    with patch("hexa.audio.sd.InputStream"), \
         patch("hexa.audio.emit_event") as mock_emit:

        stream = AudioStream(audio_config)

        # Simulate audio with known amplitude
        # 1000 amplitude -> RMS = 1000
        test_chunk = np.ones(1280, dtype=np.int16) * 1000
        stream._audio_callback(test_chunk, 1280, None, None)

        # Check event was emitted with correct RMS
        mock_emit.assert_called()
        call_args = mock_emit.call_args
        assert call_args[1]["rms_mic"] == 1000.0


def test_audio_stream_mono_conversion(audio_config: AudioConfig) -> None:
    """Test stereo to mono conversion."""
    with patch("hexa.audio.sd.InputStream"):
        stream = AudioStream(audio_config)
        q = stream.register_consumer()

        # Stereo input
        stereo_chunk = np.column_stack([
            np.ones(1280, dtype=np.int16) * 1000,
            np.ones(1280, dtype=np.int16) * 2000
        ]).flatten()  # Flatten to simulate interleaved stereo

        # The callback receives the raw buffer, we need to reshape
        # For testing, we'll call with proper shape
        stereo_proper = np.column_stack([
            np.ones(1280, dtype=np.int16) * 1000,
            np.ones(1280, dtype=np.int16) * 2000
        ])

        stream._audio_callback(stereo_proper, 1280, None, None)

        chunk = q.get(timeout=0.1)
        # Should be mono (single channel)
        assert chunk.ndim == 1
        assert len(chunk) == 1280


def test_playback_handle(audio_config: AudioConfig) -> None:
    """Test PlaybackHandle interrupt and wait."""
    handle = PlaybackHandle()
    handle._stop_event.set()  # Mark as stopped

    handle.interrupt()  # Should not raise
    handle.wait()  # Should not block


def test_global_audio_stream_singleton(audio_config: AudioConfig) -> None:
    """Test global audio stream singleton pattern."""
    shutdown_audio_stream()

    with patch("hexa.audio.sd.InputStream"):
        stream1 = get_audio_stream(audio_config)
        stream2 = get_audio_stream(audio_config)

        assert stream1 is stream2

        shutdown_audio_stream()


def test_audio_stream_dtype_conversion(audio_config: AudioConfig) -> None:
    """Test float32 to int16 conversion in callback."""
    with patch("hexa.audio.sd.InputStream"):
        stream = AudioStream(audio_config)
        q = stream.register_consumer()

        # Float32 input [-1, 1]
        float_chunk = np.ones(1280, dtype=np.float32) * 0.5
        stream._audio_callback(float_chunk, 1280, None, None)

        chunk = q.get(timeout=0.1)
        assert chunk.dtype == np.int16
        # 0.5 * 32767 ≈ 16383
        assert chunk[0] == 16383