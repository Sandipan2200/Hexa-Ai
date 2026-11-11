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


def _fake_devices() -> list[dict]:
    """Deterministic device list mirroring the Gate 3 hardware: the same Intel
    SST array exposed via MME (works, 4ch@44.1k) and WDM-KS (serves
    misformatted data, 4ch@16k), plus the Realtek mapper pins."""
    return [
        {"name": "Microsoft Sound Mapper - Input", "max_input_channels": 2,
         "default_samplerate": 44100.0, "hostapi": 0},
        {"name": "Microphone Array (2- Intel SST)", "max_input_channels": 4,
         "default_samplerate": 44100.0, "hostapi": 0},
        {"name": "Microphone (Realtek HD Audio Mic input)", "max_input_channels": 2,
         "default_samplerate": 44100.0, "hostapi": 3},
        {"name": "Microphone Array 3 ()", "max_input_channels": 4,
         "default_samplerate": 16000.0, "hostapi": 3},
    ]


def test_resolver_prefers_non_wdmks_name_match() -> None:
    """A matching name on a healthy host API must win over the WDM-KS pin.

    Gate 3: the array's WDM-KS 16kHz pin serves misformatted data; its MME
    44.1kHz pin works. Both names contain the configured substring, so WDM-KS
    must be deprioritized.
    """
    with patch("hexa.audio.sd.query_devices", return_value=_fake_devices()):
        stream = AudioStream(AudioConfig(
            sample_rate=16000, channels=4, input_device="Microphone Array", use_polling=True))
        idx, ch, native = stream._resolve_input_device()
    assert (idx, ch, native) == (1, 4, 44100)


def test_resolver_valid_int_used_directly() -> None:
    """A still-valid int index is honored as-is (channels clamped to device)."""
    with patch("hexa.audio.sd.query_devices", return_value=_fake_devices()):
        stream = AudioStream(AudioConfig(
            sample_rate=16000, channels=4, input_device=1, use_polling=True))
        idx, ch, native = stream._resolve_input_device()
    assert (idx, ch, native) == (1, 4, 44100)


def test_resolver_stale_int_does_not_crash() -> None:
    """A stale int index (device list shifted between sessions) must fall back
    to resolution instead of failing the stream."""
    with patch("hexa.audio.sd.query_devices", return_value=_fake_devices()):
        stream = AudioStream(AudioConfig(
            sample_rate=16000, channels=4, input_device=21, use_polling=True))
        idx, ch, native = stream._resolve_input_device()
    assert idx is not None          # resolved to some valid input
    assert ch >= 1
    assert native > 0


def test_resolver_wdmks_only_when_no_healthy_match() -> None:
    """If only a WDM-KS device matches the configured name, it is still used
    (better than no input at all) - but only as a last resort."""
    with patch("hexa.audio.sd.query_devices", return_value=_fake_devices()):
        stream = AudioStream(AudioConfig(
            sample_rate=16000, channels=2, input_device="Realtek", use_polling=True))
        idx, ch, native = stream._resolve_input_device()
    assert (idx, ch, native) == (2, 2, 44100)


def test_resolver_no_match_falls_back_to_os_default() -> None:
    """An unresolvable name falls back to the OS default input (device=None).
    Deliberately NOT a native-rate auto-match: that would re-select the
    poisoned WDM-KS 16kHz pin on this hardware."""
    with patch("hexa.audio.sd.query_devices", return_value=_fake_devices()):
        stream = AudioStream(AudioConfig(
            sample_rate=16000, channels=4, input_device="Nonexistent Mic", use_polling=True))
        idx, ch, native = stream._resolve_input_device()
    assert idx is None


def test_audio_stream_polling_mode(audio_config: AudioConfig) -> None:
    """Polling mode lifecycle: no InputStream, NATIVE-rate sd.rec capture,
    mono downmix, resample to 16kHz, chunk distribution.

    Polling + native-rate capture exists because (Gate 1) Intel SST arrays
    drop PortAudio callbacks and (Gate 3) the array's WDM-KS 16kHz pin serves
    misformatted data - valid audio only arrives via its MME 44.1kHz pin.
    """
    devices = _fake_devices()
    poll_config = AudioConfig(
        sample_rate=16000, chunk_size=1280, channels=4,
        input_device="Microphone Array", use_polling=True
    )
    # 800ms window at 44100Hz = 35280 frames; constant 8000 on every channel
    fake_window = np.full((35280, 4), 8000, dtype=np.int16)
    mock_stream = MagicMock()
    mock_stream.read.return_value = (fake_window, False)
    with patch("hexa.audio.sd.query_devices", return_value=devices), \
         patch("hexa.audio.sd.InputStream", return_value=mock_stream) as mock_cls:
        stream = AudioStream(poll_config)
        q = stream.register_consumer()
        stream.start()

        assert stream.is_running()
        assert stream._capture_device == 1          # MME pin, not the WDM-KS pin
        assert stream._capture_native_rate == 44100

        chunk = q.get(timeout=5.0)
        assert chunk.shape == (1280,)
        # Downmix+resample of a constant must stay that constant (unit DC gain)
        assert int(np.abs(chunk.astype(np.int32) - 8000).max()) <= 100

        # ONE persistent pull-mode capture stream at the device's NATIVE rate
        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["samplerate"] == 44100
        mock_stream.read.assert_called()            # blocking read, not sd.rec

        stream.stop()
        assert not stream.is_running()
        # InputStream must never be created in polling mode (callback path)
        assert stream._stream is None


def test_poll_resample_44k_to_16k_sine() -> None:
    """The 44100->16000 resample must preserve tone frequency and window length
    (800ms @44100 = 35280 samples -> exactly 12800 samples @16kHz)."""
    stream = AudioStream(AudioConfig(sample_rate=16000, chunk_size=1280, channels=4))
    stream._capture_device = 1
    stream._capture_channels = 4
    stream._capture_native_rate = 44100

    t = np.arange(35280) / 44100.0
    sine = (np.sin(2 * np.pi * 1000.0 * t) * 8000).astype(np.int16)
    window = np.stack([sine] * 4, axis=1)

    pcm = stream._resample_window(window)
    assert len(pcm) == 12800
    assert pcm.dtype == np.int16

    # The 1kHz tone must survive the resample (dominant spectral peak)
    spec = np.abs(np.fft.rfft(pcm.astype(np.float32)))
    freqs = np.fft.rfftfreq(len(pcm), 1 / 16000)
    assert abs(freqs[int(np.argmax(spec))] - 1000.0) < 50


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


def test_decode_int16_bytes_wav_header_stripped() -> None:
    """WAV containers must be decoded as PCM data, header stripped (Gate 2).

    Piper/pyttsx3 produce complete WAV files; playing them via np.frombuffer
    decoded the 44-byte RIFF header as audio - an audible click before every
    utterance.
    """
    import io
    import wave

    from hexa.audio import _decode_int16_bytes

    pcm = np.arange(0, 800, dtype=np.int16)  # known ramp
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(pcm.tobytes())
    wav_bytes = buf.getvalue()

    data, rate = _decode_int16_bytes(wav_bytes)
    assert rate == 22050
    assert len(data) == len(pcm)  # header stripped, no +22 garbage samples
    np.testing.assert_allclose(data, pcm / 32767.0, atol=1e-6)

    # Raw PCM path still works (no container)
    raw, rate_raw = _decode_int16_bytes(pcm.tobytes())
    assert rate_raw == -1
    np.testing.assert_allclose(raw, pcm / 32767.0, atol=1e-6)


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