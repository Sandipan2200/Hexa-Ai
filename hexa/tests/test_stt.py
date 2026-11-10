"""Tests for speech-to-text and VAD."""
from __future__ import annotations

import queue
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hexa.stt import SpeechRecognizer, STTConfig, VADConfig, SileroVAD, WhisperTranscriber


@pytest.fixture
def stt_config() -> STTConfig:
    return STTConfig(
        model="small.en",
        device="cpu",
        compute_type="int8",
        language="en",
        beam_size=5,
        vad_threshold=0.5
    )


@pytest.fixture
def vad_config() -> VADConfig:
    return VADConfig(
        enabled=True,
        silence_timeout=1.5,
        min_speech_duration=0.5
    )


class TestSileroVAD:
    """Tests for SileroVAD."""

    def test_vad_energy_fallback(self) -> None:
        """Test energy-based VAD fallback."""
        vad = SileroVAD(threshold=0.5)
        # Force fallback by setting model to None
        vad._model = None

        # Silence chunk - very low energy
        silence = np.zeros(1280, dtype=np.int16)
        prob = vad.get_speech_prob(silence)
        assert prob < 0.01  # Near zero

        # Loud chunk - use max int16 value to ensure max energy
        loud = np.full(1280, 32767, dtype=np.int16)
        prob = vad.get_speech_prob(loud)
        # 32767^2 / 1000 = 1073676289 / 1000 = 1073676 -> capped at 1.0
        assert prob == 1.0

    def test_vad_with_model_mock(self) -> None:
        """Test VAD with mocked model."""
        vad = SileroVAD(threshold=0.5)
        vad._model = MagicMock()
        vad._model.return_value = MagicMock(item=MagicMock(return_value=0.8))

        chunk = np.ones(512, dtype=np.int16) * 10000
        prob = vad.get_speech_prob(chunk)
        assert prob == 0.8


class TestWhisperTranscriber:
    """Tests for WhisperTranscriber."""

    def test_transcriber_load_model(self, stt_config: STTConfig) -> None:
        """Test model loading."""
        with patch("faster_whisper.WhisperModel") as mock_model_class:
            transcriber = WhisperTranscriber(stt_config)
            mock_model_class.assert_called_once_with(
                "small.en", device="cpu", compute_type="int8", local_files_only=True
            )

    def test_transcriber_empty_chunks(self, stt_config: STTConfig) -> None:
        """Test transcription with empty chunks."""
        with patch("faster_whisper.WhisperModel"):
            transcriber = WhisperTranscriber(stt_config)
            result = transcriber.transcribe([])
            assert result == ""

    def test_transcriber_success(self, stt_config: STTConfig) -> None:
        """Test successful transcription."""
        with patch("faster_whisper.WhisperModel") as mock_model_class:
            mock_model = MagicMock()
            mock_segment = MagicMock()
            mock_segment.text = "hello world"
            mock_model.transcribe.return_value = ([mock_segment], MagicMock())
            mock_model_class.return_value = mock_model

            transcriber = WhisperTranscriber(stt_config)
            chunks = [np.ones(16000, dtype=np.int16) * 1000]  # 1 second
            result = transcriber.transcribe(chunks)

            assert result == "hello world"
            mock_model.transcribe.assert_called_once()

    def test_transcriber_error_handling(self, stt_config: STTConfig) -> None:
        """Test transcription error handling."""
        with patch("faster_whisper.WhisperModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model.transcribe.side_effect = Exception("Model error")
            mock_model_class.return_value = mock_model

            transcriber = WhisperTranscriber(stt_config)
            chunks = [np.ones(16000, dtype=np.int16) * 1000]
            result = transcriber.transcribe(chunks)

            assert result == ""


class TestSpeechRecognizer:
    """Tests for SpeechRecognizer."""

    def test_recognizer_start_stop(self, stt_config: STTConfig, vad_config: VADConfig) -> None:
        """Test recognizer lifecycle."""
        transcript_cb = MagicMock()

        with patch("faster_whisper.WhisperModel"):
            recognizer = SpeechRecognizer(stt_config, vad_config, transcript_cb)

            with patch("hexa.stt.get_audio_stream") as mock_get_stream:
                mock_stream = MagicMock()
                mock_queue = queue.Queue()
                mock_stream.register_consumer.return_value = mock_queue
                mock_get_stream.return_value = mock_stream

                recognizer.start()
                assert recognizer._running
                assert recognizer._thread is not None

                recognizer.stop()
                assert not recognizer._running
                mock_stream.unregister_consumer.assert_called_once()

    def test_recognizer_vad_speech_detection(self, stt_config: STTConfig, vad_config: VADConfig) -> None:
        """Test VAD speech detection and transcript callback."""
        transcript_cb = MagicMock()

        with patch("faster_whisper.WhisperModel") as mock_model_class:
            mock_model = MagicMock()
            mock_segment = MagicMock()
            mock_segment.text = "test speech"
            mock_model.transcribe.return_value = ([mock_segment], MagicMock())
            mock_model_class.return_value = mock_model

            # Mock VAD to return high probability for speech, low for silence
            def mock_get_speech_prob(chunk):
                # If chunk has high amplitude, it's speech
                if np.mean(np.abs(chunk)) > 1000:
                    return 0.8
                return 0.1

            with patch("hexa.stt.get_audio_stream") as mock_get_stream:
                mock_stream = MagicMock()
                mock_queue = queue.Queue()
                mock_stream.register_consumer.return_value = mock_queue
                mock_get_stream.return_value = mock_stream

                with patch("faster_whisper.WhisperModel") as mock_model_class:
                    mock_model = MagicMock()
                    mock_segment = MagicMock()
                    mock_segment.text = "test speech"
                    mock_model.transcribe.return_value = ([mock_segment], MagicMock())
                    mock_model_class.return_value = mock_model

                    recognizer = SpeechRecognizer(stt_config, vad_config, transcript_cb)
                    recognizer._vad.get_speech_prob = mock_get_speech_prob

                    recognizer.start()

                    # Feed speech chunks (high amplitude)
                    speech_chunk = np.full(1280, 10000, dtype=np.int16)
                    for _ in range(10):  # Enough for min_speech_duration
                        mock_queue.put(speech_chunk)

                    # Feed silence to trigger end (low amplitude)
                    silence_chunk = np.zeros(1280, dtype=np.int16)
                    for _ in range(int(1.5 * 1000 / 80) + 1):  # silence_timeout / chunk_duration
                        mock_queue.put(silence_chunk)

                    time.sleep(0.5)
                    recognizer.stop()

                transcript_cb.assert_called_once_with("test speech")

    def test_recognizer_short_speech_discarded(self, stt_config: STTConfig, vad_config: VADConfig) -> None:
        """Test speech shorter than min_speech_duration is discarded."""
        transcript_cb = MagicMock()

        with patch("faster_whisper.WhisperModel") as mock_model_class:
            mock_model = MagicMock()
            mock_segment = MagicMock()
            mock_segment.text = "short"
            mock_model.transcribe.return_value = ([mock_segment], MagicMock())
            mock_model_class.return_value = mock_model

            recognizer = SpeechRecognizer(stt_config, vad_config, transcript_cb)
            recognizer._vad.get_speech_prob = MagicMock(return_value=0.8)

            with patch("hexa.stt.get_audio_stream") as mock_get_stream:
                mock_stream = MagicMock()
                mock_queue = queue.Queue()
                mock_stream.register_consumer.return_value = mock_queue
                mock_get_stream.return_value = mock_stream

                recognizer.start()

                # Feed only a few speech chunks (less than min_speech_chunks)
                speech_chunk = np.ones(1280, dtype=np.int16) * 10000
                for _ in range(2):  # Less than min_speech_chunks (6)
                    mock_queue.put(speech_chunk)

                # Feed silence
                silence_chunk = np.zeros(1280, dtype=np.int16)
                for _ in range(int(1.5 * 1000 / 80) + 1):
                    mock_queue.put(silence_chunk)

                time.sleep(0.5)
                recognizer.stop()

            # Should not call transcript callback for too-short speech
            transcript_cb.assert_not_called()

    def test_recognizer_vad_disabled(self, stt_config: STTConfig, vad_config: VADConfig) -> None:
        """Test recognizer with VAD disabled."""
        vad_config.enabled = False
        transcript_cb = MagicMock()

        with patch("faster_whisper.WhisperModel"):
            recognizer = SpeechRecognizer(stt_config, vad_config, transcript_cb)

            with patch("hexa.stt.get_audio_stream") as mock_get_stream:
                mock_stream = MagicMock()
                mock_queue = queue.Queue()
                mock_stream.register_consumer.return_value = mock_queue
                mock_get_stream.return_value = mock_stream

                recognizer.start()
                assert recognizer._running
                recognizer.stop()

    def test_recognizer_transcript_empty(self, stt_config: STTConfig, vad_config: VADConfig) -> None:
        """Test handling of empty transcription result."""
        transcript_cb = MagicMock()

        with patch("faster_whisper.WhisperModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = ([], MagicMock())  # Empty result
            mock_model_class.return_value = mock_model

            recognizer = SpeechRecognizer(stt_config, vad_config, transcript_cb)
            recognizer._vad.get_speech_prob = MagicMock(return_value=0.8)

            with patch("hexa.stt.get_audio_stream") as mock_get_stream:
                mock_stream = MagicMock()
                mock_queue = queue.Queue()
                mock_stream.register_consumer.return_value = mock_queue
                mock_get_stream.return_value = mock_stream

                recognizer.start()

                # Feed speech then silence
                speech_chunk = np.ones(1280, dtype=np.int16) * 10000
                for _ in range(10):
                    mock_queue.put(speech_chunk)

                silence_chunk = np.zeros(1280, dtype=np.int16)
                for _ in range(int(1.5 * 1000 / 80) + 1):
                    mock_queue.put(silence_chunk)

                time.sleep(0.5)
                recognizer.stop()

            # Should not call callback for empty result
            transcript_cb.assert_not_called()