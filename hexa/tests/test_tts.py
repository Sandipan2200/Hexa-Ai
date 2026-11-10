"""Tests for text-to-speech."""
from __future__ import annotations

import io
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hexa.tts import TTSManager, TTSConfig, PiperTTS, Pyttsx3TTS


@pytest.fixture
def tts_config() -> TTSConfig:
    return TTSConfig(
        voice="en_GB-alan-medium",
        voice_path="models/en_GB-alan-medium.onnx",
        sample_rate=22050,
        fallback="pyttsx3"
    )


class TestPiperTTS:
    """Tests for PiperTTS."""

    def test_piper_load_voice_not_found(self, tts_config: TTSConfig) -> None:
        """Test Piper voice loading failure."""
        with patch("hexa.tts.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                PiperTTS(tts_config)

    def test_piper_load_voice_success(self, tts_config: TTSConfig) -> None:
        """Test Piper voice loading success."""
        with patch("hexa.tts.Path.exists", return_value=True), \
             patch("piper.PiperVoice.load") as mock_load:

            mock_voice = MagicMock()
            mock_load.return_value = mock_voice

            tts = PiperTTS(tts_config)
            assert tts._voice is mock_voice

    def test_piper_synthesize(self, tts_config: TTSConfig) -> None:
        """Test Piper synthesis."""
        with patch("hexa.tts.Path.exists", return_value=True), \
             patch("piper.PiperVoice.load") as mock_load:

            mock_voice = MagicMock()
            mock_load.return_value = mock_voice

            tts = PiperTTS(tts_config)

            # Mock the synthesize_wav to write to buffer
            def mock_synthesize_wav(text, wf):
                wf.writeframes(b"\x00\x00" * 1000)  # dummy audio

            mock_voice.synthesize_wav = mock_synthesize_wav

            audio_bytes = tts.synthesize("Hello world")

            assert isinstance(audio_bytes, bytes)
            assert len(audio_bytes) > 0


class TestPyttsx3TTS:
    """Tests for Pyttsx3TTS."""

    def test_pyttsx3_init(self, tts_config: TTSConfig) -> None:
        """Test pyttsx3 initialization."""
        with patch("pyttsx3.init") as mock_init:
            mock_engine = MagicMock()
            mock_engine.getProperty.return_value = []
            mock_init.return_value = mock_engine

            tts = Pyttsx3TTS(tts_config)
            assert tts._engine is mock_engine
            mock_engine.setProperty.assert_called()

    def test_pyttsx3_synthesize(self, tts_config: TTSConfig) -> None:
        """Test pyttsx3 synthesis."""
        with patch("pyttsx3.init") as mock_init, \
             patch("tempfile.NamedTemporaryFile") as mock_temp, \
             patch("os.unlink") as mock_unlink:

            mock_engine = MagicMock()
            mock_init.return_value = mock_engine

            # Mock temp file
            mock_temp_file = MagicMock()
            mock_temp_file.name = "/tmp/test.wav"
            mock_temp.return_value.__enter__.return_value = mock_temp_file

            # Mock wave file read
            dummy_wav = io.BytesIO()
            with wave.open(dummy_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                wf.writeframes(b"\x00\x00" * 1000)
            dummy_wav.seek(0)

            with patch("builtins.open", return_value=dummy_wav):
                tts = Pyttsx3TTS(tts_config)
                audio_bytes = tts.synthesize("Hello world")

            assert isinstance(audio_bytes, bytes)
            mock_engine.save_to_file.assert_called_once()
            mock_engine.runAndWait.assert_called_once()
            mock_unlink.assert_called_once_with("/tmp/test.wav")


class TestTTSManager:
    """Tests for TTSManager.

    Both engine classes are mocked at their constructor boundary: pyttsx3 COM
    init and Piper ONNX load cannot run inside unit tests. Everything else
    (TTSManager fallback/dispatch/interrupt logic) is real code.
    """

    @pytest.fixture
    def manager(self, tts_config: TTSConfig):
        """TTSManager with both engines as MagicMocks (no real synth engines)."""
        with patch("hexa.tts.PiperTTS") as mock_piper_class, \
             patch("hexa.tts.Pyttsx3TTS") as mock_fallback_class:
            mock_piper_class.return_value = MagicMock()
            mock_fallback_class.return_value = MagicMock()
            yield TTSManager(tts_config)

    def test_manager_piper_primary(self, manager: TTSManager) -> None:
        """Test Piper chosen as primary engine."""
        assert manager._piper is not None
        assert manager._fallback is not None

    def test_manager_fallback_on_piper_failure(self, manager: TTSManager) -> None:
        """Test fallback used when Piper synthesis fails."""
        manager._piper.synthesize.side_effect = Exception("Piper failed")
        manager._fallback.synthesize.return_value = b"\x00\x00" * 1000

        with patch("hexa.tts.play_audio_bytes_async") as mock_play:
            mock_handle = MagicMock()
            mock_play.return_value = mock_handle
            handle = manager.speak("Hello")

        manager._piper.synthesize.assert_called_once_with("Hello")
        manager._fallback.synthesize.assert_called_once_with("Hello")
        assert handle is mock_handle

    def test_manager_both_fail(self, manager: TTSManager) -> None:
        """Test readable RuntimeError when both engines fail."""
        manager._piper.synthesize.side_effect = Exception("Piper failed")
        manager._fallback.synthesize.side_effect = Exception("Fallback failed")
        with pytest.raises(RuntimeError, match="No TTS engine available"):
            manager.speak("Hello")

    def test_manager_interrupt(self, manager: TTSManager) -> None:
        """Test interrupting current playback stops it immediately."""
        manager._piper.synthesize.return_value = b"\x00\x00" * 1000

        with patch("hexa.tts.play_audio_bytes_async") as mock_play:
            mock_handle = MagicMock()
            mock_play.return_value = mock_handle
            manager.speak("Hello")
            assert manager._current_handle is mock_handle

            manager.interrupt()
            mock_handle.interrupt.assert_called_once()
            assert manager._current_handle is None

    def test_interrupt_clears_no_handle_safely(self, manager: TTSManager) -> None:
        """Interrupting with nothing playing must be a no-op, not a crash."""
        manager.interrupt()  # no handle yet
        assert manager._current_handle is None

    def test_manager_is_speaking(self, manager: TTSManager) -> None:
        manager._piper.synthesize.return_value = b"\x00\x00" * 1000
        assert not manager.is_speaking()

        with patch("hexa.tts.play_audio_bytes_async") as mock_play:
            mock_handle = MagicMock()
            mock_handle.is_playing = True
            mock_play.return_value = mock_handle
            manager.speak("Hello")
            assert manager.is_speaking()

            mock_handle.is_playing = False
            assert not manager.is_speaking()

    def test_manager_wait(self, manager: TTSManager) -> None:
        manager._piper.synthesize.return_value = b"\x00\x00" * 1000
        with patch("hexa.tts.play_audio_bytes_async") as mock_play:
            mock_handle = MagicMock()
            mock_play.return_value = mock_handle
            manager.speak("Hello")
            manager.wait()
            mock_handle.wait.assert_called_once()

    def test_manager_wait_no_playback(self, manager: TTSManager) -> None:
        """Waiting with nothing playing must not block or raise."""
        manager.wait()  # no handle
        assert manager._current_handle is None

    def test_manager_concurrent_speak(self, manager: TTSManager) -> None:
        """A new speak() call must interrupt the previous playback.

        Regression: speak() used to call self.interrupt() while already
        holding the non-reentrant lock -> deadlock on the second turn.
        """
        manager._piper.synthesize.return_value = b"\x00\x00" * 1000

        with patch("hexa.tts.play_audio_bytes_async") as mock_play:
            mock_handle1 = MagicMock()
            mock_handle2 = MagicMock()
            mock_play.side_effect = [mock_handle1, mock_handle2]

            manager.speak("First")
            manager.speak("Second")  # must not deadlock

            mock_handle1.interrupt.assert_called_once()
            assert manager._current_handle is mock_handle2