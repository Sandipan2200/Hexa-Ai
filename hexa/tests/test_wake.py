"""Tests for wake word detection and hotkey."""
from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pynput import keyboard

from hexa.wake import WakeWordDetector, WakeWordConfig, HotkeyListener, HotkeyConfig


@pytest.fixture
def wake_config() -> WakeWordConfig:
    return WakeWordConfig(
        enabled=True,
        model_name="hey_jarvis",
        inference_framework="onnx",
        threshold=0.5,
        debounce_time=1.0
    )


@pytest.fixture
def hotkey_config() -> HotkeyConfig:
    return HotkeyConfig(enabled=True, key="<ctrl>+<alt>+<space>")


class TestWakeWordDetector:
    """Tests for WakeWordDetector."""

    def test_wake_detector_disabled(self, wake_config: WakeWordConfig) -> None:
        """Test detector respects enabled flag."""
        wake_config.enabled = False
        detector = WakeWordDetector(wake_config, lambda: None)
        detector.start()
        # Should not start thread
        assert not detector._running
        detector.stop()

    def test_wake_detector_model_loading(self, wake_config: WakeWordConfig) -> None:
        """Test model loading."""
        with patch("openwakeword.Model") as mock_model_class:
            mock_model = MagicMock()
            mock_model.model_inputs = {"hey_jarvis": 1280}
            mock_model.models = {"hey_jarvis": mock_model}
            mock_model.predict.return_value = {"hey_jarvis": 0.3}  # Below threshold
            mock_model_class.return_value = mock_model

            detector = WakeWordDetector(wake_config, lambda: None)

            with patch("hexa.wake.get_audio_stream") as mock_get_stream:
                mock_stream = MagicMock()
                mock_queue = queue.Queue()
                mock_stream.register_consumer.return_value = mock_queue
                mock_get_stream.return_value = mock_stream

                detector.start()
                time.sleep(0.1)
                detector.stop()

            assert mock_model_class.called

    def test_wake_detector_threshold(self, wake_config: WakeWordConfig) -> None:
        """Test wake word threshold."""
        wake_callback = MagicMock()

        with patch("openwakeword.Model") as mock_model_class:
            mock_model = MagicMock()
            mock_model.model_inputs = {"hey_jarvis": 1280}
            mock_model.models = {"hey_jarvis": mock_model}
            # Return score above threshold
            mock_model.predict.return_value = {"hey_jarvis": 0.8}
            mock_model_class.return_value = mock_model

            detector = WakeWordDetector(wake_config, wake_callback)

            with patch("hexa.wake.get_audio_stream") as mock_get_stream:
                mock_stream = MagicMock()
                mock_queue = queue.Queue()
                mock_stream.register_consumer.return_value = mock_queue
                mock_get_stream.return_value = mock_stream

                detector.start()

                # Put audio chunk that triggers detection
                mock_queue.put(np.ones(1280, dtype=np.int16))

                time.sleep(0.2)
                detector.stop()

            wake_callback.assert_called_once()

    def test_wake_detector_debounce(self, wake_config: WakeWordConfig) -> None:
        """Test debounce prevents multiple rapid detections."""
        wake_callback = MagicMock()

        with patch("openwakeword.Model") as mock_model_class:
            mock_model = MagicMock()
            mock_model.model_inputs = {"hey_jarvis": 1280}
            mock_model.models = {"hey_jarvis": mock_model}
            mock_model.predict.return_value = {"hey_jarvis": 0.8}
            mock_model_class.return_value = mock_model

            detector = WakeWordDetector(wake_config, wake_callback)

            with patch("hexa.wake.get_audio_stream") as mock_get_stream:
                mock_stream = MagicMock()
                mock_queue = queue.Queue()
                mock_stream.register_consumer.return_value = mock_queue
                mock_get_stream.return_value = mock_stream

                detector.start()

                # Put multiple chunks quickly
                for _ in range(5):
                    mock_queue.put(np.ones(1280, dtype=np.int16))
                    time.sleep(0.01)

                time.sleep(0.2)
                detector.stop()

            # Should only trigger once due to debounce
            assert wake_callback.call_count == 1

    def test_wake_detector_consumer_registration(self, wake_config: WakeWordConfig) -> None:
        """Test detector registers with audio stream."""
        with patch("openwakeword.Model"), \
             patch("hexa.wake.get_audio_stream") as mock_get_stream:

            mock_stream = MagicMock()
            mock_queue = queue.Queue()
            mock_stream.register_consumer.return_value = mock_queue
            mock_get_stream.return_value = mock_stream

            detector = WakeWordDetector(wake_config, lambda: None)
            detector.start()

            mock_stream.register_consumer.assert_called_once()
            assert detector._audio_queue is mock_queue

            detector.stop()
            mock_stream.unregister_consumer.assert_called_once_with(mock_queue)


class TestHotkeyListener:
    """Tests for HotkeyListener."""

    def test_hotkey_disabled(self, hotkey_config: HotkeyConfig) -> None:
        """Test hotkey respects enabled flag."""
        hotkey_config.enabled = False
        listener = HotkeyListener(hotkey_config, lambda: None, lambda: None)
        listener.start()
        # Should not start listener
        assert listener._listener is None
        listener.stop()

    def test_hotkey_press_release(self, hotkey_config: HotkeyConfig) -> None:
        """True combo-state tracking: on_press fires once when the full
        combination becomes held, on_release fires when any key of it is
        released (Gate 5: GlobalHotKeys never fired on release, breaking
        push-to-talk)."""
        press_cb = MagicMock()
        release_cb = MagicMock()

        with patch("hexa.wake.keyboard.Listener") as mock_listener_class:
            # No real OS hook; canonical() = identity on the listener INSTANCE
            # (pynput's canonical is an instance method - Gate 5 lesson)
            mock_listener_class.return_value.canonical = lambda k: k

            listener = HotkeyListener(hotkey_config, press_cb, release_cb)
            listener.start()

            ctrl, alt, space = keyboard.HotKey.parse("<ctrl>+<alt>+<space>")
            # Activation must fire exactly when the LAST combo key is pressed
            listener._on_key_press(ctrl)
            press_cb.assert_not_called()
            listener._on_key_press(alt)
            press_cb.assert_not_called()
            listener._on_key_press(space)
            press_cb.assert_called_once()

            # Re-pressing an already-held key must NOT re-fire
            listener._on_key_press(space)
            assert press_cb.call_count == 1

            # Releasing ANY combo key deactivates -> release callback
            listener._on_key_release(space)
            release_cb.assert_called_once()

            # A second press cycle works (state fully reset on release)
            listener._on_key_press(space)
            assert press_cb.call_count == 2
            listener._on_key_release(alt)
            assert release_cb.call_count == 2

            listener.stop()
        mock_listener_class.return_value.stop.assert_called_once()

    def test_hotkey_ignores_unrelated_keys(self, hotkey_config: HotkeyConfig) -> None:
        """Keys outside the combination must not trigger press or release."""
        press_cb = MagicMock()
        release_cb = MagicMock()

        with patch("hexa.wake.keyboard.Listener") as mock_listener_class:
            mock_listener_class.return_value.canonical = lambda k: k

            listener = HotkeyListener(hotkey_config, press_cb, release_cb)
            listener.start()

            ctrl, alt, space = keyboard.HotKey.parse("<ctrl>+<alt>+<space>")
            listener._on_key_press(space)          # partial combo (no ctrl/alt)
            press_cb.assert_not_called()
            listener._on_key_release(space)
            release_cb.assert_not_called()

            # holding shift (unrelated) then releasing: no state corruption
            listener._on_key_press("unrelated-key")
            listener._on_key_release("unrelated-key")
            press_cb.assert_not_called()
            release_cb.assert_not_called()

            listener.stop()

    def test_hotkey_stop(self, hotkey_config: HotkeyConfig) -> None:
        """Test hotkey listener stop."""
        with patch("hexa.wake.keyboard.Listener") as mock_listener_class:
            mock_listener_class.return_value.canonical = lambda k: k

            listener = HotkeyListener(hotkey_config, lambda: None, lambda: None)
            listener.start()
            listener.stop()

            mock_listener_class.return_value.stop.assert_called_once()