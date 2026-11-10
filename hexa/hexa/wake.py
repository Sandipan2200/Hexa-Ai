"""Wake word detection and hotkey handling for Hexa."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from pynput import keyboard

from hexa.audio import get_audio_stream
from hexa.config import get_config
from hexa.events import emit_event, EventType

logger = logging.getLogger(__name__)


@dataclass
class WakeWordConfig:
    """Wake word configuration."""
    enabled: bool = True
    model_name: str = "hey_jarvis"
    inference_framework: str = "onnx"
    threshold: float = 0.5
    debounce_time: float = 1.0


@dataclass
class HotkeyConfig:
    """Hotkey configuration."""
    enabled: bool = True
    key: str = "<ctrl>+<alt>+<space>"


class WakeWordDetector:
    """
    Wake word detection using openWakeWord.
    Consumes audio chunks from the shared audio stream.
    """

    def __init__(self, config: WakeWordConfig, on_wake: Callable[[], None]) -> None:
        self.config = config
        self._on_wake = on_wake
        self._model = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._audio_queue: Optional[queue.Queue] = None
        self._last_detection = 0.0
        self._audio_stream = None

    def _load_model(self) -> None:
        """Load the openWakeWord model."""
        try:
            from openwakeword import Model
            self._model = Model(
                wakeword_models=[self.config.model_name],
                inference_framework=self.config.inference_framework
            )
            logger.info("Wake word model loaded: %s", self.config.model_name)
        except Exception as e:
            logger.exception("Failed to load wake word model")
            emit_event(EventType.ERROR, source="wake_word", error=f"Model load failed: {e}")
            raise

    def start(self) -> None:
        """Start wake word detection."""
        if self._running:
            return

        if not self.config.enabled:
            logger.info("Wake word detection disabled in config")
            return

        self._load_model()

        # Get shared audio stream and register consumer
        self._audio_stream = get_audio_stream()
        self._audio_queue = self._audio_stream.register_consumer(maxsize=50)

        self._running = True
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        logger.info("Wake word detection started (threshold=%.2f)", self.config.threshold)

    def stop(self) -> None:
        """Stop wake word detection."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._audio_queue and self._audio_stream:
            self._audio_stream.unregister_consumer(self._audio_queue)
        logger.info("Wake word detection stopped")

    def _detection_loop(self) -> None:
        """Main detection loop."""
        chunk_size = self._model.model_inputs[list(self._model.models.keys())[0]]

        while self._running:
            try:
                # Get audio chunk from queue
                chunk = self._audio_queue.get(timeout=0.1)
                if chunk is None:
                    continue

                # Ensure correct frame size for model
                if len(chunk) != chunk_size:
                    # Resample or pad/truncate if needed
                    if len(chunk) > chunk_size:
                        chunk = chunk[:chunk_size]
                    else:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')

                # Run prediction
                predictions = self._model.predict(chunk[np.newaxis, :])

                # Check all models
                for model_name, score in predictions.items():
                    if score >= self.config.threshold:
                        now = time.time()
                        if now - self._last_detection >= self.config.debounce_time:
                            self._last_detection = now
                            logger.info("Wake word detected: %s (score=%.3f)", model_name, score)
                            emit_event(EventType.WAKE_DETECTED, model=model_name, score=float(score))
                            self._on_wake()
                        break

            except queue.Empty:
                continue
            except Exception as e:
                logger.exception("Error in wake word detection loop")
                emit_event(EventType.ERROR, source="wake_word", error=str(e))
                time.sleep(0.1)


class HotkeyListener:
    """Global hotkey listener for push-to-talk."""

    def __init__(self, config: HotkeyConfig, on_press: Callable[[], None], on_release: Callable[[], None]) -> None:
        self.config = config
        self._on_press = on_press
        self._on_release = on_release
        self._listener: Optional[keyboard.GlobalHotKeys] = None
        self._pressed = False

    def _parse_key(self) -> dict:
        """Parse key string to pynput format."""
        key_str = self.config.key
        # Convert <ctrl>+<alt>+<space> to {'<ctrl>+<alt>+<space>': callback}
        # pynput uses format like '<ctrl>+<alt>+<space>'
        return {key_str: self._on_hotkey}

    def _on_hotkey(self) -> None:
        """Handle hotkey press/release."""
        if not self._pressed:
            self._pressed = True
            logger.debug("Hotkey pressed")
            emit_event(EventType.HOTKEY_PRESSED)
            self._on_press()
        else:
            self._pressed = False
            logger.debug("Hotkey released")
            self._on_release()

    def start(self) -> None:
        """Start listening for hotkey."""
        if not self.config.enabled:
            logger.info("Hotkey disabled in config")
            return

        try:
            hotkey_map = self._parse_key()
            self._listener = keyboard.GlobalHotKeys(hotkey_map)
            self._listener.start()
            logger.info("Hotkey listener started: %s", self.config.key)
        except Exception as e:
            logger.exception("Failed to start hotkey listener")
            emit_event(EventType.ERROR, source="hotkey", error=f"Hotkey start failed: {e}")

    def stop(self) -> None:
        """Stop hotkey listener."""
        if self._listener:
            self._listener.stop()
            self._listener = None
        logger.info("Hotkey listener stopped")


# Import queue here to avoid circular import
import queue