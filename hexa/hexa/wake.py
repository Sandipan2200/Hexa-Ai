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
        try:
            chunk_size = self._model.model_inputs[list(self._model.models.keys())[0]]
        except (IndexError, AttributeError):
            # Model exposes no input specs (e.g. no wake models loaded):
            # exit the thread cleanly instead of crashing it with a traceback.
            logger.error("Wake word model exposes no input specs; detection disabled")
            return

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
    """Global hotkey listener for push-to-talk.

    Gate 5 root cause: pynput's GlobalHotKeys fires its callback ONLY on
    activation (HotKey.release() never invokes the callback), so the previous
    toggle design misread every 2nd press as a release and could never detect
    an actual release. This implementation tracks the combination state with a
    raw Listener and fires on_press()/on_release() at true combo boundaries.
    """

    def __init__(self, config: HotkeyConfig, on_press: Callable[[], None], on_release: Callable[[], None]) -> None:
        self.config = config
        self._on_press = on_press
        self._on_release = on_release
        self._listener: Optional[keyboard.Listener] = None
        self._combo: Optional[frozenset] = None  # canonical keys of the combination
        self._held: set = set()                  # canonical keys currently held
        self._active = False                     # full combination held right now

    def _parse_combo(self) -> frozenset:
        """Parse the configured key string into canonical pynput keys."""
        return frozenset(keyboard.HotKey.parse(self.config.key))

    def _canonical(self, key):
        """Normalize a listener key event to the form HotKey.parse produced.

        pynput's canonical() is an INSTANCE method (it needs the listener's
        scan-code translator); calling it on the class raises, which silently
        disabled the whole hotkey (Gate 5: 0 presses on hardware). Use the live
        listener instance, falling back to the raw key if unavailable.
        """
        listener = self._listener
        if listener is not None:
            try:
                return listener.canonical(key)
            except Exception:
                logger.debug("Hotkey: canonicalization failed for %r", key, exc_info=True)
        return key

    def _on_key_press(self, key, *args) -> None:
        """Key pressed: activate when the full combination becomes held."""
        k = self._canonical(key)
        if k in self._combo and k not in self._held:
            self._held.add(k)
            if self._held == self._combo and not self._active:
                self._active = True
                logger.debug("Hotkey pressed")
                emit_event(EventType.HOTKEY_PRESSED)
                self._on_press()

    def _on_key_release(self, key, *args) -> None:
        """Key released: deactivate as soon as the combination is no longer held."""
        k = self._canonical(key)
        if k in self._held:
            self._held.discard(k)
            if self._active and self._held != self._combo:
                self._active = False
                logger.debug("Hotkey released")
                self._on_release()

    def start(self) -> None:
        """Start listening for hotkey."""
        if not self.config.enabled:
            logger.info("Hotkey disabled in config")
            return

        try:
            self._combo = self._parse_combo()
            self._held = set()
            self._active = False
            self._listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
            )
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