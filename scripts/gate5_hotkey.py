#!/usr/bin/env python3
"""Gate 5: hotkey Ctrl+Alt+Space PRESS/RELEASE diagnostic (real pynput listener).

Press AND RELEASE Ctrl+Alt+Space THREE times, holding ~1s each time.
The script logs every callback invocation with timestamps and classifies the
internal toggle's interpretation.

Expected if press+release both fire: 6 invocations for 3 press cycles.
pynput source inspection: GlobalHotKeys fires ONLY on activation
(HotKey.release() never invokes the callback) -> expect only 3 invocations,
with every 2nd press misread as a release (toggle broken).
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, r"D:\work - personal\Hexa Ai\hexa")
from hexa.config import load_config, get_config

load_config()
cfg = get_config()

from hexa.wake import HotkeyListener, HotkeyConfig
from hexa.events import EventType, get_event_bus


def main() -> None:
    events_log: list[tuple[float, str]] = []
    bus_hits: list[tuple[float, str]] = []

    listener = HotkeyListener(
        config=HotkeyConfig(enabled=cfg.hotkey.enabled, key=cfg.hotkey.key),
        on_press=lambda: events_log.append((time.perf_counter(), "pipeline on_press()")),
        on_release=lambda: events_log.append((time.perf_counter(), "pipeline on_release()")),
    )
    counter = {"press": 0, "release": 0}

    def press_probe() -> None:
        counter["press"] += 1
        events_log.append((time.perf_counter(), f"  [true PRESS #{counter['press']}]"))

    def release_probe() -> None:
        counter["release"] += 1
        events_log.append((time.perf_counter(), f"  [true RELEASE #{counter['release']}]"))

    listener._on_press = press_probe
    listener._on_release = release_probe

    bus = get_event_bus()
    bus.subscribe(EventType.HOTKEY_PRESSED,
                  lambda e: bus_hits.append((time.perf_counter(), "HOTKEY_PRESSED emitted")))

    listener.start()
    print(f"Hotkey listener started for: {cfg.hotkey.key}")
    print("\nPRESS AND RELEASE Ctrl+Alt+Space THREE times (hold ~1s each time).")
    t0 = time.perf_counter()

    deadline = t0 + 30.0
    while time.perf_counter() < deadline and counter["release"] < 3:
        time.sleep(0.2)
    time.sleep(1.5)  # catch trailing events
    listener.stop()

    print("\nTimeline (relative seconds):")
    for t, detail in events_log:
        print(f"  +{t - t0:6.2f}s  {detail}")
    for t, detail in bus_hits:
        print(f"  +{t - t0:6.2f}s  {detail}")

    print(f"\ntrue presses={counter['press']} true releases={counter['release']} "
          f"(expected 3 of each for 3 press-release cycles)")
    ok = counter["press"] == 3 and counter["release"] == 3 and len(bus_hits) == 3
    # every press must precede its release in the log order
    seq = [d.strip() for _, d in events_log]
    alt_ok = all(
        (seq[i].startswith("[true PRESS") if i % 2 == 0 else seq[i].startswith("[true RELEASE"))
        for i in range(min(6, len(seq)))
    )
    print("alternating press->release order:", "OK" if alt_ok else "CHECK")
    print("GATE 5:", "PASS" if (ok and alt_ok) else "CHECK OUTPUT ABOVE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("gate5").exception("Gate 5 failed")
        sys.exit(1)
