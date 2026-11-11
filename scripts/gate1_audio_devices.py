#!/usr/bin/env python3
"""Gate 1: Audio Devices Diagnostic (final version, uses Hexa's real audio stack).

Lists input/output devices, shows defaults and config device.
Records 3 seconds via Hexa's AudioStream (polling mode), prints peak/RMS.
Plays a 440 Hz tone, then plays the recording back.
"""
from __future__ import annotations

import logging
import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, r"D:\work - personal\Hexa Ai\hexa")
from hexa.config import get_config, load_config


def list_devices():
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    default_output = sd.default.device[1]

    load_config()
    config = get_config()
    config_input = config.audio.input_device

    print("\n" + "=" * 70)
    print("AUDIO DEVICES")
    print("=" * 70)
    print(f"\nDefault input device:  [{default_input}] {devices[default_input]['name']}")
    print(f"Default output device: [{default_output}] {devices[default_output]['name']}")
    if config_input is not None:
        print(f"Config input device:   [{config_input}] {devices[config_input]['name']} "
              f"(hostapi: {sd.query_hostapis(devices[config_input]['hostapi'])['name']})")
        print(f"Config channels:       {config.audio.channels}")
        print(f"Config polling mode:   {config.audio.use_polling}")
    else:
        print("Config input device:   [None] (uses default)")
        print(f"Config channels:       {config.audio.channels}")
        print(f"Config polling mode:   {config.audio.use_polling}")

    print("\n--- INPUT DEVICES ---")
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = " >>> DEFAULT" if i == default_input else ""
            marker += " >>> CONFIG" if i == config_input else ""
            print(f"  [{i}] {d['name']}  in:{d['max_input_channels']} out:{d['max_output_channels']} "
                  f"sr:{d['default_samplerate']:.0f}{marker}")

    print("\n--- OUTPUT DEVICES ---")
    for i, d in enumerate(devices):
        if d["max_output_channels"] > 0:
            marker = " >>> DEFAULT" if i == default_output else ""
            print(f"  [{i}] {d['name']}  out:{d['max_output_channels']} "
                  f"sr:{d['default_samplerate']:.0f}{marker}")


def record_via_hexa(duration=3.0):
    """Record via Hexa's real AudioStream (the same one voice mode uses)."""
    from hexa.audio import get_audio_stream, shutdown_audio_stream

    print("\nRecording %.1fs via Hexa AudioStream (real voice-mode stack)..." % duration)
    print("  SPEAK NOW!")

    stream = get_audio_stream()
    q = stream.register_consumer(maxsize=500)
    stream.start()
    try:
        sd.sleep(int(duration * 1000))
    finally:
        stream.stop()

    chunks = []
    while not q.empty():
        chunks.append(q.get_nowait())
    shutdown_audio_stream()

    if not chunks:
        print("  FAILED: no chunks received")
        return None

    audio = np.concatenate(chunks)
    sr = get_config().audio.sample_rate

    peak = int(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    peak_db = 20 * np.log10(peak / 32767.0) if peak > 0 else float("-inf")
    rms_db = 20 * np.log10(rms / 32767.0) if rms > 0 else float("-inf")

    print(f"  Chunks: {len(chunks)}  Samples: {len(audio)} ({len(audio)/sr:.2f}s)")
    print(f"  Peak: {peak} ({peak_db:.1f} dBFS)")
    print(f"  RMS:  {rms:.1f} ({rms_db:.1f} dBFS)")
    if peak < 100:
        print("  FAIL: signal is essentially silent")
    elif peak < 1000:
        print("  WARN: low signal")
    else:
        print("  OK: real microphone signal captured")

    return audio, sr


def save_wav(recording, sample_rate, path):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(recording.tobytes())
    print(f"  Saved {path}")


def play_tone(frequency=440, duration=1.0):
    devices = sd.query_devices()
    default_output = sd.default.device[1]
    output_sr = int(devices[default_output]["default_samplerate"])
    print(f"\nPlaying {frequency} Hz tone for {duration}s "
          f"(output device {default_output} @ {output_sr}Hz)...")
    t = np.linspace(0, duration, int(output_sr * duration), endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    sd.play(tone, output_sr, blocking=True)
    print("  Done")


def play_recording(recording, sample_rate):
    print(f"\nPlaying back recording ({len(recording)/sample_rate:.2f}s)...")
    sd.play(recording.astype(np.float32) / 32767.0, sample_rate, blocking=True)
    print("  Done")


def main():
    print("=" * 70)
    print("GATE 1: AUDIO DEVICES DIAGNOSTIC")
    print("=" * 70)

    list_devices()

    result = record_via_hexa(duration=3.0)
    if result is None:
        sys.exit(1)
    audio, sr = result
    save_wav(audio, sr, Path("gate1_recording.wav"))

    play_tone(440, 1.0)
    play_recording(audio, sr)

    print("\n" + "=" * 70)
    print("GATE 1 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Gate 1 failed")
        sys.exit(1)
