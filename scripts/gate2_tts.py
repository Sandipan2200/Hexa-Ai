#!/usr/bin/env python3
"""Gate 2: TTS Diagnostic (real hardware, no mocks).

1. Speak "Hello, I am Hexa" with Piper; print voice sample rate vs device rate.
2. Inspect the synthesized WAV (header validity, duration).
3. Test pyttsx3 fallback.
4. Test interrupt (stop mid-sentence).
"""
from __future__ import annotations

import io
import json
import logging
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gate2")

sys.path.insert(0, r"D:\work - personal\Hexa Ai\hexa")
from hexa.config import load_config, get_config

load_config()
cfg = get_config()


def print_rates():
    print("=" * 70)
    print("SAMPLE RATES")
    print("=" * 70)
    voice_json = Path(cfg.tts.voice_path)
    if not voice_json.is_absolute():
        voice_json = Path(r"D:\work - personal\Hexa Ai") / cfg.tts.voice_path
    voice_json = voice_json.with_suffix(".onnx.json")
    voice_rate = json.loads(voice_json.read_text(encoding="utf-8"))["audio"]["sample_rate"]
    out_dev = sd.default.device[1]
    dev = sd.query_devices(out_dev)
    print(f"  Piper voice native rate: {voice_rate} Hz")
    print(f"  Config tts.sample_rate:  {cfg.tts.sample_rate} Hz")
    print(f"  Output device rate:      {int(dev['default_samplerate'])} Hz "
          f"([ {out_dev}] {dev['name']})")
    match = "MATCH" if voice_rate == cfg.tts.sample_rate else "MISMATCH - audio plays at wrong speed!"
    print(f"  voice vs config: {match}")
    print(f"  (Playback requests {cfg.tts.sample_rate} Hz; PortAudio/WASAPI resamples to the device rate.)")


def test_piper():
    print("\n" + "=" * 70)
    print("TEST 1: Piper speaks 'Hello, I am Hexa'")
    print("=" * 70)
    from hexa.tts import PiperTTS, TTSConfig

    t0 = time.perf_counter()
    piper = PiperTTS(TTSConfig(
        voice=cfg.tts.voice,
        voice_path=cfg.tts.voice_path,
        sample_rate=cfg.tts.sample_rate,
        fallback=cfg.tts.fallback,
    ))
    print(f"  Voice load: {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    wav_bytes = piper.synthesize("Hello, I am Hexa")
    t_syn = time.perf_counter() - t0

    # Inspect the WAV container
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        rate = wf.getframerate()
        nch = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.getnframes()
        header_size = 44
        print(f"  Synthesis time: {t_syn:.2f}s")
        print(f"  WAV: rate={rate}, ch={nch}, width={width}B, frames={frames} "
              f"({frames/rate:.2f}s), bytes={len(wav_bytes)}, header={header_size}")
        if len(wav_bytes) != header_size + frames * nch * width:
            print(f"  WARNING: byte length {len(wav_bytes)} != header+data "
                  f"{header_size + frames*nch*width} (header size may differ)")

    Path("gate2_piper.wav").write_bytes(wav_bytes)
    print("  Saved gate2_piper.wav")

    # Play through the SAME path TTSManager uses (this is where the bug shows)
    from hexa.audio import play_audio_bytes
    print("  Playing via TTSManager's play path (play_audio_bytes, np.frombuffer)...")
    t0 = time.perf_counter()
    play_audio_bytes(wav_bytes, cfg.tts.sample_rate, blocking=True)
    print(f"  Playback done ({time.perf_counter()-t0:.2f}s)")

    # Decode properly to measure peak/RMS for the record
    data, sr = sf.read("gate2_piper.wav", dtype="float32")
    print(f"  Decoded audio: sr={sr}, peak={np.max(np.abs(data)):.3f}, "
          f"rms={np.sqrt(np.mean(data**2)):.3f}")
    return wav_bytes


def test_ttsmanager_path():
    print("\n" + "=" * 70)
    print("TEST 2: TTSManager.speak (the path voice mode actually uses)")
    print("=" * 70)
    from hexa.tts import TTSManager, TTSConfig

    mgr = TTSManager(TTSConfig(
        voice=cfg.tts.voice,
        voice_path=cfg.tts.voice_path,
        sample_rate=cfg.tts.sample_rate,
        fallback=cfg.tts.fallback,
    ))
    t0 = time.perf_counter()
    handle = mgr.speak("Hello, I am Hexa, speaking through the manager path.")
    print(f"  speak() returned in {time.perf_counter()-t0:.2f}s")
    handle.wait()
    print("  Playback complete")


def test_pyttsx3_fallback():
    print("\n" + "=" * 70)
    print("TEST 3: pyttsx3 fallback")
    print("=" * 70)
    from hexa.tts import Pyttsx3TTS, TTSConfig
    from hexa.audio import play_audio_bytes

    tts = Pyttsx3TTS(TTSConfig(
        voice=cfg.tts.voice,
        voice_path=cfg.tts.voice_path,
        sample_rate=cfg.tts.sample_rate,
        fallback=cfg.tts.fallback,
    ))
    t0 = time.perf_counter()
    wav_bytes = tts.synthesize("This is the pyttsx3 fallback engine speaking.")
    t_syn = time.perf_counter() - t0
    print(f"  Synthesis time: {t_syn:.2f}s, bytes={len(wav_bytes)}")
    Path("gate2_pyttsx3.wav").write_bytes(wav_bytes)
    print("  Saved gate2_pyttsx3.wav")
    print("  Playing...")
    play_audio_bytes(wav_bytes, cfg.tts.sample_rate, blocking=True)
    print("  Playback done")

    # Check the pyttsx3 WAV's real sample rate vs what we pass
    try:
        data, sr = sf.read("gate2_pyttsx3.wav", dtype="float32")
        print(f"  pyttsx3 WAV real rate: {sr} Hz; played at {cfg.tts.sample_rate} Hz -> "
              f"{'OK' if sr == cfg.tts.sample_rate else 'WRONG SPEED (ratio %.2fx)' % (sr / cfg.tts.sample_rate)}")
    except Exception as e:
        print(f"  Could not decode pyttsx3 WAV: {e}")


def test_interrupt():
    print("\n" + "=" * 70)
    print("TEST 4: Interrupt mid-sentence")
    print("=" * 70)
    from hexa.tts import TTSManager, TTSConfig

    mgr = TTSManager(TTSConfig(
        voice=cfg.tts.voice,
        voice_path=cfg.tts.voice_path,
        sample_rate=cfg.tts.sample_rate,
        fallback=cfg.tts.fallback,
    ))
    long_text = ("This is a long sentence that should keep playing for several seconds. "
                 "If the interrupt works you will hear it stop right now, "
                 "and you will not hear this part.")
    handle = mgr.speak(long_text)
    time.sleep(2.5)  # let it speak a couple of seconds
    t0 = time.perf_counter()
    mgr.interrupt()
    print(f"  Interrupt called at {time.perf_counter()-t0:.3f}s after ~2.5s of speech")
    time.sleep(1.0)
    print(f"  is_speaking after interrupt: {mgr.is_speaking()}")
    assert not mgr.is_speaking(), "TTS still speaking after interrupt!"
    print("  OK: playback stopped")


def main():
    print("=" * 70)
    print("GATE 2: TTS DIAGNOSTIC")
    print("=" * 70)
    print_rates()
    test_piper()
    test_ttsmanager_path()
    test_pyttsx3_fallback()
    test_interrupt()
    print("\n" + "=" * 70)
    print("GATE 2 COMPLETE - please confirm what you heard")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Gate 2 failed")
        sys.exit(1)
