#!/usr/bin/env python3
"""Decisive test: MME array pin @44100 -> resample 16k -> Whisper -> transcript.

Speak the target sentence when prompted. This validates the fix path before
touching hexa/audio.py.
"""
import sys
import time

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

sys.path.insert(0, r"D:\work - personal\Hexa Ai\hexa")
from hexa.config import load_config, get_config

load_config()
cfg = get_config()

REFERENCE = "Hexa is a local voice assistant that runs completely offline"


def find_mme_array():
    for i, d in enumerate(sd.query_devices()):
        if (d["max_input_channels"] >= 4
                and int(d["default_samplerate"]) == 44100
                and d["hostapi"] == 0
                and "array" in d["name"].lower()):
            return i, d
    return None, None


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    i, d = find_mme_array()
    if i is None:
        print("MME array device not found!")
        sys.exit(1)
    print(f"Device {i}: {d['name']} (MME, 4ch @44100)")

    print("\nRecording 5s... READ ALOUD NOW:")
    print(f"  '{REFERENCE}'")
    rec = sd.rec(int(5 * 44100), samplerate=44100, channels=4, dtype="int16", device=i)
    sd.wait()
    print(f"Captured {rec.shape}")

    for ch in range(4):
        a = rec[:, ch].astype(np.float32)
        rms = float(np.sqrt(np.mean(a ** 2)))
        print(f"  ch{ch}: rms={rms:.0f}")

    # Resample each channel to 16k
    n16 = int(len(rec) * 16000 / 44100)
    ch16 = np.stack([
        resample_poly(rec[:, c].astype(np.float64), 16000, 44100)[:n16]
        for c in range(4)
    ], axis=1)

    from hexa.stt import WhisperTranscriber, STTConfig
    transcriber = WhisperTranscriber(STTConfig(
        model=cfg.stt.model, device=cfg.stt.device,
        compute_type=cfg.stt.compute_type, language=cfg.stt.language,
        beam_size=cfg.stt.beam_size, vad_threshold=cfg.stt.vad_threshold,
    ))

    ref_words = REFERENCE.lower().replace(",", "").split()
    print("\nTranscription per channel (resampled to 16k):")
    for ch in range(4):
        a = ch16[:, ch].astype(np.int16)
        t0 = time.perf_counter()
        text = transcriber.transcribe([a])
        hyp = text.lower().replace(",", "").split()
        dist = levenshtein(ref_words, hyp)
        wer = 100.0 * dist / len(ref_words) if ref_words else 0.0
        print(f"  ch{ch}: ({time.perf_counter()-t0:.1f}s, WER={wer:.0f}%) '{text}'")

    # mix
    mix = ch16[:, :3].astype(np.float32).mean(axis=1)
    text = transcriber.transcribe([mix.astype(np.int16)])
    hyp = text.lower().replace(",", "").split()
    wer = 100.0 * levenshtein(ref_words, hyp) / len(ref_words)
    print(f"  mix(ch0-2): WER={wer:.0f}% '{text}'")


if __name__ == "__main__":
    main()
