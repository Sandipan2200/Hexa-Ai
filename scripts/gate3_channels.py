#!/usr/bin/env python3
"""Gate 3 follow-up: which channel of the 4-ch Intel SST array carries usable speech?

Records 5s (all 4 channels, raw), prints per-channel peak/RMS, saves per-channel
WAVs, and transcribes each channel (and a x0.25 attenuated ch0) with and without
the internal VAD filter.
"""
from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, r"D:\work - personal\Hexa Ai\hexa")
from hexa.config import load_config, get_config
from hexa.stt import WhisperTranscriber, STTConfig

load_config()
cfg = get_config()


def resolve_device():
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] >= 2 and int(d["default_samplerate"]) == 16000:
            return i, d
    return None, None


def save(path, audio, sr):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())


def main():
    dev_idx, dev = resolve_device()
    if dev_idx is None:
        print("No 16kHz input device found!")
        sys.exit(1)
    print(f"Device {dev_idx}: {dev['name']} ({dev['max_input_channels']}ch native)")

    print("\nRecording 5s... READ ALOUD NOW:")
    print("  'Hexa is a local voice assistant that runs completely offline'")
    raw = sd.rec(int(5 * 16000), samplerate=16000,
                 channels=dev["max_input_channels"], dtype="int16", device=dev_idx)
    sd.wait()
    print(f"Shape: {raw.shape}")

    # Per-channel stats + crest factor
    for ch in range(raw.shape[1]):
        a = raw[:, ch]
        peak = int(np.max(np.abs(a)))
        rms = float(np.sqrt(np.mean(a.astype(np.float32) ** 2)))
        clipped = int(np.sum(np.abs(a) >= 32700))
        crest = (peak / rms) if rms > 0 else float("inf")
        # speech has crest factor ~3-12; ~1.7 = hard clipped; >50 = spikes/silence
        print(f"  ch{ch}: peak={peak:6d} rms={rms:9.1f} crest={crest:6.1f} "
              f"clipped_samples={clipped}")
        save(f"gate3_ch{ch}.wav", a, 16000)
    np.save("gate3_raw4ch.npy", raw)

    transcriber = WhisperTranscriber(STTConfig(
        model=cfg.stt.model, device=cfg.stt.device,
        compute_type=cfg.stt.compute_type, language=cfg.stt.language,
        beam_size=cfg.stt.beam_size, vad_threshold=cfg.stt.vad_threshold,
    ))

    print("\nPer-channel transcription (raw):")
    for ch in range(raw.shape[1]):
        a = raw[:, ch]
        text = transcriber.transcribe([a])
        print(f"  ch{ch}: '{text}'")

    print("\nPer-channel transcription (normalized to rms=2000):")
    for ch in range(raw.shape[1]):
        a = raw[:, ch].astype(np.float32)
        rms = float(np.sqrt(np.mean(a ** 2)))
        if rms < 1:
            print(f"  ch{ch}: silent")
            continue
        scaled = np.clip(a * (2000.0 / rms), -32767, 32767).astype(np.int16)
        text = transcriber.transcribe([scaled])
        print(f"  ch{ch}: '{text}'")

    # Down-mix average of ch0+ch1+ch2 (ch3 dead), normalized
    mix = raw[:, :3].astype(np.float32).mean(axis=1)
    rms = float(np.sqrt(np.mean(mix ** 2)))
    if rms >= 1:
        scaled = np.clip(mix * (2000.0 / rms), -32767, 32767).astype(np.int16)
        text = transcriber.transcribe([scaled])
        print(f"  mix(ch0+1+2, norm): '{text}'")

    print("\nDone. Saved gate3_ch0..3.wav + gate3_raw4ch.npy")


if __name__ == "__main__":
    main()
