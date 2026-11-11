#!/usr/bin/env python3
"""Deep debug: why does Whisper return '' on real mic audio that has good level/shape?"""
import sys

import numpy as np

sys.path.insert(0, r"D:\work - personal\Hexa Ai\hexa")
from hexa.config import load_config, get_config

load_config()
cfg = get_config()

raw = np.load("gate3_raw4ch.npy")
print(f"raw shape: {raw.shape}, dtype={raw.dtype}")
a = raw[:, 1].astype(np.float32)  # ch1
print(f"ch1: min={a.min():.0f} max={a.max():.0f} mean(DC)={a.mean():.1f} "
      f"rms={np.sqrt(np.mean(a**2)):.1f}")

# Spectrum: where is the energy?
spec = np.abs(np.fft.rfft(a))
freqs = np.fft.rfftfreq(len(a), 1/16000)
top = np.argsort(spec)[-8:][::-1]
print("top spectral peaks (Hz, magnitude):")
for f in top:
    print(f"  {freqs[f]:8.1f} Hz  {spec[f]:.3e}")

# Speech-band energy fraction
band = (freqs >= 85) & (freqs <= 3000)
print(f"speech-band (85-3000Hz) energy fraction: "
      f"{float(np.sum(spec[band]**2)/np.sum(spec**2)):.3f}")

# What does faster-whisper's own VAD see?
from faster_whisper import WhisperModel
model = WhisperModel(cfg.stt.model, device=cfg.stt.device,
                     compute_type=cfg.stt.compute_type, local_files_only=True)
audio32 = (a / 32768.0).astype(np.float32)

print("\n--- vad_filter=False (raw transcription) ---")
segments, info = model.transcribe(audio32, language="en", beam_size=5, vad_filter=False)
texts = [seg.text for seg in segments]
print(f"segments={len(texts)}, text: '{' '.join(texts).strip()}'")

print("\n--- vad_filter=True with default params ---")
segments, info = model.transcribe(audio32, language="en", beam_size=5, vad_filter=True)
texts = [seg.text for seg in segments]
print(f"segments={len(texts)}, text: '{' '.join(texts).strip()}'")

print("\n--- vad_filter=True, threshold lowered ---")
segments, info = model.transcribe(
    audio32, language="en", beam_size=5, vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500, threshold=0.2))
texts = [seg.text for seg in segments]
print(f"segments={len(texts)}, text: '{' '.join(texts).strip()}'")

# Silero VAD probs directly (the same VAD engine faster-whisper bundles)
print("\n--- Silero VAD speech probability over time (512-sample hops) ---")
try:
    from faster_whisper.vad import get_speech_timestamps
    from faster_whisper.audio import decode_audio
    ts = get_speech_timestamps(audio32, vad_threshold=0.5, sampling_rate=16000)
    print(f"speech timestamps: {ts[:5]}{'...' if len(ts) > 5 else ''} (total {len(ts)})")
except Exception as e:
    print(f"get_speech_timestamps failed: {e}")

import torch
from silero_vad import load_silero_vad
vad = load_silero_vad()
probs = []
with torch.no_grad():
    for i in range(0, len(audio32) - 512, 512):
        t = torch.from_numpy(audio32[i:i+512])
        probs.append(float(vad(t, 16000).item()))
probs = np.array(probs)
print(f"silero probs: min={probs.min():.3f} max={probs.max():.3f} mean={probs.mean():.3f} "
      f"frac>0.5={np.mean(probs > 0.5):.2f}")
print("first 20 probs:", np.round(probs[:20], 2).tolist())
