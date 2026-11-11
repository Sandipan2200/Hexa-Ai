#!/usr/bin/env python3
"""Is clipping what breaks Whisper transcription?"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, r"D:\work - personal\Hexa Ai\hexa")
from hexa.config import load_config, get_config
from hexa.stt import WhisperTranscriber, STTConfig
from hexa.audio import _decode_int16_bytes

load_config()
cfg = get_config()

transcriber = WhisperTranscriber(STTConfig(
    model=cfg.stt.model, device=cfg.stt.device,
    compute_type=cfg.stt.compute_type, language=cfg.stt.language,
    beam_size=cfg.stt.beam_size, vad_threshold=cfg.stt.vad_threshold,
))

# 1. The actual clipped mic recording from the failed Part A
data, rate = sf.read("gate3_mic.wav", dtype="int16")
print(f"gate3_mic.wav: {len(data)/rate:.2f}s rms={np.sqrt(np.mean(data.astype(np.float32)**2)):.1f}")
print(f"  -> '{transcriber.transcribe([data])}'")

# 2. Synthetic hard clip of clean TTS audio
clean, rate = _decode_int16_bytes(Path("gate3_tts_roundtrip.wav").read_bytes())
pcm = clean * 32767.0
for gain in (3.0, 6.0, 12.0):
    hot = np.clip(pcm * gain, -32767, 32767).astype(np.int16)
    rms = float(np.sqrt(np.mean(hot.astype(np.float32) ** 2)))
    clipped_pct = 100.0 * np.mean(np.abs(hot) >= 32000)
    text = transcriber.transcribe([hot])
    print(f"synthetic clip x{gain}: rms={rms:.0f} clipped={clipped_pct:.1f}% -> '{text}'")
