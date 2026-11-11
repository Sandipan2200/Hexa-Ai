#!/usr/bin/env python3
"""Controlled test: does Whisper+its VAD fail on audio at the mic's observed level?"""
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

data, rate = _decode_int16_bytes(Path("gate3_tts_roundtrip.wav").read_bytes())
pcm16 = (data * 32767).astype(np.int16)
base_rms = float(np.sqrt(np.mean(pcm16.astype(np.float32) ** 2)))
print(f"Clean TTS audio: {len(pcm16)/rate:.2f}s, rms={base_rms:.1f}")

transcriber = WhisperTranscriber(STTConfig(
    model=cfg.stt.model, device=cfg.stt.device,
    compute_type=cfg.stt.compute_type, language=cfg.stt.language,
    beam_size=cfg.stt.beam_size, vad_threshold=cfg.stt.vad_threshold,
))

# Mic observations: ch0 raw ~113 rms (last run), ~307 (gate1), clipped 18929 (loud)
# Target ASR-friendly rms: ~1000-3000 (-30 to -21 dBFS)
for gain in (1.0, 0.25, 0.1, 0.05):
    a = (pcm16.astype(np.float32) * gain).astype(np.int16)
    rms = float(np.sqrt(np.mean(a.astype(np.float32) ** 2)))
    text = transcriber.transcribe([a])
    print(f"gain={gain:5} -> rms={rms:8.1f} ({20*np.log10(rms/32767):6.1f} dBFS): '{text}'")
