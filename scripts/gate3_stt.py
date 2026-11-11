#!/usr/bin/env python3
"""Gate 3: STT Diagnostic (real hardware, real models, no mocks).

Part A: record 5 s from the mic via Hexa's real AudioStream, transcribe with the
        real WhisperTranscriber, print transcript + timings.
Part B: roundtrip - Piper TTS a known sentence to WAV, transcribe it, print WER.
"""
from __future__ import annotations

import logging
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gate3")

REFERENCE = "Hexa is a local voice assistant that runs completely offline"

sys.path.insert(0, r"D:\work - personal\Hexa Ai\hexa")
from hexa.config import load_config, get_config

load_config()


def levenshtein(a: list, b: list) -> int:
    """Standard Levenshtein distance over token lists."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> tuple[float, int, int, int]:
    """Word Error Rate with simple substitution=1 scoring.
    Returns (wer%, substitutions+ins+del, ref words, hyp words)."""
    ref = reference.lower().replace(",", "").replace(".", "").split()
    hyp = hypothesis.lower().replace(",", "").replace(".", "").split()
    dist = levenshtein(ref, hyp)
    n = len(ref)
    return (dist / n * 100.0) if n else 0.0, dist, n, len(hyp)


def record_mic(duration: float = 5.0) -> tuple[np.ndarray, int]:
    """Record via Hexa's real AudioStream (polling) - the exact voice-mode stack."""
    from hexa.audio import get_audio_stream, shutdown_audio_stream

    stream = get_audio_stream()
    q = stream.register_consumer(maxsize=1000)
    stream.start()
    print(f"  Capture: device={stream._capture_device} native={stream._capture_native_rate}Hz "
          f"ch={stream._capture_channels} -> {get_config().audio.sample_rate}Hz mono")
    print(f"  Recording {duration:.0f}s... start speaking ~1s from now:")
    print(f"  '{REFERENCE}'")
    sd.sleep(int(duration * 1000))
    stream.stop()

    chunks = []
    while not q.empty():
        chunks.append(q.get_nowait())
    shutdown_audio_stream()

    audio = np.concatenate(chunks)
    sr = get_config().audio.sample_rate
    peak = int(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    print(f"  Captured {len(chunks)} chunks, {len(audio)/sr:.2f}s, peak={peak}, rms={rms:.1f}")
    return audio, sr


def save_wav(audio: np.ndarray, sr: int, path: str) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    print(f"  Saved {path}")


def part_a():
    print("=" * 70)
    print("PART A: MIC -> WHISPER (5 s recording)")
    print("=" * 70)
    from hexa.stt import WhisperTranscriber, STTConfig

    cfg = get_config()
    t0 = time.perf_counter()
    transcriber = WhisperTranscriber(STTConfig(
        model=cfg.stt.model,
        device=cfg.stt.device,
        compute_type=cfg.stt.compute_type,
        language=cfg.stt.language,
        beam_size=cfg.stt.beam_size,
        vad_threshold=cfg.stt.vad_threshold,
    ))
    print(f"  Model load: {time.perf_counter()-t0:.2f}s")

    audio, sr = record_mic(6.0)
    save_wav(audio, sr, "gate3_mic.wav")

    chunks = [audio]  # transcriber takes list of int16 chunks
    t0 = time.perf_counter()
    text = transcriber.transcribe(chunks)
    t_trans = time.perf_counter() - t0

    print(f"  Transcription time: {t_trans:.2f}s (for {len(audio)/sr:.2f}s audio)")
    print(f"  TRANSCRIPT: '{text}'")
    w, dist, nref, nhyp = wer(REFERENCE, text)
    print(f"  MIC WER vs reference: {w:.1f}% (edit distance {dist}/{nref} words)")
    return text


def part_b():
    print("\n" + "=" * 70)
    print("PART B: ROUNDTRIP Piper TTS -> WAV -> Whisper STT (WER)")
    print("=" * 70)
    from hexa.stt import WhisperTranscriber, STTConfig
    from hexa.tts import PiperTTS, TTSConfig
    from hexa.audio import _decode_int16_bytes

    cfg = get_config()
    reference = "The quick brown fox jumps over the lazy dog"

    tts = PiperTTS(TTSConfig(
        voice=cfg.tts.voice,
        voice_path=cfg.tts.voice_path,
        sample_rate=cfg.tts.sample_rate,
        fallback=cfg.tts.fallback,
    ))
    t0 = time.perf_counter()
    wav_bytes = tts.synthesize(reference)
    print(f"  TTS synthesis: {time.perf_counter()-t0:.2f}s")
    Path("gate3_tts_roundtrip.wav").write_bytes(wav_bytes)
    print("  Saved gate3_tts_roundtrip.wav")

    data, rate = _decode_int16_bytes(wav_bytes)
    pcm16 = (data * 32767).astype(np.int16)
    print(f"  TTS audio: {len(pcm16)/rate:.2f}s @ {rate}Hz")

    transcriber = WhisperTranscriber(STTConfig(
        model=cfg.stt.model,
        device=cfg.stt.device,
        compute_type=cfg.stt.compute_type,
        language=cfg.stt.language,
        beam_size=cfg.stt.beam_size,
        vad_threshold=cfg.stt.vad_threshold,
    ))

    t0 = time.perf_counter()
    text = transcriber.transcribe([pcm16])
    t_trans = time.perf_counter() - t0

    print(f"  STT time: {t_trans:.2f}s")
    print(f"  REFERENCE: '{reference}'")
    print(f"  TRANSCRIPT: '{text}'")
    w, dist, nref, nhyp = wer(reference, text)
    print(f"  WER: {w:.1f}%  (edit distance {dist} over {nref} ref words, {nhyp} hyp words)")
    if w <= 15:
        print("  OK: WER <= 15% (clean audio roundtrip)")
    else:
        print("  FAIL: WER > 15% - investigate")


def main():
    print("=" * 70)
    print("GATE 3: STT DIAGNOSTIC")
    print("=" * 70)
    part_a()
    part_b()
    print("\n" + "=" * 70)
    print("GATE 3 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Gate 3 failed")
        sys.exit(1)
