#!/usr/bin/env python3
"""Fully automated REAL-STACK speech proof (no human needed).

Piper synthesizes the reference sentence, plays it through the speakers WHILE
Hexa's real AudioStream (persistent pull-mode InputStream @44.1kHz native ->
mono downmix -> scipy resample -> 1280-sample chunks) captures it through the
mic. Then the real WhisperTranscriber transcribes the captured chunks and WER
is scored. This exercises the exact voice-mode path end-to-end with real
speech; only the human mouth is replaced by the laptop speaker.
"""
import sys
import threading
import time

import numpy as np

sys.path.insert(0, r"D:\work - personal\Hexa Ai\hexa")
from hexa.config import load_config, get_config

load_config()

REFERENCE = "Hexa is a local voice assistant that runs completely offline"


def main() -> None:
    from hexa.audio import (get_audio_stream, shutdown_audio_stream,
                            play_audio_bytes, _decode_int16_bytes)
    from hexa.tts import PiperTTS, TTSConfig
    from hexa.stt import WhisperTranscriber, STTConfig

    cfg = get_config()

    # 1. Piper TTS -> wav bytes
    tts = PiperTTS(TTSConfig(
        voice=cfg.tts.voice,
        voice_path=cfg.tts.voice_path,
        sample_rate=cfg.tts.sample_rate,
        fallback=cfg.tts.fallback,
    ))
    wav_bytes = tts.synthesize(REFERENCE)
    data, rate = _decode_int16_bytes(wav_bytes)
    dur = len(data) / rate
    print(f"TTS: '{REFERENCE}' -> {dur:.2f}s @ {rate}Hz")

    # 2. Start mic capture via the real Hexa stack
    stream = get_audio_stream()
    q = stream.register_consumer(maxsize=1000)
    stream.start()
    print(f"Capture: device={stream._capture_device} native={stream._capture_native_rate}Hz "
          f"ch={stream._capture_channels} -> {cfg.audio.sample_rate}Hz mono")
    time.sleep(0.5)  # let the stream open and settle

    # 3. Play through speakers while the mic records (loopback through air)
    def play():
        try:
            play_audio_bytes(wav_bytes, rate, blocking=True)
        finally:
            pass

    player = threading.Thread(target=play, daemon=True)
    t0 = time.perf_counter()
    player.start()
    time.sleep(dur + 2.0)  # capture the utterance plus margins
    elapsed = time.perf_counter() - t0
    stream.stop()

    chunks = []
    while not q.empty():
        chunks.append(q.get_nowait())
    shutdown_audio_stream()

    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
    peak = int(np.max(np.abs(audio))) if len(audio) else 0
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2))) if len(audio) else 0.0
    print(f"Captured {len(chunks)} chunks, {len(audio)/cfg.audio.sample_rate:.2f}s "
          f"in {elapsed:.2f}s wall, peak={peak}, rms={rms:.1f}")
    if peak < 500:
        print("INCONCLUSIVE: captured level too low (speakers muted/quiet?)")
        sys.exit(2)

    # 4. Real Whisper transcribes the captured chunks
    transcriber = WhisperTranscriber(STTConfig(
        model=cfg.stt.model,
        device=cfg.stt.device,
        compute_type=cfg.stt.compute_type,
        language=cfg.stt.language,
        beam_size=cfg.stt.beam_size,
        vad_threshold=cfg.stt.vad_threshold,
    ))
    t0 = time.perf_counter()
    text = transcriber.transcribe([audio])
    print(f"Transcription: {time.perf_counter()-t0:.2f}s")
    print(f"TRANSCRIPT: '{text}'")

    # 5. WER
    def levenshtein(a, b):
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    ref = REFERENCE.lower().replace(",", "").replace(".", "").split()
    hyp = text.lower().replace(",", "").replace(".", "").split()
    dist = levenshtein(ref, hyp)
    w = 100.0 * dist / len(ref) if ref else 0.0
    print(f"WER vs reference: {w:.1f}% (edit distance {dist}/{len(ref)} words)")
    print("OK: real Hexa stack captured and transcribed real speech"
          if w <= 40 else "WEAK: WER > 40% - check speaker volume")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("gate3loop").exception("gate3 loopback failed")
        sys.exit(1)
