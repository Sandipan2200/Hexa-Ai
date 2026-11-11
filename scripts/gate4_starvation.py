"""Pinpoint the chunk starvation: is the poll thread dead, or is the recognizer stuck?

Monitors every second: VAD event count, queue depth, recognizer thread alive,
stream running. Loads Whisper + Silero exactly like SpeechRecognizer does.
"""
import sys
import threading
import time

import numpy as np

sys.path.insert(0, r"D:\work - personal\Hexa Ai\hexa")
from hexa.config import load_config, get_config

load_config()

from hexa.audio import get_audio_stream
from hexa.stt import WhisperTranscriber, STTConfig, SileroVAD

cfg = get_config()

stream = get_audio_stream()
q = stream.register_consumer(maxsize=100)
stream.start()

# Load the same models SpeechRecognizer loads (Whisper + Silero)
print("Loading Whisper...")
t0 = time.perf_counter()
WhisperTranscriber(STTConfig(
    model=cfg.stt.model, device=cfg.stt.device, compute_type=cfg.stt.compute_type,
    language=cfg.stt.language, beam_size=cfg.stt.beam_size,
    vad_threshold=cfg.stt.vad_threshold,
))
print(f"Whisper loaded in {time.perf_counter()-t0:.1f}s")

vad = SileroVAD(0.5)
vad_calls = []

def vad_worker():
    for i in range(200):
        chunk = np.zeros(1280, dtype=np.int16)
        t0 = time.perf_counter()
        vad.get_speech_prob(chunk)
        vad_calls.append(time.perf_counter() - t0)
        time.sleep(0.08)

t = threading.Thread(target=vad_worker, daemon=True)
t.start()

for i in range(9):
    time.sleep(1.0)
    drained = 0
    while not q.empty():
        q.get_nowait()
        drained += 1
    print(f"t={i+1}s: chunks_last_1s={drained} stream_running={stream.is_running()} "
          f"vad_calls={len(vad_calls)} "
          f"vad_ms_med={np.median(vad_calls)*1000:.1f}" if vad_calls else f"t={i+1}s: no vad calls yet")

stream.stop()
