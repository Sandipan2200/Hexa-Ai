"""Measure Silero VAD per-call latency on 1280-sample chunks (the real input size)."""
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\work - personal\Hexa Ai\hexa")
from hexa.stt import SileroVAD

vad = SileroVAD(threshold=0.5)
rng = np.random.default_rng(0)

# Warm-up
for _ in range(5):
    vad.get_speech_prob(np.zeros(1280, dtype=np.int16))

# Timed calls with realistic noise chunks
times = []
for i in range(20):
    chunk = (rng.normal(0, 300, 1280)).astype(np.int16)
    t0 = time.perf_counter()
    vad.get_speech_prob(chunk)
    times.append((time.perf_counter() - t0) * 1000)

print(f"per-call ms: min={min(times):.1f} median={sorted(times)[10]:.1f} max={max(times):.1f}")
print(f"budget: 80ms per chunk -> {'OK (keeps up)' if sorted(times)[10] < 60 else 'TOO SLOW'}")

# Also test the 512-sample path cost
times512 = []
for i in range(20):
    chunk = (rng.normal(0, 300, 512)).astype(np.int16)
    t0 = time.perf_counter()
    vad.get_speech_prob(chunk)
    times512.append((time.perf_counter() - t0) * 1000)
print(f"512-sample per-call ms: median={sorted(times512)[10]:.1f}")
