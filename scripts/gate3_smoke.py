"""Smoke test (no human speech needed): new polling stack resolves the MME pin,
captures at native 44.1kHz, resamples to 16kHz, and chunks flow in real time."""
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\work - personal\Hexa Ai\hexa")
from hexa.config import load_config

load_config()

from hexa.audio import get_audio_stream, shutdown_audio_stream

stream = get_audio_stream()
q = stream.register_consumer(maxsize=1000)
stream.start()
print(f"resolved: device={stream._capture_device} "
      f"native_rate={stream._capture_native_rate} ch={stream._capture_channels}")

time.sleep(10.0)
stream.stop()

chunks = []
while not q.empty():
    chunks.append(q.get_nowait())
shutdown_audio_stream()

audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
peak = int(np.max(np.abs(audio))) if len(audio) else 0
rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2))) if len(audio) else 0.0
print(f"chunks={len(chunks)} total={len(audio)/16000:.2f}s peak={peak} rms={rms:.1f}")
# 10s wall time should yield ~10s of audio minus stream-open overhead (steady state = no loss)
rate_ok = (len(audio) / 16000) / 10.0 >= 0.9
print("real-time pacing:", "OK" if rate_ok else "CHECK")
