"""Test whether the VENV python gets real mic audio (permission check)."""
import sys

import numpy as np

sys.path.insert(0, r"D:\work - personal\Hexa Ai\hexa")
from hexa.config import load_config
from hexa.audio import get_audio_stream, shutdown_audio_stream
import sounddevice as sd

print("Python executable:", sys.executable)

load_config()
stream = get_audio_stream()
q = stream.register_consumer(maxsize=500)
stream.start()

print("Recording 3s... SPEAK NOW!")
sd.sleep(3000)
stream.stop()

chunks = []
while not q.empty():
    chunks.append(q.get_nowait())
shutdown_audio_stream()

if chunks:
    audio = np.concatenate(chunks)
    peak = int(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    print(f"Chunks: {len(chunks)}  Peak: {peak}  RMS: {rms:.1f}")
    if peak > 1000:
        print("OK: venv python receives real mic audio")
    else:
        print("FAIL: near-silence (mic permission blocked for this executable)")
        sys.exit(1)
else:
    print("FAIL: no chunks")
    sys.exit(1)
