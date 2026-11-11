"""Gate 1 test: Hexa AudioStream with polling mode (real hardware, no mocks)."""
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\work - personal\Hexa Ai\hexa")
from hexa.config import load_config
from hexa.audio import get_audio_stream, shutdown_audio_stream

load_config()

stream = get_audio_stream()
q = stream.register_consumer(maxsize=200)
stream.start()

print("Recording... SPEAK NOW!")
time.sleep(4)
stream.stop()

chunks = []
while not q.empty():
    chunks.append(q.get_nowait())

if chunks:
    audio = np.concatenate(chunks)
    print(f"Chunks received: {len(chunks)}")
    print(f"Total samples: {len(audio)} ({len(audio)/16000:.2f}s)")
    print(f"Peak: {np.max(np.abs(audio))}")
    print(f"RMS: {np.sqrt(np.mean(audio.astype(np.float32)**2)):.1f}")

    # Save WAV for playback
    import wave
    with wave.open("gate1_hexa_stream.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio.tobytes())
    print("Saved gate1_hexa_stream.wav")

    # Play back
    import sounddevice as sd
    print("Playing back...")
    sd.play(audio.astype(np.float32) / 32767.0, 16000, blocking=True)
    print("Done")
else:
    print("FAILED: no chunks received!")
    sys.exit(1)

shutdown_audio_stream()
