import sounddevice as sd
import numpy as np
import time

# Check device 1
devinfo = sd.query_devices(1)
print(f"Device 1: {devinfo['name']}, hostapi: {devinfo['hostapi']} - {sd.query_hostapis(devinfo['hostapi'])['name']}")

# Try callback with device 1, 4ch at 44100Hz
print("\nTrying callback with device 1 (4ch @ 44100Hz)...")
chunks = []
def cb(indata, frames, time_info, status):
    if status: print("Status:", status)
    if len(chunks) == 0:
        print("First chunk shape:", indata.shape)
    chunks.append(indata.copy())

try:
    stream = sd.InputStream(
        samplerate=44100,
        channels=4,
        blocksize=4410,  # 100ms
        dtype='int16',
        device=1,
        callback=cb,
    )
    stream.start()
    print("Active:", stream.active)
    time.sleep(3)
    stream.stop()
    stream.close()
    print("Chunks:", len(chunks))
    if chunks:
        audio = np.concatenate(chunks, axis=0)
        print("Shape:", audio.shape)
        print("Peak per ch:", np.max(np.abs(audio), axis=0))
except Exception as e:
    print("Error:", e)