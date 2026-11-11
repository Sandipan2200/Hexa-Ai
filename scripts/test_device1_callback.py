import sounddevice as sd
import numpy as np
import time

# Check device 1
devinfo = sd.query_devices(1)
print(f"Device 1: {devinfo['name']}, hostapi: {devinfo['hostapi']} - {sd.query_hostapis(devinfo['hostapi'])['name']}")

# Try callback with device 1 (default)
print("\nTrying callback with device 1...")
chunks = []
def cb(indata, frames, time_info, status):
    if status: print("Status:", status)
    print("Got", frames, "frames, shape:", indata.shape)
    chunks.append(indata.copy())

try:
    stream = sd.InputStream(
        samplerate=16000,
        channels=1,
        blocksize=1280,
        dtype='int16',
        device=1,
        callback=cb,
    )
    stream.start()
    print("Active:", stream.active)
    time.sleep(2)
    stream.stop()
    stream.close()
    print("Chunks:", len(chunks))
    if chunks:
        audio = np.concatenate(chunks, axis=0)
        print("Shape:", audio.shape)
        print("Peak:", np.max(np.abs(audio)))
except Exception as e:
    print("Error:", e)