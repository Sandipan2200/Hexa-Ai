import sounddevice as sd
import numpy as np
import time

print("Host APIs:")
for i, api in enumerate(sd.query_hostapis()):
    print(f"  {i}: {api['name']}")

devinfo = sd.query_devices(21)
print(f"Device 21 hostapi: {devinfo['hostapi']} - {sd.query_hostapis(devinfo['hostapi'])['name']}")

# Try with hostapi=0 (MME)
print("\nTrying with MME (hostapi=0)...")
chunks = []
def cb(indata, frames, time_info, status):
    if status: print("Status:", status)
    print("Got", frames, "frames")
    chunks.append(indata.copy())

try:
    stream = sd.InputStream(
        samplerate=16000,
        channels=4,
        blocksize=1280,
        dtype='int16',
        device=21,
        callback=cb,
        hostapi=0,
    )
    stream.start()
    print("Active:", stream.active)
    time.sleep(3)
    stream.stop()
    stream.close()
    print("Chunks:", len(chunks))
except Exception as e:
    print("Error:", e)

# Try with hostapi=2 (Windows DirectSound)
print("\nTrying with DirectSound (hostapi=2)...")
chunks = []
try:
    stream = sd.InputStream(
        samplerate=16000,
        channels=4,
        blocksize=1280,
        dtype='int16',
        device=21,
        callback=cb,
        hostapi=2,
    )
    stream.start()
    print("Active:", stream.active)
    time.sleep(3)
    stream.stop()
    stream.close()
    print("Chunks:", len(chunks))
except Exception as e:
    print("Error:", e)