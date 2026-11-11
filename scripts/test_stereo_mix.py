import sounddevice as sd
import numpy as np
import time

# Check device 18 (Stereo Mix)
devinfo = sd.query_devices(18)
print(f"Device 18: {devinfo['name']}, hostapi: {devinfo['hostapi']} - {sd.query_hostapis(devinfo['hostapi'])['name']}")

# Try callback with device 18, 2ch at 48000Hz
print("\nTrying callback with device 18 (2ch @ 48000Hz)...")
chunks = []
def cb(indata, frames, time_info, status):
    if status: print("Status:", status)
    if len(chunks) == 0:
        print("First chunk shape:", indata.shape)
    chunks.append(indata.copy())

try:
    stream = sd.InputStream(
        samplerate=48000,
        channels=2,
        blocksize=4800,
        dtype='int16',
        device=18,
        callback=cb,
    )
    stream.start()
    print("Active:", stream.active)
    print("Play some audio on the system (YouTube, etc.)...")
    time.sleep(5)
    stream.stop()
    stream.close()
    print("Chunks:", len(chunks))
    if chunks:
        audio = np.concatenate(chunks, axis=0)
        print("Shape:", audio.shape)
        print("Peak per ch:", np.max(np.abs(audio), axis=0))
        print("RMS per ch:", np.sqrt(np.mean(audio.astype(np.float32)**2, axis=0)))
except Exception as e:
    print("Error:", e)