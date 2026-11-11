import sounddevice as sd
import numpy as np
import time

# Check device 17
devinfo = sd.query_devices(17)
print(f"Device 17: {devinfo['name']}, hostapi: {devinfo['hostapi']} - {sd.query_hostapis(devinfo['hostapi'])['name']}")

# Try callback with device 17, 1ch at 44100Hz
print("\nTrying callback with device 17 (1ch @ 44100Hz)...")
chunks = []
def cb(indata, frames, time_info, status):
    if status: print("Status:", status)
    if len(chunks) == 0:
        print("First chunk shape:", indata.shape)
    chunks.append(indata.copy())

try:
    stream = sd.InputStream(
        samplerate=44100,
        channels=1,
        blocksize=4410,
        dtype='int16',
        device=17,
        callback=cb,
    )
    stream.start()
    print("Active:", stream.active)
    print("Recording... SPEAK NOW!")
    time.sleep(3)
    stream.stop()
    stream.close()
    print("Chunks:", len(chunks))
    if chunks:
        audio = np.concatenate(chunks, axis=0)
        print("Shape:", audio.shape)
        print("Peak:", np.max(np.abs(audio)))
        print("RMS:", np.sqrt(np.mean(audio.astype(np.float32)**2)))
except Exception as e:
    print("Error:", e)

# Also try sd.rec
print("\nTrying sd.rec with device 17 (1ch @ 44100Hz)...")
rec = sd.rec(int(3 * 44100), samplerate=44100, channels=1, dtype='int16', device=17)
print("Recording... SPEAK NOW!")
sd.wait()
print("Shape:", rec.shape)
print("Peak:", np.max(np.abs(rec)))
print("RMS:", np.sqrt(np.mean(rec.astype(np.float32)**2)))