import sounddevice as sd
import numpy as np
import time

# Try callback with device 1, 1ch at 16000Hz
print("Trying callback with device 1 (1ch @ 16000Hz)...")
chunks = []
def cb(indata, frames, time_info, status):
    if status: print("Status:", status)
    if len(chunks) == 0:
        print("First chunk shape:", indata.shape)
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