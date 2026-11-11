#!/usr/bin/env python3
"""Gate 1: Test Intel array with 4 channels and Stereo Mix"""
from __future__ import annotations

import logging
import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, r"D:\work - personal\Hexa Ai\hexa")


def record_test(duration=3.0, sample_rate=16000, channels=1, device=None):
    print(f"\nRecording {duration}s from device {device} at {sample_rate}Hz, channels={channels}...")
    
    chunk_size = 1280
    frames = int(duration * sample_rate)
    recording = np.zeros(frames, dtype=np.int16)
    
    def callback(indata, frames, time_info, status):
        if status:
            logger.warning("Recording status: %s", status)
        if indata.dtype != np.int16:
            chunk = (indata * 32767).astype(np.int16)
        else:
            chunk = indata.copy()
        chunk = chunk.flatten()
        
        nonlocal write_pos
        end_pos = write_pos + len(chunk)
        if end_pos <= frames:
            recording[write_pos:end_pos] = chunk[:frames - write_pos]
        write_pos = end_pos
    
    write_pos = 0
    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            blocksize=chunk_size,
            dtype='int16',
            device=device,
            callback=callback
        ):
            print("  Recording... SPEAK LOUDLY NOW!")
            sd.sleep(int(duration * 1000))
    except Exception as e:
        logger.exception("Recording failed")
        raise
    
    recording = recording[:write_pos]
    
    if len(recording) > 0:
        peak = np.max(np.abs(recording))
        rms = float(np.sqrt(np.mean(recording.astype(np.float32) ** 2)))
        peak_db = 20 * np.log10(peak / 32767.0) if peak > 0 else -float('inf')
        rms_db = 20 * np.log10(rms / 32767.0) if rms > 0 else -float('inf')
        
        print(f"  Samples: {len(recording)} ({len(recording)/sample_rate:.2f}s)")
        print(f"  Peak:  {peak:6d} ({peak_db:6.1f} dBFS)")
        print(f"  RMS:   {rms:6.1f} ({rms_db:6.1f} dBFS)")
        
        if peak < 100:
            print("  WARNING: Very low signal!")
        elif peak < 1000:
            print("  WARNING: Low signal")
        else:
            print("  OK: Good signal level")
    else:
        print("  ERROR: No audio recorded!")
        peak = rms = 0
    
    return recording, sample_rate


def play_recording(recording, sample_rate, device=None):
    print(f"\nPlaying back ({len(recording)/sample_rate:.2f}s)...")
    audio_float = recording.astype(np.float32) / 32767.0
    sd.play(audio_float, sample_rate, device=device, blocking=True)
    print("  Done")


def save_wav(recording, sample_rate, path):
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(recording.tobytes())
    print(f"  Saved to {path}")


def test_device(device_id, sample_rate, channels=1, name=""):
    print(f"\n{'='*70}")
    print(f"TESTING DEVICE {device_id}: {name} ({channels}ch @ {sample_rate}Hz)")
    print(f"{'='*70}")
    try:
        recording, sr = record_test(
            duration=3.0,
            sample_rate=sample_rate,
            channels=channels,
            device=device_id
        )
        save_wav(recording, sr, Path(f"gate1_recording_dev{device_id}_{channels}ch.wav"))
        play_recording(recording, sr)
        return recording
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def main():
    print("=" * 70)
    print("GATE 1: TESTING INTEL ARRAY 4CH AND STEREO MIX")
    print("=" * 70)
    
    # Test device 1 (Intel array) with 4 channels at native 44100Hz
    print("\nTrying device 1 with 4 channels at native 44100Hz...")
    test_device(1, 44100, channels=4, name="Dev 1 Intel Array 4ch")
    
    # Test device 1 (Intel array) with 4 channels at 16000Hz
    print("\nTrying device 1 with 4 channels at 16000Hz...")
    test_device(1, 16000, channels=4, name="Dev 1 Intel Array 4ch 16kHz")
    
    # Test device 5 (another Intel array) with 4 channels at native 44100Hz
    print("\nTrying device 5 with 4 channels at native 44100Hz...")
    test_device(5, 44100, channels=4, name="Dev 5 Intel Array 4ch")
    
    # Test device 18 (Stereo Mix) - captures system audio
    print("\nTrying device 18 (Stereo Mix) at native 48000Hz...")
    test_device(18, 48000, channels=2, name="Dev 18 Stereo Mix")
    
    # Test device 18 at 16000Hz
    print("\nTrying device 18 (Stereo Mix) at 16000Hz...")
    test_device(18, 16000, channels=2, name="Dev 18 Stereo Mix 16kHz")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Gate 1 failed")
        sys.exit(1)