#!/usr/bin/env python3
"""Format-mismatch hunt: which dtype actually decodes the Intel SST array pins?

For each candidate device (WDM-KS 4ch@16k, MME 4ch@44.1k), record 3s with
dtype in (int16, int32, float32) and check whether the signal is real speech:
- DC offset (real mics: ~0)
- speech-band energy fraction EXCLUDING DC (real speech: >0.9)
- crest factor (real speech: 3-12)
"""
import sys

import numpy as np
import sounddevice as sd

sys.path.insert(0, r"D:\work - personal\Hexa Ai\hexa")

SECONDS = 3.0
SR = 16000


def analyze(name, data, sr):
    a = np.asarray(data)
    if a.dtype == np.float32:
        a16 = a  # already float
        f = a16.flatten()
    elif a.dtype == np.int32:
        f = (a.flatten().astype(np.float64) / 2**31)
    else:  # int16
        f = a.flatten().astype(np.float64) / 32768.0
    dc = float(np.mean(f))
    ac = f - dc
    rms_ac = float(np.sqrt(np.mean(ac ** 2)))
    rms_total = float(np.sqrt(np.mean(f ** 2)))
    crest = float(np.max(np.abs(f)) / rms_ac) if rms_ac > 0 else float("inf")
    # spectrum
    spec = np.abs(np.fft.rfft(ac)) ** 2
    freqs = np.fft.rfftfreq(len(ac), 1 / sr)
    speech = (freqs >= 85) & (freqs <= 4000)
    total = float(np.sum(spec))
    frac = float(np.sum(spec[speech]) / total) if total > 0 else 0.0
    print(f"  {name:28s} dtype={str(np.asarray(data).dtype):8s} "
          f"DC={dc:+.4f} rms_ac={rms_ac:.5f} crest={crest:6.1f} speechband={frac:.3f}")
    return rms_ac, crest, frac, dc


def main():
    print("Candidates:")
    inputs = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] >= 4:
            inputs.append((i, d))
            print(f"  [{i}] {d['name']} ch={d['max_input_channels']} "
                  f"native_sr={d['default_samplerate']:.0f} hostapi={d['hostapi']}")

    print("\n=== WDM-KS pin @ native 16000, 4ch ===")
    for i, d in inputs:
        if int(d["default_samplerate"]) != 16000:
            continue
        for dt in ("int16", "int32", "float32"):
            try:
                rec = sd.rec(int(SECONDS * 16000), samplerate=16000,
                             channels=4, dtype=dt, device=i)
                sd.wait()
                analyze(f"dev{i} {d['name'][:20]}", rec, 16000)
                if dt == "float32":
                    np.save(f"gate3_fmt_dev{i}_f32.npy", rec)
                elif dt == "int32":
                    np.save(f"gate3_fmt_dev{i}_i32.npy", rec)
                else:
                    np.save(f"gate3_fmt_dev{i}_i16.npy", rec)
            except Exception as e:
                print(f"  dev{i} {dt}: FAILED {e}")

    print("\n=== MME pin @ native 44100, 4ch ===")
    for i, d in inputs:
        if int(d["default_samplerate"]) != 44100:
            continue
        for dt in ("int16", "int32", "float32"):
            try:
                rec = sd.rec(int(SECONDS * 44100), samplerate=44100,
                             channels=4, dtype=dt, device=i)
                sd.wait()
                analyze(f"dev{i} {d['name'][:20]}", rec, 44100)
                np.save(f"gate3_fmt_mme_dev{i}_{dt}.npy", rec)
            except Exception as e:
                print(f"  dev{i} {dt}: FAILED {e}")


if __name__ == "__main__":
    main()
