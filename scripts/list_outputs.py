import sounddevice as sd

for i, d in enumerate(sd.query_devices()):
    if d["max_output_channels"] > 0:
        marker = " <== DEFAULT OUTPUT" if i == sd.default.device[1] else ""
        print(f"  [{i}] {d['name']} out:{d['max_output_channels']} "
              f"sr:{d['default_samplerate']:.0f} hostapi:{d['hostapi']}{marker}")
