import sounddevice as sd

print("Host APIs:", [api["name"] for api in sd.query_hostapis()])
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        print(f"  [{i}] {d['name']} in:{d['max_input_channels']} sr:{d['default_samplerate']:.0f} hostapi:{d['hostapi']}")
