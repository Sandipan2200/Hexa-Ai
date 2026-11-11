#!/usr/bin/env python3
"""Gate 4 root-cause fix verification: VoicePipeline.start() must START the mic stream.

Before the fix: stream created but never started -> wake word + recognizer deaf.
"""
import sys
import time

sys.path.insert(0, r"D:\work - personal\Hexa Ai\hexa")
from hexa.config import load_config

load_config()

from hexa.voice import get_voice_pipeline
from hexa.audio import get_audio_stream

pipeline = get_voice_pipeline()
pipeline.start()
time.sleep(1.0)

stream = get_audio_stream()
print(f"stream running after pipeline.start(): {stream.is_running()}")
print(f"stream device={stream._capture_device} native={stream._capture_native_rate}")

pipeline.stop()
print("pipeline stopped cleanly")
