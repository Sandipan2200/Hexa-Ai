# Hexa Stabilization Log

## Gate 0: Baseline

**Health Check (`--check`):**
```
============================================================
HEXA HEALTH CHECK
============================================================
  [PASS] Python version: 3.12.10
  [PASS] Ollama reachable: local API answered
  [PASS] Model present: hexa-brain
  [INFO] GPU residency: hexa-brain is available in Ollama but not loaded into memory (start a chat to load it)
  [PASS] Microphone found: input device present
  [PASS] Piper voice file: D:\work - personal\Hexa Ai\models\en_GB-alan-medium.onnx
  [PASS] Whisper model loads: small.en (local cache)
  [PASS] Wake word model loads: hey_jarvis (local ONNX)
============================================================
OVERALL: ALL CHECKS PASSED
============================================================
```

**Pytest Suite (185/186 tests passed, 1 timed out):**
- All tests pass except `test_vad_energy_fallback` and `test_recognizer_start_stop` which timeout (likely waiting for real model loading)
- Voice pipeline tests: 11/11 passed

**Voice Mode Launch Command:**
```bash
cd "D:\work - personal\Hexa Ai\hexa"
..\.venv\Scripts\python.exe -m hexa --voice
```
Or via `run_hexa.bat` (no args defaults to `--voice`)

**First Real Run Output:**
```
2026-11-11 15:35:21,922 INFO __main__: Starting voice pipeline...
2026-11-11 15:35:22,094 INFO hexa.voice: Starting voice pipeline
2026-11-11 15:35:27,893 INFO hexa.tts: Piper voice loaded: D:\work - personal\Hexa Ai\models\en_GB-alan-medium.onnx
2026-11-11 15:35:27,893 INFO hexa.tts: Primary TTS: Piper
2026-11-11 15:35:28,022 INFO comtypes.client._code_cache: Imported existing <module 'comtypes.gen'...>
2026-11-11 15:35:28,414 INFO hexa.tts: pyttsx3 engine initialized
2026-11-11 15:35:28,414 INFO hexa.tts: Fallback TTS: pyttsx3
2026-11-11 15:35:31,713 INFO hexa.stt: Whisper model loaded locally: small.en
2026-11-11 15:35:36,646 INFO hexa.stt: Silero VAD model loaded
2026-11-11 15:35:39,969 INFO hexa.wake: Wake word model loaded: hey_jarvis
2026-11-11 15:35:39,970 INFO hexa.wake: Wake word detection started (threshold=0.50)
2026-11-11 15:35:39,982 INFO hexa.wake: Hotkey listener started: <ctrl>+<alt>+<space>
2026-11-11 15:35:39,982 INFO hexa.voice: Voice pipeline started
INFO:     Started server process [26304]
INFO:     Waiting for application startup.
2026-11-11 15:35:40,009 INFO hexa.api: Starting Hexa API
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```
**No errors on startup.** The service runs and listens. However, user reports errors during actual voice interaction. Proceeding to Gate 1.

---

## Gate 1: Audio Devices

**Root cause of "most voice parts throw errors":**
The Intel SST microphone array on this laptop is exposed by PortAudio through **two** different host APIs:
- Device 1 (`Microphone Array (2- Intel Sma...`), hostapi **MME**, default device — callbacks work but the device delivers garbage at non-native rates (peak=1 out of 32767, i.e. digital silence) when opened at 16 kHz.
- Device 21 (`Microphone Array 3 ()`), hostapi **Windows WDM-KS**, native **16 kHz, 4 channels** — delivers real audio via `sd.rec()` (peak 32643), but **PortAudio callbacks never fire** for it (`InputStream` starts, reports `active`, yet zero callback invocations).

Hexa's `audio.py` used exactly the broken combination: callback-based `InputStream` at 16 kHz on the default device. Result: `SpeechRecognizer`, `WakeWordDetector` and everything downstream starved — no chunks were ever delivered.

**Fix (minimal):**
- `hexa/hexa/audio.py`: added a polling capture mode (`AudioConfig.use_polling`) — a daemon thread captures 800 ms windows via `sd.rec()` (verified to deliver real audio on WDM-KS) and splits them into 1280-sample chunks for consumers. Shared `_process_chunk()` normalizes to int16 mono (channel 0) for both callback and polling paths.
- `hexa/hexa/config.py` + `hexa/config.yaml`: added `use_polling: true`, `channels: 4`, `input_device: 21` (the native 16 kHz WDM-KS device).

**Files changed:**
- `hexa/hexa/audio.py` — polling capture loop + shared chunk processing (root cause above)
- `hexa/hexa/config.py` — `use_polling` field
- `hexa/config.yaml` — select device 21, 4 channels, polling mode
- `hexa/tests/test_audio.py` — regression tests: polling lifecycle + multichannel→mono
- `scripts/gate1_audio_devices.py` — Gate 1 diagnostic (real hardware, no mocks)

**Diagnostic script:** `scripts/gate1_audio_devices.py`

**Real output (run with venv Python):**
```
Default input device:  [1] Microphone Array (2- Intel Sma
Default output device: [3] Speaker (2- Realtek(R) Audio)
Config input device:   [21] Microphone Array 3 () (hostapi: Windows WDM-KS)
Config channels:       4
Config polling mode:   True

Recording 3.0s via Hexa AudioStream (real voice-mode stack)...
  SPEAK NOW!
  Chunks: 40  Samples: 51200 (3.20s)
  Peak: 24576 (-2.5 dBFS)
  RMS:  307.4 (-40.6 dBFS)
  OK: real microphone signal captured

Playing 440 Hz tone for 1.0s (output device 3 @ 44100Hz)...
  Done
Playing back recording (3.20s)...
  Done
```
Also verified the **venv** python (what `run_hexa.bat` uses) receives real mic audio:
```
Python executable: D:\work - personal\Hexa Ai\.venv\Scripts\python.exe
Recording 3s... SPEAK NOW!
Chunks: 40  Peak: 28672  RMS: 423.0
OK: venv python receives real mic audio
```

**Pytest:** 188 passed (full suite).

**USER CONFIRMATION:**
- [x] 440 Hz tone heard — user: "yea i did hear it"
- [x] Recording playback heard / sounded clear — user: "pass"

---

## Gate 2: TTS

**Root cause found:**
`TTSManager.speak()` passes Piper's output — a complete WAV **file** (44-byte RIFF header + PCM) — to `play_audio_bytes*()`, which decoded it with `np.frombuffer(audio_bytes, np.int16)`. The 44-byte header was decoded as 22 int16 "samples" of garbage → an audible click at the start of **every** utterance, on both the Piper and pyttsx3 paths.

**Fix (minimal, one place):**
- `hexa/hexa/audio.py`: added `_decode_int16_bytes()` which detects RIFF/WAVE containers, parses them with the `wave` module, strips the header, returns the true rate (mono downmix when needed). `play_audio_bytes` / `play_audio_bytes_async` use it and trust the container rate over the caller's.

**Files changed:**
- `hexa/hexa/audio.py` — WAV-aware decoding in the two play_audio_bytes functions (root cause above)
- `hexa/tests/test_audio.py` — regression test `test_decode_int16_bytes_wav_header_stripped`
- `scripts/gate2_tts.py` — Gate 2 diagnostic (real Piper + pyttsx3 + interrupt)

**Diagnostic script:** `scripts/gate2_tts.py`

**Real output (venv python):**
```
Piper voice native rate: 22050 Hz
Config tts.sample_rate:  22050 Hz
Output device rate:      44100 Hz ([ 3] Speaker (2- Realtek(R) Audio))
voice vs config: MATCH

TEST 1: Piper speaks 'Hello, I am Hexa'
  Voice load: 5.31s
  Synthesis time: 0.26s
  WAV: rate=22050, ch=1, width=2B, frames=37376 (1.70s), bytes=74796, header=44
  Playback done (2.30s)
  Decoded audio: sr=22050, peak=1.000, rms=0.159

TEST 2: TTSManager.speak (the path voice mode actually uses)
  speak() returned in 0.27s
  Playback complete

TEST 3: pyttsx3 fallback
  Synthesis time: 0.11s, bytes=153022
  pyttsx3 WAV real rate: 22050 Hz; played at 22050 Hz -> OK

TEST 4: Interrupt mid-sentence
  Interrupt called at 0.301s after ~2.5s of speech
  is_speaking after interrupt: False
  OK: playback stopped
```

**Pytest:** 189 passed (full suite).

**USER CONFIRMATION:**
- [x] Heard "Hello, I am Hexa" via Piper, sounds natural / correct speed
- [x] Heard pyttsx3 fallback
- [x] Interrupt stopped speech mid-sentence
- (user: "pass"; repo connected to https://github.com/Sandipan2200/Hexa-Ai.git and pushed)

---

## Gate 3: STT

*In progress...*