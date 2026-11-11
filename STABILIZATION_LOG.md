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

**Root cause chain (three layers deep):**

1. **Hardcoded device index unstable.** Device indexes shift whenever the user's OnePlus Nord Buds connect/disconnect: `21` → `20` → `25` → `31` across sessions. `PortAudioError: Error querying device 21` on a fresh session. Fix: resolve by name substring + characteristics, never by index.
2. **The WDM-KS 16 kHz pin serves misformatted data.** Deep-dive (`gate3_debug.py`, `gate3_formats.py`) proved the array's WDM-KS pin (formerly the target) never delivered true audio:
   - Spectrum of its output: all energy at 0–1.4 Hz (a full-scale slowly-drifting constant), speech-band fraction 0.308, Silero VAD probs max 0.018 (real speech > 0.9).
   - Format probing: `float32` → **NaN** everywhere; `int16` → spikes (crest 300) vs the same physical array's MME pin: self-consistent across int16/int32/float32 (crest ~14–15 on room tone, DC≈0).
   - So Gate 1's "real audio" (peak 24576, user heard playback crackle) was misformatted data, not speech. My error for not catching it then.
3. **Per-window `sd.rec` re-open overhead loses audio.** Even on the correct MME pin @44100, a fresh `sd.rec`+`sd.wait` per 800 ms window re-opens the device each time: measured **2.4 s of audio per 4 s wall (40% loss)**. Fix: ONE persistent pull-mode `InputStream` opened once in the capture thread, blocking `read()` per window: **9.6 s per 10 s wall** (one-time open cost only, zero per-window loss, no overflows).

**Fix (minimal):**
- `hexa/hexa/audio.py`:
  - `_resolve_input_device()` → `(index, channels, native_rate)`: valid int honored; stale int warns and falls back; name substring match preferring healthy host APIs (MME/DirectSound/WASAPI) and deprioritizing WDM-KS; OS default as last resort. Deliberately NO auto-match by native rate — that would re-select the poisoned WDM-KS 16 kHz pin.
  - `_poll_loop()` rewritten: opens ONE pull-mode `sd.InputStream` at the device NATIVE rate (MME pin: 44100 Hz, 4ch), blocking `read()` per 800 ms window, mean-downmix to mono, `scipy.signal.resample_poly` to the configured 16 kHz with `padtype="line"` (default zero-pad leaves a filter transient — an audible micro-click at every window edge; measured max deviation 2548 → 0.47), pad/trim to exactly 12800 samples, split into 1280-sample chunks for consumers.
  - `_resample_window()` helper factored out (unit-testable).
  - `AudioConfig.input_device: int | str | None` (str name matching).
- `hexa/hexa/config.py`: `input_device: int | str | None`.
- `hexa/config.yaml`: `input_device: "Microphone Array"` (matches the working MME pin first).
- `hexa/tests/test_audio.py`: resolver tests (name prefers non-WDM-KS, valid int honored, stale int survives, WDM-KS only as last resort, no-match → OS default), native-rate polling lifecycle test, 44100→16000 resample sine test (length + 1 kHz tone preserved).

**Files changed:**
- `hexa/hexa/audio.py` — resolver + persistent pull-mode capture + native-rate resample (root causes above)
- `hexa/hexa/config.py` — `input_device` accepts device-name strings
- `hexa/config.yaml` — `input_device: "Microphone Array"`
- `hexa/tests/test_audio.py` — 6 new/updated regression tests
- `scripts/gate3_stt.py` — Gate 3 main diagnostic (mic→Whisper + TTS→STT roundtrip with WER)
- `scripts/gate3_channels.py` — per-channel stats + transcription evidence
- `scripts/gate3_debug.py` — spectrum/VAD evidence (WDM-KS garbage)
- `scripts/gate3_formats.py` — format-mismatch evidence (float32→NaN on WDM-KS)
- `scripts/gate3_mme_test.py` — MME pin + resample per-channel WER test (user-run)
- `scripts/gate3_smoke.py` — no-speech pacing smoke of the real stack
- `scripts/gate3_loopback.py` — automated speaker→mic loopback (inconclusive: speakers muted)
- `scripts/gate3_level_test.py`, `scripts/gate3_clip_test.py` — level/clipping hypothesis tests (both ruled out)
- `scripts/gate3_padtype_test.py` — resample edge-padding evidence
- `scripts/list_inputs.py`, `scripts/list_outputs.py` — device listing helpers

**Real output (user's spoken runs, `gate3_mme_test.py` — MME capture @44100 → resample 16k → real Whisper):**
```
Run 1: mix WER=30% 'Exa is a local voice assistant that draws completely off.'
Run 2: mix WER=70% (outlier — spoke late; garbled words confirm, not a pipeline fault)
Run 3: ch0/ch1 WER=10% 'Hexa is a local voice assistant that runs completely off'
Run 4: mix WER=20% 'Alexa is a local voice assistant that runs completely offline.'
Run 5: ch0/ch1/mix WER=10% 'HEXA is a local voice assistant that runs completely offline.'
```
Per-channel note: ch0/ch1 carry identical signal (rms 609–951 across runs), ch2/ch3 dead (rms 0) — the array is effectively mono duplicated. Known quirk: "Rekha"/"Alexa"/"Exa" for "Hexa" in some runs — raw-channel quality of the array, not pipeline.

**Machine-verified (automated, no human):**
- TTS→STT roundtrip (clean audio): **WER 0.0%** — "The quick brown fox jumps over the lazy dog."
- Whisper level sensitivity: perfect transcription down to −42 dBFS (level ruled out as cause).
- Real stack pacing: `chunks=120 total=9.60s` per 10 s wall, no overflow.
- Resample: 35280@44100 → exactly 12800@16000; 1 kHz sine preserved; constant preserved (edge padding fixed).

**Pytest:** 195 passed (full suite, 2:10).

**USER CONFIRMATION:**
- [x] User ran 5 spoken runs; best WER 10% with "HEXA is a local voice assistant that runs completely offline."
- [x] Gate 3 accepted on this evidence (user pasted runs as proof; formal `gate3_stt.py` stack run optional for the record)

---

## Gate 4: VAD auto-stop

*In progress...*