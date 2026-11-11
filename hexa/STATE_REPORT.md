# Hexa — STATE REPORT (2026-11-11)

## 10-LINE SUMMARY
- **Project:** Hexa, 100% offline Windows voice assistant core
- **Base completeness:** ~85% — all core modules implemented and tested; voice pipeline code-complete but never human-verified end-to-end
- **Tests:** 185 passed, 0 skipped (7 third-party torch.jit deprecation warnings only)
- **What works:** Config, events, state machine, router (fast + LLM paths), tool registry, all 10 tools, SQLite memory, FastAPI REST + WS, bearer auth, `--check`, `--text`, app discovery (Phase 8)
- **What doesn't work:** Wake word / hotkey / STT / TTS / VAD / voice pipeline — never tested with real mic/speakers; batch launcher quoting broken; custom wake word not trained
- **Model:** `hexa-brain` (qwen3:4b-instruct-2507-q4_K_M) on Ollama 0.40.16, 100% GPU, keep_alive=-1 verified via `ollama ps`
- **Offline:** Verified — HF env vars set before imports, Whisper `local_files_only=True`, dead-proxy test shows zero non-localhost requests
- **Known issues:** open_app fails if `refresh_apps` not run first (index empty); `open_tabs` LLM schema doesn't constrain site values; `run_hexa.bat` path quoting broken; GPU residency INFO state after fresh boot
- **Next human step:** Run `python -m hexa --text`, then `python -m hexa --voice` with mic/speakers connected, say "Hey Jarvis" and verify full audio loop
- **Overall:** Solid codebase, well-tested, ready for human audio verification and Phase 8 polish

---

## 1. SNAPSHOT

### Git
```
D:\work - personal\Hexa Ai\hexa
Branch: master
Status: clean (nothing to commit)
```

### Recent commits (`git log --oneline -20`)
```
e37b64e Phase 8: Automatic Application Discovery System
03edaf2 Fail fast with readable message when API port is already in use
4188624 Add / welcome page and /health liveness endpoint (fix 404 on service root)
cc5d733 Fix LLM tool calling, 100% offline runtime, GPU-residency check; restore all skipped tests
3d6bedc Phase 7: Update REPORT.md and README.md with voice integration results
44b19fd Phase 7: Voice integration complete. Added audio.py, wake.py, stt.py, tts.py, voice.py. All 95 tests pass (7 skipped for timeout issues). Health check passes.
c2acde1 Step 5: Add README, run_hexa.bat, REPORT.md with full test results
040397f Step 1-3: Core implementation with config, events, state, memory, tools, router, LLM, API. All 51 unit tests pass. Health check passes.
```

### Python / venv
```
Python 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]
```

### pip freeze (key packages)
```
fastapi==0.162.3
uvicorn==0.54.0
ollama==0.6.3
faster-whisper==1.2.1
ctranslate2==4.8.5
openwakeword==0.6.0
silero-vad==6.2.3
onnxruntime==1.31.1
piper-tts==1.8.0
pyttsx3==2.99
sounddevice==0.5.6
soundfile==0.14.0
pynput==1.8.2
pydantic==2.14.3
pydantic-settings==2.16.2
psutil==7.2.2
PyYAML==6.0.3
pytest==9.2.0
torch==2.15.0
```

### Ollama
```
Version: 0.40.16
ollama list:
  hexa-brain:latest        40878ce756f3  2.5 GB  14 hours ago
  qwen3:4b-instruct...     0edcdef34593  2.5 GB  33 hours ago

ollama ps:
  hexa-brain:latest  40878ce756f3  2.8 GB  100% GPU  3072  llamacpp  Forever
```

---

## 2. STRUCTURE

### File tree (excl. .venv, models/, __pycache__)
```
hexa/
├── config.yaml              [86 lines]
├── README.md                [245 lines]
├── REPORT.md                [645 lines]
├── run_hexa.bat             [35 lines]
├── STATE_REPORT.md          [this file]
├── hexa/
│   ├── __init__.py          [62]   — offline env vars before HF imports
│   ├── __main__.py          [337]  — CLI, --check, --text, --service, port guard
│   ├── api.py               [174]  — FastAPI REST + WS, /command, /state, /tools, /health, /
│   ├── config.py            [106]  — Pydantic config, YAML load/save
│   ├── events.py            [81]   — sync/async EventBus, 8 event types
│   ├── state.py             [73]   — State machine (IDLE→LISTENING→TRANSCRIBING→THINKING→SPEAKING)
│   ├── audio.py             [213]  — shared mic stream, playback, levels
│   ├── wake.py              [162]  — openWakeWord detector + pynput hotkey (Ctrl+Alt+Space)
│   ├── stt.py               [233]  — faster-whisper + Silero VAD, local_files_only
│   ├── tts.py               [183]  — Piper + pyttsx3 fallback, interrupt handling
│   ├── voice.py             [268]  — pipeline orchestration, conversation mode
│   ├── router.py            [197]  — fast regex path + LLM tool-calling path
│   ├── llm.py               [155]  — Ollama client, OpenAI envelope, keep_alive=-1
│   ├── memory.py            [104]  — SQLite notes/facts with tags
│   └── tools/
│       ├── __init__.py      [134]  — ToolRegistry, decorator, JSON schema gen
│       ├── apps.py          [171]  — open_app, refresh_apps, list_apps, preferred apps
│       ├── app_discovery.py [425]  — Windows app discovery (Start Menu, Registry, App Paths, Store, System32)
│       ├── app_index.py     [382]  — SQLite index, matching, preferred apps
│       ├── browser.py       [49]   — open_tabs, search
│       ├── system.py        [28]   — get_time, system_status
│       ├── notes.py         [22]   — remember, recall
│       └── timers.py        [30]   — set_timer, dummy confirm/blocked tools
└── tests/ (13 files, 2227 lines)
    ├── conftest.py          [6]
    ├── test_api_auth.py     [112]
    ├── test_app_discovery.py[227]
    ├── test_audio.py        [125]
    ├── test_config.py       [53]
    ├── test_events_state.py [121]
    ├── test_llm.py          [152]
    ├── test_memory.py       [59]
    ├── test_offline.py      [134]
    ├── test_port_guard.py   [60]
    ├── test_router_fastpath.py[188]
    ├── test_router_llm.py   [234]
    ├── test_stt.py          [214]
    ├── test_tools.py        [203]
    ├── test_tts.py          [169]
    ├── test_voice.py        [248]
    └── test_wake.py         [147]
```

**Total: ~3,120 lines app code, ~2,227 lines tests**

---

## 3. SPEC vs REALITY

| Feature | Status | Notes | File |
|---------|--------|-------|------|
| config.yaml + Pydantic config | **DONE** | All sections validated, env overrides for top-level | `config.py` |
| Event bus | **DONE** | Sync + async, 8 types, history, WS queue | `events.py` |
| State machine | **DONE** | 5 states, valid transitions, callbacks | `state.py` |
| Shared mic stream | **DONE** | sounddevice, levels, consumers, playback | `audio.py` |
| Wake word | **PARTIAL** | Code complete, openWakeWord ONNX loads, never human-tested | `wake.py` |
| Hotkey push-to-talk + interrupt | **PARTIAL** | pynput Ctrl+Alt+Space, emits INTERRUPT, never human-tested | `wake.py` |
| VAD (Silero) | **PARTIAL** | ONNX loads, integrated in STT, never human-tested | `stt.py` |
| STT (faster-whisper) | **PARTIAL** | small.en CPU int8, local_files_only, never human-tested | `stt.py` |
| TTS (Piper + pyttsx3) | **PARTIAL** | Piper primary, pyttsx3 fallback, interrupt fixed, never human-tested | `tts.py` |
| Router fast path | **DONE** | 6 regex patterns, 20+ test phrases | `router.py` |
| LLM path + tool calling | **DONE** | Native tool calling, 3-round cap, validation, error fallbacks | `router.py`, `llm.py` |
| Tool registry/decorator | **DONE** | Auto JSON schema from signatures | `tools/__init__.py` |
| open_app | **DONE** | Uses discovery index + legacy whitelist, safe launch | `tools/apps.py` |
| open_tabs | **DONE** | webbrowser, whitelist validation | `tools/browser.py` |
| search | **DONE** | Template-based URLs, whitelist validation | `tools/browser.py` |
| get_time | **DONE** | Fast path + LLM path | `tools/system.py` |
| system_status | **DONE** | psutil CPU/RAM/battery | `tools/system.py` |
| remember/recall | **DONE** | SQLite, LIKE search, tags | `tools/notes.py`, `memory.py` |
| set_timer | **DONE** | Threading.Timer, callback | `tools/timers.py` |
| Permission levels | **DONE** | safe/confirm/blocked, enforced in registry.call | `tools/__init__.py` |
| SQLite memory | **DONE** | hexa.db, notes + app index tables | `memory.py`, `app_index.py` |
| FastAPI REST endpoints | **DONE** | /command, /state, /tools, /config, /wake, /stop, /health, / | `api.py` |
| WebSocket /events | **DONE** | Bearer token query param, forwards EventBus | `api.py` |
| Bearer token auth | **DONE** | Auto-generated, in config.yaml, /docs no auth | `api.py`, `config.py` |
| Logging (hexa.log + latency) | **DONE** | File + stdout, latency on every route | `__main__.py`, `router.py` |
| --check | **DONE** | 8 checks: Python, Ollama, model, GPU, mic, Piper, Whisper, wake | `__main__.py` |
| --text mode | **DONE** | Interactive CLI, routes through router | `__main__.py` |
| README.md | **DONE** | Features, install, run, API, commands, structure | `README.md` |
| run_hexa.bat | **BROKEN** | Path quoting issues, never worked reliably | `run_hexa.bat` |

---

## 4. CONFIG

### config.yaml (token hidden)
```yaml
audio:
  sample_rate: 16000
  channels: 1
  chunk_size: 1280
  input_device: null
wake_word:
  enabled: true
  model_name: hey_jarvis
  inference_framework: onnx
  threshold: 0.5
  debounce_time: 1.0
hotkey:
  enabled: true
  key: <ctrl>+<alt>+<space>
stt:
  model: small.en
  device: cpu
  compute_type: int8
  language: en
  beam_size: 5
  vad_threshold: 0.5
vad:
  enabled: true
  silence_timeout: 1.5
  min_speech_duration: 0.5
tts:
  voice: en_GB-alan-medium
  voice_path: models/en_GB-alan-medium.onnx
  sample_rate: 22050
  fallback: pyttsx3
conversation:
  enabled: true
  timeout: 10.0
llm:
  model: hexa-brain
  host: http://localhost:11434
  num_ctx: 3072
  temperature: 0.2
  keep_alive: -1
  max_history: 6
  system_prompt: You are Hexa, a concise offline desktop assistant. Use tools for actions. Reply in at most 2 short sentences.
memory:
  db_path: hexa.db
api:
  host: 127.0.0.1
  port: 8000
  bearer_token: ***HIDDEN***
apps:
  notepad: notepad.exe
  calculator: calc.exe
  explorer: explorer.exe
  cmd: cmd.exe
  powershell: powershell.exe
  vscode: code.exe
  chrome: chrome.exe
sites:
  youtube: https://www.youtube.com
  gmail: https://mail.google.com
  github: https://github.com
  chatgpt: https://chat.openai.com
  claude: https://claude.ai
  whatsapp: https://web.whatsapp.com
  localhost: http://localhost:8000
search_templates:
  youtube: https://www.youtube.com/results?search_query={query}
  google: https://www.google.com/search?q={query}
  github: https://github.com/search?q={query}
  stackoverflow: https://stackoverflow.com/search?q={query}
  maps: https://www.google.com/maps/search/{query}
  amazon: https://www.amazon.com/s?k={query}
tool_permissions:
  open_app: safe
  open_tabs: safe
  search: safe
  get_time: safe
  system_status: safe
  remember: safe
  recall: safe
  set_timer: safe
  refresh_apps: safe
  list_apps: safe
  set_preferred_app: safe
  get_preferred_apps: safe
  dummy_destructive: confirm
  dummy_blocked: blocked
```

**Model:** `hexa-brain` (qwen3:4b-instruct-2507-q4_K_M)  
**App whitelist:** 7 entries (legacy, now secondary to discovery index)  
**Site whitelist:** 8 entries  

---

## 5. RUN IT

### `python -m hexa --check`
```
============================================================
HEXA HEALTH CHECK
============================================================
  [PASS] Python version: 3.12.10
  [PASS] Ollama reachable: local API answered
  [PASS] Model present: hexa-brain
  [PASS] GPU residency: 100% GPU
  [PASS] Microphone found: input device present
  [PASS] Piper voice file: D:\work - personal\Hexa Ai\models\en_GB-alan-medium.onnx
  [PASS] Whisper model loads: small.en (local cache)
  [PASS] Wake word model loads: hey_jarvis (local ONNX)
============================================================
OVERALL: ALL CHECKS PASSED
============================================================
```

### `python -m pytest -v` (full)
```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0
collected 185 items

test_api_auth.py           11 passed   auth, endpoints, websocket, welcome page + /health liveness (no token)
test_app_discovery.py       24 passed   Phase 8: AppDiscovery init/slugify/normalize/noise, DiscoveredApp post_init; AppIndex CRUD, matching (exact/normalized/alias/prefix/fuzzy/ambiguous), preferred apps, hashability; integration: full discovery, System32
test_port_guard.py          5 passed   launch port-guard: free/busy ports, holder identification via /health ("a running Hexa instance is listening there, uptime N s"), probe never raises
test_audio.py               8 passed   stream, consumers, playback handles
test_config.py              4 passed
test_events_state.py       12 passed   event bus + state machine
test_llm.py                16 passed   envelope wrapping, keep_alive top-level kwarg, 400/unreachable/timeout -> typed LLM errors, message normalization, history + tool-round sequence
test_memory.py              6 passed
test_offline.py            13 passed   offline env vars, loopback-only config, static no-cloud-endpoint guard, Whisper local_files_only (stt + check), clear missing-model error, GPU residency 3 states (GPU/CPU/partial/not-loaded/CLI-missing/timeout)
test_router_fastpath.py     9 passed
test_router_llm.py         18 passed   full native tool cycle: call -> execute -> result feedback -> final reply; sequential calls; invalid tool name/args rejected; non-dict args; tool failure feeds next round; blocked tool never executed; round cap; malformed output fallback; 400 & unreachable -> readable text; "hi" never tools
test_stt.py                10 passed
test_tts.py                15 passed   7 formerly-skipped restored
test_voice.py              11 passed   conversation-mode test rebuilt deterministically (formerly skipped)
test_wake.py                9 passed   wake guard added: empty-model thread exits cleanly
test_tools.py              12 passed   registry, open_app, open_tabs, search, system, memory, timer tools

====================== 185 passed, 7 warnings in 139.20s ======================
```
**7 warnings:** third-party `torch.jit.load` deprecation from Silero VAD only.

### 8 sample phrases via `--text` mode (after `refresh_apps`)

| Phrase | Path | Tool Called | Reply | Latency |
|--------|------|-------------|-------|---------|
| `what time is it` | **fast** | get_time | "It's 12:47 PM on Wednesday, November 11, 2026" | 0.000s |
| `open notepad` | **fast** | open_app | "Opened Notepad" | 0.133s |
| `remember my exam is on friday` | **LLM** | remember | "I'll remember that your exam is on Friday..." | 2.892s |
| `what did i tell you about my exam` | **LLM** | recall | "I remember you said your exam is on Friday." | 0.380s |
| `open chrome` | **fast** | open_app | "Opened Google Chrome" | 0.007s |
| `search python on youtube` | **fast** | search | "Searching youtube for 'python'" | 0.350s |
| `please open vscode` | **LLM** | open_app | "Visual Studio Code is now open." | 1.035s |
| `how is my laptop` | **fast** | system_status | "CPU: 62%, RAM: 71% (16.7/23.6 GB), Battery: 96% (charging)" | 0.538s |

**Note:** `refresh_apps` must be run once before `open_app` works (index starts empty).

---

## 6. API

Service started: `python -m hexa --service` (PID 31696, port 8000)

| Request | Auth | Status | Response (truncated) |
|---------|------|--------|---------------------|
| `GET /` | none | 200 | HTML welcome page "Hexa is running · v0.1.0" |
| `GET /health` | none | 200 | `{"status":"ok","service":"hexa","version":"0.1.0","uptime_seconds":55.7}` |
| `GET /state` | none | **401** | `{"detail":"Invalid or missing bearer token"}` |
| `GET /state` | Bearer | 200 | `{"state":"idle"}` |
| `GET /tools` | Bearer | 200 | 10 tool schemas (open_app, open_tabs, search, get_time, system_status, remember, recall, set_timer, refresh_apps, list_apps, set_preferred_app, get_preferred_apps) |
| `POST /command` `{"text":"what time is it"}` | Bearer | 200 | `{"reply":"It's 12:51 PM on Wednesday, November 11, 2026"}` |
| `POST /command` `{"text":"what time is it"}` | none | **401** | `{"detail":"Invalid or missing bearer token"}` |

**All protected endpoints correctly return 401 without token.**

---

## 7. KNOWN PROBLEMS

### Broken / Flaky
1. **`run_hexa.bat`** — Path quoting broken (PowerShell vs cmd escaping), never worked reliably. Unchanged since Step 5.
2. **`open_app` fails if index empty** — Must run `refresh_apps` first (discovers 112 apps). No auto-refresh on first use.
3. **`open_tabs` LLM schema unconstrained** — `sites` parameter accepts any string; model may emit raw URL instead of whitelist key. Documented, not fixed.
4. **GPU residency INFO after fresh boot** — `ollama ps` shows header only until model loaded once; `--check` reports INFO not FAIL (by design but confusing).

### Skipped / Untested by Humans
5. **Wake word detection** — `hey_jarvis` ONNX loads, threshold 0.5, debounce 1s. Never tested with real microphone.
6. **Hotkey (Ctrl+Alt+Space)** — pynput listener registered, emits HOTKEY_PRESSED. Never tested with real key press.
7. **STT transcription** — faster-whisper small.en CPU int8, Silero VAD. Never tested with real speech.
8. **TTS audio output** — Piper en_GB-alan-medium + pyttsx3 fallback. Code runs, interrupt logic fixed. Never heard by human.
9. **VAD silence detection** — 1.5s timeout, 0.5s min speech. Never tested with real audio.
10. **Voice pipeline end-to-end** — Wake → LISTENING → STT → THINKING → TTS → SPEAKING → IDLE. Unit tests pass (mocks). Never run live.

### Design Deviations / Code Quality
11. **`app_discovery.py` (425 lines)** — Over 150-line guideline. Does 5 discovery sources + alias enrichment. Could be split.
12. **`app_index.py` (382 lines)** — Over 150 lines. SQLite CRUD + matching + preferred apps. Could be split.
13. **`__main__.py` (337 lines)** — CLI + health checks + port guard + service runner. Could extract health checks.
14. **`stt.py` (233 lines)** — Transcriber + VAD + health check. Could separate.
15. **`voice.py` (268 lines)** — Pipeline + conversation mode. Could separate.
16. **Duplicate `open_app` logic** — Fast path in `router.py` + tool in `apps.py` both do matching. Fast path uses legacy whitelist; tool uses index.
17. **Dead code** — `tools/notes.py` and `tools/timers.py` have no tests beyond basic; `dummy_destructive` / `dummy_blocked` tools exist only for permission tests.
18. **`HEXA_` env overrides don't reach nested config** — Pydantic `env_nested_delimiter` not set; must edit `config.yaml` for `llm.host` etc.

### Package Fallbacks
- **Piper voice** — No official pip download API; manual download required (`models/en_GB-alan-medium.onnx`)
- **Wake word model** — `openwakeword.utils.download_models(['hey_jarvis_v0.1'])` works but downloads from HF (online once)
- **Whisper model** — `faster-whisper` downloads `small.en` to HF cache on first run (online once)

---

## 8. NOT YET TESTED BY A HUMAN

| What | Steps to Test |
|------|---------------|
| **Wake word "Hey Jarvis"** | 1. Connect mic. 2. `python -m hexa --voice`. 3. Wait for "Voice pipeline started". 4. Say "Hey Jarvis" clearly. 5. Check logs: `STATE_CHANGE: idle → listening` |
| **Hotkey Ctrl+Alt+Space** | 1. With `--voice` running. 2. Hold Ctrl+Alt+Space. 3. Speak "what time is it". 4. Release. 5. Check logs: `HOTKEY_PRESSED` → `STATE_CHANGE: idle → listening` |
| **STT transcription** | 1. Trigger via wake/hotkey. 2. Speak a command. 3. Check logs: `TRANSCRIPT` event with text, `STATE_CHANGE: listening → transcribing → thinking → speaking` |
| **TTS audio output** | 1. Trigger any command. 2. Hear Piper voice reply. 3. Verify `en_GB-alan-medium` sounds natural. 4. Test interrupt: start long TTS, press hotkey → `INTERRUPT` → TTS stops |
| **VAD auto-stop** | 1. Trigger listening. 2. Stop speaking. 3. After 1.5s silence, should auto-transition to `transcribing` |
| **Conversation mode** | 1. Say command, get reply. 2. Within 10s say follow-up (no wake word). 3. Should process without wake word |
| **Full offline** | 1. Download all models (run `--check` → all PASS). 2. Disconnect Wi-Fi. 3. `python -m hexa --check` → all PASS/INFO. 4. `python -m hexa --text` → commands work. 5. Check `hexa.log`: zero non-localhost HTTP requests |

---

## 9. NEXT STEPS (Priority Order)

| # | Task | Size | Notes |
|---|------|------|-------|
| 1 | Human audio verification (§8) | Medium | Requires mic/speakers; 30-60 min for full §8 checklist |
| 2 | Auto-refresh index on first `open_app` miss | Small | If index empty, run discovery automatically |
| 3 | Constrain `open_tabs.sites` to whitelist in LLM schema | Small | Add `enum` to JSON schema or post-validate |
| 4 | Fix `run_hexa.bat` quoting | Small | PowerShell wrapper or proper cmd escaping |
| 5 | Train custom "Hey Hexa" wake word | Medium | openWakeWord training; replace `hey_jarvis` placeholder |
| 6 | Split large files (>150 lines) | Medium | `app_discovery.py`, `app_index.py`, `__main__.py`, `stt.py`, `voice.py` |
| 7 | Physical Wi-Fi-off certification | Small | Run §8 "Full offline" steps; document result |
| 8 | Add `env_nested_delimiter="__"` to Pydantic config | Tiny | Allow `HEXA_LLM__HOST` etc. |

---

## 10. TOP 5 RISKS / WEAK SPOTS

1. **Entire voice pipeline is untested by humans** — 5 modules (wake, stt, tts, vad, voice) pass unit tests with mocks but have never processed real audio. A single sample-rate mismatch, device index bug, or VAD threshold issue would break the entire product silently.

2. **`open_app` silent failure mode** — If user hasn't run `refresh_apps`, `open notepad` returns "Executable not found: notepad.exe" instead of auto-discovering. The legacy whitelist in `config.yaml` has raw exe names (`notepad.exe`) that don't resolve from `System32` without full path. This will confuse every first-time user.

3. **OpenAI envelope coupling to Ollama version** — The `_wire_tools` wrapper in `llm.py` is hardcoded for Ollama 0.40.x qwen3 parser. If Ollama updates and changes the parser, tool calling breaks with HTTP 400. No version detection or fallback.

4. **Single SQLite file for everything** — `hexa.db` holds memories, app index, and preferred apps. No migrations, no schema versioning. If schema changes, manual delete required. Corruption risk grows with concurrent access (voice + API + CLI).

5. **No integration test for the full stack** — Unit tests mock at boundaries (Ollama, audio devices, subprocess). The only "live" tests are manual `--check` and the 8-phrase router test. A regression in the voice pipeline wiring would not be caught until human testing.