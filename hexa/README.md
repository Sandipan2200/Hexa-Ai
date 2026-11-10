# Hexa - Offline Voice Assistant Core

A 100% offline, privacy-first voice assistant for Windows. No cloud APIs, no telemetry, no internet calls at runtime.

## Features

- **Wake word + Hotkey activation**: "Hey Jarvis" (placeholder) and Ctrl+Alt+Space push-to-talk
- **Hybrid command routing**: Fast regex path for common commands, LLM path for complex queries
- **Local LLM**: Uses Ollama with `hexa-brain` model (qwen3:4b-instruct on GPU), native tool calling verified working
- **STT**: faster-whisper `small.en` on CPU (loaded from local cache only)
- **TTS**: Piper `en_GB-alan-medium` with pyttsx3 fallback
- **VAD**: Silero VAD via ONNX
- **API**: FastAPI on 127.0.0.1 with bearer token auth
- **Memory**: SQLite for notes/facts
- **Tools**: Open apps, browser tabs, search, system status, timers, memory
- **App Discovery (Phase 8)**: Automatic discovery of installed Windows applications from Start Menu, Registry, App Paths, Microsoft Store, and System32; intelligent matching with aliases and fuzzy fallback; preferred apps for generic requests like "open my browser"

### How tool calling works (wire format)

Ollama 0.40.x's automatic tool-call parser (qwen3 template) requires the
OpenAI-style tool envelope:

```json
{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
```

Hexa's tool registry emits the plain `{name, description, parameters}` shape;
the wrapping happens once at the LLM client boundary (`hexa/llm.py`,
`_wire_tools`). The plain shape triggers the HTTP 400
`JSON schema error at #: properties must be an object` inside qwen3's parser.
Do not remove the wrapper.

### GPU residency & keep_alive

- Every chat request passes `keep_alive=-1`, so the model stays resident in
  VRAM indefinitely: `ollama ps` shows `UNTIL = Forever` after Hexa requests
  (the default 5-minute countdown would otherwise unload the model mid-use).
- `--check` reads actual residency via `ollama ps` with three outcomes:
  - `[PASS]` — model loaded on GPU (e.g. `100% GPU`, `78%/22% CPU/GPU`)
  - `[INFO]` — model known to Ollama but not loaded yet (no failure; run a chat first)
  - `[FAIL]` — model loaded on CPU, `ollama` CLI missing, or `ollama ps` failed

## Strict offline operation

Hexa makes **zero external network requests at runtime**. Enforcement:

1. Offline switches are set in `hexa/__init__.py` *before* any model library
   imports: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
   `HF_HUB_DISABLE_TELEMETRY=1`, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`.
2. Whisper is loaded with `local_files_only=True` (stt.py and `--check` alike).
   This was the one library that phoned `huggingface.co` on model load.
   Silero VAD, openWakeWord (local ONNX) and Piper (local file) make no calls.
3. Verified: with a dead proxy (`HTTP_PROXY=HTTPS_PROXY=http://127.0.0.1:9`) and
   a fresh log, a full `--check` + text session logs only `localhost` requests.

**Required local models** (download once while online):

| Component | What | Where |
|-----------|------|-------|
| LLM | `hexa-brain` in Ollama | `ollama list` shows it |
| STT | faster-whisper `small.en` | HF cache in `%USERPROFILE%\.cache\huggingface` |
| VAD | Silero ONNX | shipped via `silero-vad` pip package |
| Wake | `hey_jarvis_v0.1` ONNX | `%APPDATA%\openWakeWord` |
| TTS | `en_GB-alan-medium.onnx` + `.json` | `models/` in the project root |

**Manually verify no-Wi-Fi operation** (requires a real network toggle; only
the simulation was automated in CI):

1. Download all models above while online.
2. Turn off Wi-Fi/disconnect ethernet.
3. `...python.exe -m hexa --check` → all PASS/INFO.
4. `...python.exe -m hexa --text`, try: `hi`, `remember my exam is on Friday`,
   `what did i tell you about my exam`, `what time is it`.
5. Check `hexa.log` for any non-localhost `HTTP Request` lines — there must be none.
6. If a model is missing offline, Hexa fails with a clear message naming the
   exact model and the command to download it (no silent cloud fallback).

## Requirements

- Windows 10/11
- Python 3.12
- Ollama running with `hexa-brain` model
- NVIDIA GPU (for LLM) - RTX 2050 4GB VRAM tested

## Installation

```bash
# Create venv with Python 3.12
py -3.12 -m venv .venv

# Activate and install
.venv\Scripts\pip.exe install -r requirements.txt

# Download Piper voice
mkdir models
# Download en_GB-alan-medium.onnx and .json from HuggingFace to models/

# Download wake word model
.venv\Scripts\python.exe -c "from openwakeword.utils import download_models; download_models(['hey_jarvis_v0.1'])"
```

## Running

```bash
# Health check
.venv\Scripts\python.exe -m hexa --check

# Text mode (type commands)
.venv\Scripts\python.exe -m hexa --text

# Full service (API + voice)
.venv\Scripts\python.exe -m hexa --voice
```

Launching a second instance while one is running fails fast with a readable
one-line message (exit code 1, no model loading): "Cannot start Hexa: port
8000 is already in use (a running Hexa instance is listening there ...)".

Or use the batch file:
```
run_hexa.bat --check
run_hexa.bat --text
run_hexa.bat
```

## API Usage

The API runs on `http://127.0.0.1:8000` with bearer token auth (token in `hexa.log` on first run).

```bash
# Root and health - no token needed (liveness/info only, no secrets)
# Open http://127.0.0.1:8000/          -> welcome page
curl http://127.0.0.1:8000/health     # {"status": "ok", "service": "hexa", ...}

# Interactive docs
# Open http://127.0.0.1:8000/docs

# Get state
curl -H "Authorization: Bearer YOUR_TOKEN" http://127.0.0.1:8000/state

# Execute command
curl -X POST -H "Authorization: Bearer YOUR_TOKEN" -H "Content-Type: application/json" \
  -d '{"text": "what time is it"}' http://127.0.0.1:8000/command

# List tools
curl -H "Authorization: Bearer YOUR_TOKEN" http://127.0.0.1:8000/tools

# WebSocket events
wscat -c "ws://127.0.0.1:8000/events?token=YOUR_TOKEN"
```

## Fast-Path Commands

These run instantly without LLM:

- `stop` / `cancel` / `quit` / `exit` - Stop current operation
- `what time is it` - Current time
- `how is my laptop` / `system status` - CPU/RAM/battery
- `open notepad` / `open calculator` / `open chrome` - Launch whitelisted apps
- `open youtube` / `open github, gmail` - Open browser tabs
- `search python on youtube` / `look up rust on github` - Search sites

## LLM Commands

Everything else goes to the local LLM with tool calling:

- `remember my exam is on Friday`
- `what did I ask you to remember`
- `set a timer for 10 seconds`
- `what's the difference between a list and a tuple`
- `yo pull up flutter provider tutorial on google`

## App Discovery (Phase 8)

Hexa can automatically discover installed applications on Windows and launch them by name or natural-language description.

### Discovery Sources

Applications are discovered from trusted Windows sources only (no full-drive scanning):

1. **Start Menu shortcuts** — `.lnk` files in `ProgramData\Microsoft\Windows\Start Menu\Programs` and user `AppData`
2. **Registry Uninstall keys** — HKLM/HKCU `SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall` (both 32-bit and 64-bit views)
3. **App Paths registry** — `SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths`
4. **Microsoft Store / UWP apps** — Packaged applications via `Get-AppxPackage`
5. **Well-known System32 executables** — Notepad, Calculator, Paint, Terminal, Control Panel, etc.
6. **User-approved portable folders** — (Configurable, optional)

### Intelligent Matching

When you say "open X", Hexa matches in priority order:

1. Exact application ID
2. Exact display name
3. Normalized name (lowercase, alphanumeric only)
4. Configured aliases (e.g., "code" → "Visual Studio Code", "chrome" → "Google Chrome")
5. Prefix match on normalized name
6. Fuzzy match (typo-tolerant) above threshold

If multiple apps match closely, Hexa asks which one you mean instead of launching the wrong one.

### Preferred Apps for Generic Requests

Configure preferred apps for categories so generic requests work:

- "open my browser" → launches your preferred browser
- "open my editor" → launches your preferred code editor
- "open my terminal" → launches your preferred terminal
- "open my music player" → launches your preferred music app

Configure with `set preferred app <category> <app_name>` (categories: browser, editor, terminal, music, video, pdf, image, email).

### Commands

| Command | Description |
|---------|-------------|
| `refresh apps` | Re-scan Windows for installed applications (5-15s) |
| `list apps` | Show discovered applications with sources |
| `open <name>` | Launch an app by name (uses intelligent matching) |
| `set preferred app <category> <app>` | Set preferred app for a category |
| `get preferred apps` | Show all preferred app mappings |

### Examples

```
> refresh apps
Discovered and indexed 112 applications from 4 sources.

> open notepad
Opened Notepad

> open code
Opened Visual Studio Code

> open browser
Opened Google Chrome  (uses preferred browser)

> set preferred app editor code
Set Visual Studio Code as preferred editor.

> get preferred apps
Preferred applications:
  browser: Google Chrome
  editor: Visual Studio Code
```

### Safety & Security

- Only launches applications present in the discovered index or legacy config.yaml whitelist
- Never trusts arbitrary executable paths from the LLM
- No `shell=True` — uses `subprocess.Popen([exe_path])` with list arguments
- Validates executable exists before launching
- Handles UWP apps via `explorer.exe shell:AppsFolder\<AUMID>`
- Graceful handling of missing executables and launch failures

### Offline Guarantee

Discovery is 100% local:
- No network requests during discovery
- Uses only Windows APIs (Registry, WMI, PowerShell, file system)
- Results cached in local SQLite (`hexa.db`)
- `refresh apps` runs in background, doesn't block voice interaction

## Project Structure

```
hexa/
├── config.yaml          # Configuration (apps, sites, models, etc.)
├── requirements.txt     # Pinned dependencies
├── run_hexa.bat         # Windows launcher
├── hexa/
│   ├── __main__.py      # CLI entry (--check, --text, service)
│   ├── config.py        # Pydantic config loading
│   ├── events.py        # Event bus (pub/sub + async queue)
│   ├── state.py         # State machine (IDLE->LISTENING->...)
│   ├── audio.py         # Shared mic stream, levels, playback
│   ├── wake.py          # Wake word + hotkey
│   ├── stt.py           # faster-whisper transcription
│   ├── tts.py           # Piper + pyttsx3 fallback
│   ├── router.py        # Fast path + LLM path
│   ├── llm.py           # Ollama client
│   ├── memory.py        # SQLite notes/facts
│   ├── api.py           # FastAPI REST + WebSocket
│   └── tools/           # Tool registry + implementations
│       ├── apps.py      # open_app, refresh_apps, list_apps, preferred apps
│       ├── app_discovery.py  # Windows app discovery (Phase 8)
│       ├── app_index.py      # Persistent app index + matching (Phase 8)
│       ├── browser.py   # open_tabs, search
│       ├── system.py    # get_time, system_status
│       ├── notes.py     # remember, recall
│       └── timers.py    # set_timer, dummy tools
└── tests/               # 185 unit tests
```

## Configuration

Edit `hexa/config.yaml` to customize:

- `apps` whitelist for `open_app` (legacy, still supported)
- `sites` whitelist for `open_tabs`
- `search_templates` for search URLs
- `tool_permissions`: "safe", "confirm", "blocked" (includes new tools: `refresh_apps`, `list_apps`, `set_preferred_app`, `get_preferred_apps`)
- Audio, wake word, STT, TTS, LLM settings

Note: config overrides require a restart (`HEXA_*` env overrides apply to
top-level fields only; nested fields like `llm.host` must be edited in YAML).

## Testing

```bash
.venv\Scripts\python.exe -m pytest hexa/tests/ -v
```

185 tests passing, 0 skipped: config, launch port-guard, router fast-path (20+ phrases), full
native tool-calling cycle (execution, result feedback, final reply, invalid
tool name/args, blocked tools, round cap, LLM 400/unreachable fallbacks),
LLM client (envelope wrapping, keep_alive kwarg, typed errors, history),
offline guarantees (env vars, local_files_only, no cloud endpoints,
GPU-residency three states), tools, registry schemas, permissions, memory,
event bus, state machine, API auth, audio stream, wake word, hotkey, STT,
VAD, TTS (incl. interrupt/deadlock regression), voice pipeline,
**app discovery (Start Menu, Registry, App Paths, Store, System32), app index
persistence, name normalization, alias matching, fuzzy matching, ambiguous
matches, missing executables, safe launching, refresh/caching, preferred apps**.