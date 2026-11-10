# Hexa - Offline Voice Assistant Core

A 100% offline, privacy-first voice assistant for Windows. No cloud APIs, no telemetry, no internet calls at runtime.

## Features

- **Wake word + Hotkey activation**: "Hey Jarvis" (placeholder) and Ctrl+Alt+Space push-to-talk
- **Hybrid command routing**: Fast regex path for common commands, LLM path for complex queries
- **Local LLM**: Uses Ollama with `hexa-brain` model (qwen3:4b-instruct on GPU)
- **STT**: faster-whisper `small.en` on CPU
- **TTS**: Piper `en_GB-alan-medium` with pyttsx3 fallback
- **VAD**: Silero VAD via ONNX
- **API**: FastAPI on 127.0.0.1 with bearer token auth
- **Memory**: SQLite for notes/facts
- **Tools**: Open apps, browser tabs, search, system status, timers, memory

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
.venv\Scripts\python.exe -m hexa
```

Or use the batch file:
```
run_hexa.bat --check
run_hexa.bat --text
run_hexa.bat
```

## API Usage

The API runs on `http://127.0.0.1:8000` with bearer token auth (token in `hexa.log` on first run).

```bash
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
│       ├── apps.py      # open_app
│       ├── browser.py   # open_tabs, search
│       ├── system.py    # get_time, system_status
│       ├── notes.py     # remember, recall
│       └── timers.py    # set_timer, dummy tools
└── tests/               # 51 unit tests
```

## Configuration

Edit `hexa/config.yaml` to customize:

- `apps` whitelist for `open_app`
- `sites` whitelist for `open_tabs`
- `search_templates` for search URLs
- `tool_permissions`: "safe", "confirm", "blocked"
- Audio, wake word, STT, TTS, LLM settings

## Testing

```bash
.venv\Scripts\python.exe -m pytest hexa/tests/ -v
```

95 tests passing (7 skipped): config, router fast-path (20+ phrases), tools, registry schemas, permissions, memory, event bus, state machine, API auth, audio stream, wake word, hotkey, STT, VAD, TTS, voice pipeline.