"""Hexa CLI entry point."""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from hexa.config import load_config, get_config
from hexa.events import get_event_bus, EventType
from hexa.llm import get_llm
from hexa.memory import get_memory
from hexa.router import get_router
from hexa.state import get_state_machine
from hexa.tools import get_registry
from hexa.voice import get_voice_pipeline, shutdown_voice_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("hexa.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Human-readable status labels for the health-check table.
PASS, INFO, FAIL = "PASS", "INFO", "FAIL"


def safe_console(text: str) -> str:
    """Make arbitrary model/tool output safe for any Windows console codepage.

    Model replies can contain emoji; printing them raw crashes cp1252 consoles
    with UnicodeEncodeError. Replaces unencodable characters instead of raising.
    """
    encoding = (getattr(sys.stdout, "encoding", None) or "utf-8")
    try:
        text.encode(encoding)
        return text
    except UnicodeEncodeError:
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def check_gpu_residency(model_name: str) -> tuple[bool, str, str]:
    """Inspect actual model residency via `ollama ps` (local CLI, no network).

    Returns (is_failure, status_label, detail):
      - model loaded and GPU-resident          -> (True,  PASS, "100% GPU")
      - known model, not currently loaded      -> (True,  INFO, "...not loaded...")
      - model loaded but running on CPU        -> (False, FAIL, "loaded on CPU ...")
      - ollama CLI missing / server degraded   -> (False, FAIL, reason)
    `is_failure` only gates the overall PASS/FAIL of the health check; the
    label is what gets printed.
    """
    try:
        result = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=10
        )
    except FileNotFoundError:
        return False, FAIL, "ollama CLI not found on PATH"
    except subprocess.TimeoutExpired:
        return False, FAIL, "ollama ps timed out"
    except OSError as e:
        return False, FAIL, f"ollama ps failed: {e}"

    if result.returncode != 0:
        err = (result.stderr or "unknown error").strip()
        return False, FAIL, f"ollama ps error: {err}"

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    rows = lines[1:] if len(lines) > 1 else []

    # Locate the model row (match on the base name, e.g. "hexa-brain")
    model_base = model_name.split(":")[0].lower()
    row = next((line for line in rows if model_base in line.lower()), None)

    if row is None:
        return True, INFO, (
            f"{model_name} is available in Ollama but not loaded into memory "
            f"(start a chat to load it)"
        )

    # Classify the processor column by substring. Data rows have multi-token
    # values ("3.0 GB", "100% GPU"), so whitespace-split column indexes are
    # unreliable. Candidate values: "100% GPU", "100% CPU", "78%/22% CPU/GPU".
    # Model IDs are hex (digits a-f), so CPU/GPU substrings can only come from
    # the processor column.
    if re.search(r"CPU/GPU", row):
        pct = re.search(r"(\d+%/\d+%)", row)
        return True, PASS, (f"{pct.group(1)} CPU/GPU" if pct else "CPU/GPU")
    if re.search(r"GPU", row):
        pct = re.search(r"(\d+%)", row)
        return True, PASS, (f"{pct.group(1)} GPU" if pct else "GPU")
    if re.search(r"CPU", row):
        pct = re.search(r"(\d+%)", row)
        detail = f"loaded on CPU ({pct.group(1)})" if pct else "loaded on CPU"
        return False, FAIL, detail
    return False, FAIL, "residency unclear from ollama ps output"


def check_whisper_load(critical_model: str, device: str, compute_type: str) -> tuple[str, str]:
    """Load Whisper strictly from the local cache. Returns (label, detail)."""
    try:
        from faster_whisper import WhisperModel
        WhisperModel(critical_model, device=device, compute_type=compute_type, local_files_only=True)
        return PASS, f"{critical_model} (local cache)"
    except Exception as e:
        hint = (
            f"download '{critical_model}' once while online: "
            f".venv\\Scripts\\python.exe -c \"from faster_whisper import WhisperModel; "
            f"WhisperModel('{critical_model}')\""
        )
        return FAIL, f"{e} | {hint}"


def run_check() -> int:
    """Run health checks and return exit code.

    Each check yields (name, status_label, detail) where status_label is
    PASS / FAIL / INFO. Exit code 1 only if any check is FAIL (INFO means
    "works, but needs the model loaded first" and must not fail the report).
    """
    print("=" * 60)
    print("HEXA HEALTH CHECK")
    print("=" * 60)

    checks: list[tuple[str, str, str]] = []
    config = get_config()

    # 1. Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 12)
    checks.append(("Python version", PASS if ok else FAIL,
                   py_version if ok else f"{py_version} (need 3.12+)"))

    # 2. Ollama reachable (local service on 127.0.0.1, not an external call)
    try:
        import ollama
        client = ollama.Client(host=config.llm.host)
        client.list()
        checks.append(("Ollama reachable", PASS, "local API answered"))
    except Exception as e:
        checks.append(("Ollama reachable", FAIL, str(e)))

    # 3. Model present
    try:
        models = client.list()
        model_names = [m.model for m in models.models]
        # Handle :latest suffix
        model_base = config.llm.model.split(":")[0]
        ok = any(model_base in m for m in model_names)
        checks.append(("Model present", PASS if ok else FAIL, f"{config.llm.model}"))
    except Exception as e:  # includes NameError if the reachability check failed first
        checks.append(("Model present", FAIL, str(e)))

    # 4. GPU residency (three states: PASS / INFO-not-loaded / FAIL-on-CPU)
    gpu_ok, gpu_label, gpu_detail = check_gpu_residency(config.llm.model)
    checks.append(("GPU residency", gpu_label, gpu_detail))

    # 5. Microphone found
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        has_input = any(d["max_input_channels"] > 0 for d in devices)
        checks.append(("Microphone found", PASS if has_input else FAIL,
                       "input device present" if has_input else "no input device"))
    except Exception as e:
        checks.append(("Microphone found", FAIL, str(e)))

    # 6. Piper voice file present
    voice_path = Path(config.tts.voice_path)
    if not voice_path.is_absolute():
        # Project root: hexa/__main__.py -> hexa pkg -> hexa/ -> workspace root
        voice_path = Path(__file__).parent.parent.parent / config.tts.voice_path
    checks.append(("Piper voice file", PASS if voice_path.exists() else FAIL, str(voice_path)))

    # 7. Whisper model loads (from local HF cache only; never the network)
    label, detail = check_whisper_load(config.stt.model, config.stt.device, config.stt.compute_type)
    checks.append(("Whisper model loads", label, detail))

    # 8. Wake word model loads (local ONNX files)
    try:
        from openwakeword import Model
        Model(wakeword_models=[config.wake_word.model_name], inference_framework=config.wake_word.inference_framework)
        checks.append(("Wake word model loads", PASS, f"{config.wake_word.model_name} (local ONNX)"))
    except Exception as e:
        hint = (
            f"download once while online: "
            f".venv\\Scripts\\python.exe -c \"from openwakeword.utils import download_models; "
            f"download_models(['{config.wake_word.model_name}_v0.1'])\""
        )
        checks.append(("Wake word model loads", FAIL, f"{e} | {hint}"))

    # Print results
    for name, label, detail in checks:
        print(f"  [{label}] {name}: {detail}")

    any_fail = any(label == FAIL for _, label, _ in checks)
    print("=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED' if not any_fail else 'SOME CHECKS FAILED'}")
    print("=" * 60)

    return 0 if not any_fail else 1


def run_text_mode() -> None:
    """Run in text mode (type commands, get replies)."""
    print("Hexa Text Mode - Type 'exit' to quit")
    print("-" * 40)

    try:
        router = get_router()
    except Exception as e:
        logger.exception("Failed to initialize router")
        print(f"Failed to initialize: {e}")
        return

    while True:
        try:
            user_input = input("> ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                break
            if not user_input:
                continue

            start = time.perf_counter()
            try:
                reply = router.route(user_input)
            except Exception as e:
                # One failed request must never terminate the application.
                logger.exception("Unhandled error while routing request")
                reply = f"Error: {safe_console(str(e))}"
            latency = time.perf_counter() - start

            print(f"Hexa: {safe_console(reply)}")
            print(f"  (latency: {latency:.2f}s)")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            break


async def run_service(voice: bool = False) -> None:
    """Run the full service (API + optional voice)."""
    from hexa.api import create_app
    import uvicorn

    config = get_config()
    app = create_app()

    # Start voice pipeline if requested
    voice_pipeline = None
    if voice:
        logger.info("Starting voice pipeline...")
        voice_pipeline = get_voice_pipeline()
        voice_pipeline.start()

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def signal_handler() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler for SIGTERM
            pass

    # Start the server
    server_config = uvicorn.Config(
        app,
        host=config.api.host,
        port=config.api.port,
        log_level="info",
    )
    server = uvicorn.Server(server_config)

    # Run server and wait for shutdown
    server_task = asyncio.create_task(server.serve())
    shutdown_task = asyncio.create_task(shutdown_event.wait())

    try:
        await asyncio.wait([server_task, shutdown_task], return_when=asyncio.FIRST_COMPLETED)
    finally:
        if voice_pipeline:
            logger.info("Stopping voice pipeline...")
            shutdown_voice_pipeline()
        if not server_task.done():
            server.should_exit = True
            await server_task


def main() -> int:
    parser = argparse.ArgumentParser(description="Hexa Voice Assistant")
    parser.add_argument("--check", action="store_true", help="Run health checks")
    parser.add_argument("--text", action="store_true", help="Run in text mode")
    parser.add_argument("--voice", action="store_true", help="Run voice service (implies --service)")
    parser.add_argument("--service", action="store_true", help="Run API service only (no voice)")
    parser.add_argument("--config", type=Path, help="Config file path")

    args = parser.parse_args()

    # Load config early
    load_config(args.config)

    if args.check:
        return run_check()

    if args.text:
        run_text_mode()
        return 0

    # Default: run full service with voice
    voice_enabled = args.voice or (not args.service and not args.text and not args.check)
    try:
        asyncio.run(run_service(voice=voice_enabled))
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    return 0


if __name__ == "__main__":
    sys.exit(main())