"""Hexa CLI entry point."""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
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
        logging.FileHandler("hexa.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def run_check() -> int:
    """Run health checks and return exit code."""
    print("=" * 60)
    print("HEXA HEALTH CHECK")
    print("=" * 60)

    checks = []
    config = get_config()

    # 1. Python version
    import sys
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 12)
    checks.append(("Python version", ok, f"{py_version} {'PASS' if ok else 'FAIL (need 3.12+)'}"))

    # 2. Ollama reachable
    try:
        import ollama
        client = ollama.Client(host=config.llm.host)
        client.list()
        checks.append(("Ollama reachable", True, "PASS"))
    except Exception as e:
        checks.append(("Ollama reachable", False, f"FAIL {e}"))

    # 3. Model present
    try:
        models = client.list()
        model_names = [m.model for m in models.models]
        # Handle :latest suffix
        model_base = config.llm.model.split(":")[0]
        ok = any(model_base in m for m in model_names)
        checks.append(("Model present", ok, f"{config.llm.model} {'PASS' if ok else 'FAIL'}"))
    except Exception as e:
        checks.append(("Model present", False, f"FAIL {e}"))

    # 4. GPU residency (requires model to be running)
    try:
        import subprocess
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=5)
        gpu_ok = "100% GPU" in result.stdout and config.llm.model in result.stdout
        checks.append(("GPU residency", gpu_ok, "100% GPU PASS" if gpu_ok else "FAIL (run model first)"))
    except Exception as e:
        checks.append(("GPU residency", False, f"FAIL {e}"))

    # 5. Microphone found
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        has_input = any(d["max_input_channels"] > 0 for d in devices)
        checks.append(("Microphone found", has_input, "PASS" if has_input else "FAIL"))
    except Exception as e:
        checks.append(("Microphone found", False, f"FAIL {e}"))

    # 6. Piper voice file present
    voice_path = Path(config.tts.voice_path)
    if not voice_path.is_absolute():
        # Config is in hexa/config.yaml, so project root is parent of hexa/
        voice_path = Path(__file__).parent.parent.parent / config.tts.voice_path
    voice_ok = voice_path.exists()
    checks.append(("Piper voice file", voice_ok, f"{voice_path} {'PASS' if voice_ok else 'FAIL'}"))

    # 7. Whisper model loads
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(config.stt.model, device=config.stt.device, compute_type=config.stt.compute_type)
        checks.append(("Whisper model loads", True, f"{config.stt.model} PASS"))
    except Exception as e:
        checks.append(("Whisper model loads", False, f"FAIL {e}"))

    # 8. Wake word model loads
    try:
        from openwakeword import Model
        wakeword = Model(wakeword_models=[config.wake_word.model_name], inference_framework=config.wake_word.inference_framework)
        checks.append(("Wake word model loads", True, f"{config.wake_word.model_name} PASS"))
    except Exception as e:
        checks.append(("Wake word model loads", False, f"FAIL {e}"))

    # Print results
    all_pass = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}: {detail}")

    print("=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 60)

    return 0 if all_pass else 1


def run_text_mode() -> None:
    """Run in text mode (type commands, get replies)."""
    print("Hexa Text Mode - Type 'exit' to quit")
    print("-" * 40)

    router = get_router()

    while True:
        try:
            user_input = input("> ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                break
            if not user_input:
                continue

            start = time.perf_counter()
            reply = router.route(user_input)
            latency = time.perf_counter() - start

            print(f"Hexa: {reply}")
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