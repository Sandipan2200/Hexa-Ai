"""Tests for the strict-offline runtime guarantees.

- Offline env vars are set as soon as `hexa` is imported (before any
  HF-dependency loads models).
- Whisper is loaded with local_files_only=True (no huggingface.co requests),
  both in stt.py and in the --check health check.
- A missing cached model produces a clear, actionable error.
- No cloud LLM/search endpoints exist in the source.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestOfflineEnvironment:
    def test_offline_env_vars_set_on_hexa_import(self) -> None:
        """`import hexa` (done at test collection) sets the offline switches."""
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        assert os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"

    def test_no_cloud_hosts_in_config(self) -> None:
        """Ollama is loopback-only; API binds to 127.0.0.1 (privacy requirement)."""
        from hexa.config import load_config
        cfg = load_config()
        assert cfg.llm.host in ("http://localhost:11434", "http://127.0.0.1:11434")
        assert cfg.api.host == "127.0.0.1"

    def test_source_has_no_cloud_endpoints(self) -> None:
        """Static guard: no cloud LLM/search endpoints anywhere in the package."""
        pkg = Path(__file__).parent.parent / "hexa"
        cloud_patterns = re.compile(
            r"api\.openai|api\.anthropic|generativelanguage|gemini\.google|"
            r"api\.perplexity|openrouter\.ai|api\.cohere",
            re.I,
        )
        for py in pkg.rglob("*.py"):
            text = py.read_text(encoding="utf-8", errors="ignore")
            match = cloud_patterns.search(text)
            assert match is None, f"cloud endpoint found in {py}: {match}"


class TestWhisperLocalLoading:
    def test_transcriber_uses_local_files_only(self) -> None:
        """WhisperTranscriber must pass local_files_only=True to WhisperModel."""
        from hexa.stt import STTConfig, WhisperTranscriber

        stt_cfg = STTConfig(model="small.en", device="cpu", compute_type="int8")
        with patch("faster_whisper.WhisperModel") as mock_model_cls:
            WhisperTranscriber(stt_cfg)
        kwargs = mock_model_cls.call_args.kwargs
        assert kwargs.get("local_files_only") is True
        assert kwargs.get("device") == "cpu"

    def test_missing_cached_model_gives_clear_error(self) -> None:
        from hexa.stt import STTConfig, WhisperTranscriber

        stt_cfg = STTConfig(model="small.en", device="cpu", compute_type="int8")
        with patch("faster_whisper.WhisperModel") as mock_model_cls:
            mock_model_cls.side_effect = Exception("LocalEntryNotFoundError...")
            with pytest.raises(RuntimeError) as exc_info:
                WhisperTranscriber(stt_cfg)
        # Error message must name the model and how to download it
        msg = str(exc_info.value)
        assert "small.en" in msg
        assert "download" in msg.lower()

    def test_health_check_whisper_load_is_local(self) -> None:
        """The --check whisper probe must also load from local cache."""
        from hexa.__main__ import FAIL, PASS, check_whisper_load

        with patch("faster_whisper.WhisperModel") as mock_model_cls:
            label, detail = check_whisper_load("small.en", "cpu", "int8")
        assert label == PASS
        assert mock_model_cls.call_args.kwargs.get("local_files_only") is True

        with patch("faster_whisper.WhisperModel") as mock_model_cls:
            mock_model_cls.side_effect = Exception("missing!")
            label, detail = check_whisper_load("small.en", "cpu", "int8")
        assert label == FAIL
        assert "download" in detail.lower() and "small.en" in detail


class TestGPUResidencyCheck:
    """Three-state GPU residency via `ollama ps` (local CLI, no network)."""

    HEADER = (
        "NAME              ID             SIZE      PROCESSOR    CONTEXT  RUNNER     UNTIL"
    )

    def _mock_run(self, stdout: str, returncode: int = 0):
        import subprocess
        fake = MagicMock(returncode=returncode, stdout=stdout, stderr="")
        return patch("subprocess.run", return_value=fake)

    def test_loaded_on_gpu_pass(self) -> None:
        from hexa.__main__ import PASS, check_gpu_residency
        stdout = (
            self.HEADER + "\n"
            "hexa-brain:latest 40878ce756f3 3.0 GB 100% GPU 3072 llamacpp 29 minutes from now\n"
        )
        with self._mock_run(stdout):
            ok, label, detail = check_gpu_residency("hexa-brain")
        assert ok is True and label == PASS and "GPU" in detail

    def test_loaded_on_cpu_fails(self) -> None:
        from hexa.__main__ import FAIL, check_gpu_residency
        stdout = (
            self.HEADER + "\n"
            "hexa-brain:latest 40878ce756f3 3.0 GB 100% CPU 3072 llamacpp 10 minutes from now\n"
        )
        with self._mock_run(stdout):
            ok, label, detail = check_gpu_residency("hexa-brain")
        assert ok is False and label == FAIL and "cpu" in detail.lower()

    def test_partial_gpu_still_passes(self) -> None:
        from hexa.__main__ import PASS, check_gpu_residency
        stdout = (
            self.HEADER + "\n"
            "hexa-brain:latest 40878ce756f3 3.0 GB 78%/22% CPU/GPU 3072 llamacpp 10 minutes from now\n"
        )
        with self._mock_run(stdout):
            ok, label, detail = check_gpu_residency("hexa-brain")
        assert ok is True and label == PASS

    def test_not_loaded_is_info_not_failure(self) -> None:
        from hexa.__main__ import INFO, check_gpu_residency
        # Old Ollama prints the header only when nothing is loaded; so does `ollama ps` with a warm-but-empty list
        with self._mock_run(self.HEADER):
            ok, label, detail = check_gpu_residency("hexa-brain")
        assert ok is True and label == INFO and "not loaded" in detail

    def test_model_not_in_specific_rows_is_info(self) -> None:
        from hexa.__main__ import INFO, check_gpu_residency
        stdout = (
            self.HEADER + "\n"
            "some-other-model 40878ce756f3 3.0 GB 100% GPU 4096 llamacpp 5 minutes from now\n"
        )
        with self._mock_run(stdout):
            ok, label, _ = check_gpu_residency("hexa-brain")
        assert ok is True and label == INFO

    def test_ollama_cli_missing_fails(self) -> None:
        import subprocess
        from hexa.__main__ import FAIL, check_gpu_residency
        with patch("subprocess.run", side_effect=FileNotFoundError("no ollama")):
            ok, label, detail = check_gpu_residency("hexa-brain")
        assert ok is False and label == FAIL and "not found" in detail

    def test_ollama_ps_timeout_fails(self) -> None:
        import subprocess
        from hexa.__main__ import FAIL, check_gpu_residency
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ollama ps", timeout=10)):
            ok, label, detail = check_gpu_residency("hexa-brain")
        assert ok is False and label == FAIL
