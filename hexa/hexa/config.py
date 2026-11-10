"""Configuration loading and validation using Pydantic."""
from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1280
    input_device: int | None = None


class WakeWordConfig(BaseModel):
    enabled: bool = True
    model_name: str = "hey_jarvis"
    inference_framework: str = "onnx"
    threshold: float = 0.5
    debounce_time: float = 1.0


class HotkeyConfig(BaseModel):
    enabled: bool = True
    key: str = "<ctrl>+<alt>+<space>"


class STTConfig(BaseModel):
    model: str = "small.en"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 5
    vad_threshold: float = 0.5


class TTSConfig(BaseModel):
    voice: str = "en_GB-alan-medium"
    voice_path: str = "models/en_GB-alan-medium.onnx"
    sample_rate: int = 22050
    fallback: str = "pyttsx3"


class VADConfig(BaseModel):
    enabled: bool = True
    silence_timeout: float = 1.5
    min_speech_duration: float = 0.5


class ConversationConfig(BaseModel):
    enabled: bool = True
    timeout: float = 10.0


class LLMConfig(BaseModel):
    model: str = "hexa-brain"
    host: str = "http://localhost:11434"
    num_ctx: int = 3072
    temperature: float = 0.2
    keep_alive: int = -1
    max_history: int = 6
    system_prompt: str = (
        "You are Hexa, a concise offline desktop assistant. "
        "Use tools for actions. Reply in at most 2 short sentences."
    )


class MemoryConfig(BaseModel):
    db_path: str = "hexa.db"


class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    bearer_token: str = ""


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HEXA_", case_sensitive=False)

    audio: AudioConfig = Field(default_factory=AudioConfig)
    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)
    hotkey: HotkeyConfig = Field(default_factory=HotkeyConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    apps: dict[str, str] = Field(default_factory=dict)
    sites: dict[str, str] = Field(default_factory=dict)
    search_templates: dict[str, str] = Field(default_factory=dict)
    tool_permissions: dict[str, str] = Field(default_factory=dict)


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from YAML file with defaults."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"

    config_path = Path(config_path)
    data: dict[str, Any] = {}

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    config = Config(**data)

    # Auto-generate bearer token if not set
    if not config.api.bearer_token:
        config.api.bearer_token = secrets.token_urlsafe(32)
        save_config(config, config_path)

    return config


def save_config(config: Config, config_path: str | Path) -> None:
    """Save configuration to YAML file."""
    config_path = Path(config_path)
    data = config.model_dump(mode="python")

    # Don't save the bearer token to the file (it's in the env/secrets)
    # Actually we do save it so it persists across restarts
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global config instance, loading if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config