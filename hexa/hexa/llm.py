"""Ollama LLM client for Hexa (100% local).

Wire-format note:
  Ollama's automatic tool-call parser (0.40.x, qwen3 template) requires the
  OpenAI-style tool envelope:  {"type": "function", "function": {...}}
  The tool registry emits the plain {name, description, parameters} shape,
  so wrapping happens here at the client boundary (see `_wire_tools`).
  Verified live: plain shape -> HTTP 400 "JSON schema error at #: properties
  must be an object"; wrapped shape -> native tool calls work.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx
import ollama

from hexa.config import get_config
from hexa.events import emit_event, EventType

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base class for recoverable LLM failures."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMConnectionError(LLMError):
    """Ollama is unreachable (not running, wrong host, network issue)."""


class LLMResponseError(LLMError):
    """Ollama returned an error response (e.g. HTTP 400 from tool parser)."""


def _wire_tools(tool_schemas: list[dict] | None) -> list[dict] | None:
    """Wrap registry schemas in the OpenAI envelope Ollama expects."""
    if not tool_schemas:
        return None
    return [{"type": "function", "function": tool} for tool in tool_schemas]


def extract_message(response: Any) -> dict[str, Any]:
    """Normalize an ollama chat response into a plain message dict.

    Works with both `ollama.ChatResponse` pydantic objects and raw dicts,
    and normalizes tool calls into `{"function": {"name": ..., "arguments": dict}}`.
    Never raises on strange output; callers get a safe structure back.
    """
    if isinstance(response, dict):
        msg = response.get("message", {}) or {}
    else:
        msg = getattr(response, "message", None)

    if msg is None:
        return {"role": "assistant", "content": "", "tool_calls": []}

    if isinstance(msg, dict):
        role = msg.get("role", "assistant")
        content = msg.get("content") or ""
        raw_calls = msg.get("tool_calls") or []
    else:
        role = getattr(msg, "role", "assistant") or "assistant"
        content = getattr(msg, "content", None) or ""
        raw_calls = getattr(msg, "tool_calls", None) or []

    tool_calls: list[dict[str, Any]] = []
    for tc in raw_calls:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            name = fn.get("name")
            args = fn.get("arguments")
        else:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", None) if fn else None
            args = getattr(fn, "arguments", None) if fn else None

        if not name:
            continue  # malformed call; drop it rather than crash
        try:
            args_dict = dict(args) if args else {}
        except Exception:
            args_dict = {}
        if not isinstance(args_dict, dict):
            args_dict = {}
        tool_calls.append({"function": {"name": name, "arguments": args_dict}})

    return {"role": role, "content": content, "tool_calls": tool_calls}


class LLMClient:
    """Wrapper around Ollama for Hexa's LLM needs."""

    def __init__(self) -> None:
        config = get_config()
        self.client = ollama.Client(host=config.llm.host)
        self.host = config.llm.host
        self.model = config.llm.model
        self.options = {
            "num_ctx": config.llm.num_ctx,
            "temperature": config.llm.temperature,
        }
        # keep_alive must be a top-level chat() kwarg, NOT inside options,
        # otherwise Ollama ignores it and the model unloads after the default 5 min.
        self.keep_alive = config.llm.keep_alive
        self.system_prompt = config.llm.system_prompt
        self.max_history = config.llm.max_history
        self._history: list[dict[str, Any]] = []

    def _build_messages(self, user_text: str, tool_schemas: list[dict] | None = None) -> list[dict]:
        """Build message list with system prompt and history."""
        messages = [{"role": "system", "content": self.system_prompt}]

        if tool_schemas:
            tools_desc = json.dumps(tool_schemas, indent=2)
            messages.append({
                "role": "system",
                "content": f"Available tools:\n{tools_desc}",
            })

        messages.extend(self._history[-(self.max_history * 2):])

        if user_text:
            messages.append({"role": "user", "content": user_text})

        return messages

    def chat(self, user_text: str, tool_schemas: list[dict] | None = None) -> dict | Any:
        """Send a chat request to the LLM. Raises LLMError on recoverable failures."""
        start = time.perf_counter()
        messages = self._build_messages(user_text, tool_schemas)

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=_wire_tools(tool_schemas),
                options=self.options,
                keep_alive=self.keep_alive,
            )
        except ollama.ResponseError as e:
            latency = time.perf_counter() - start
            status = getattr(e, "status_code", None)
            logger.error("LLM chat failed (%s, %.2fs): %s", status, latency, e)
            emit_event(EventType.ERROR, source="llm", error=str(e), status=status, latency=latency)
            raise LLMResponseError(str(e), status_code=status) from e
        except (httpx.ConnectError, httpx.TimeoutException, OSError) as e:
            latency = time.perf_counter() - start
            logger.error("LLM unreachable (%.2fs): %s (%s)", latency, self.host, e)
            emit_event(EventType.ERROR, source="llm", error=f"unreachable: {e}", latency=latency)
            raise LLMConnectionError(
                f"Cannot reach Ollama at {self.host} ({e.__class__.__name__})"
            ) from e
        except Exception as e:
            latency = time.perf_counter() - start
            logger.exception("Unexpected LLM chat failure")
            emit_event(EventType.ERROR, source="llm", error=str(e), latency=latency)
            raise LLMError(f"Unexpected LLM error: {e}") from e

        latency = time.perf_counter() - start
        emit_event(EventType.TOOL_CALL, tool="llm_chat", args={"messages": len(messages)}, latency=latency)
        return response

    def add_to_history(self, role: str, content: str, tool_calls: list | None = None) -> None:
        """Add a message to history."""
        msg = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self._history.append(msg)
        if len(self._history) > self.max_history * 2:
            self._history = self._history[-(self.max_history * 2):]

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()


# Global LLM client instance
_llm_client: LLMClient | None = None


def get_llm() -> LLMClient:
    """Get the global LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
