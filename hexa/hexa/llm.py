"""Ollama LLM client for Hexa."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import ollama
from hexa.config import get_config
from hexa.events import emit_event, EventType

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper around Ollama for Hexa's LLM needs."""

    def __init__(self) -> None:
        config = get_config()
        self.client = ollama.Client(host=config.llm.host)
        self.model = config.llm.model
        self.options = {
            "num_ctx": config.llm.num_ctx,
            "temperature": config.llm.temperature,
            "keep_alive": config.llm.keep_alive,
        }
        self.system_prompt = config.llm.system_prompt
        self.max_history = config.llm.max_history
        self._history: list[dict[str, Any]] = []

    def _build_messages(self, user_text: str, tool_schemas: list[dict] | None = None) -> list[dict]:
        """Build message list with system prompt and history."""
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add tool schemas as system message if provided
        if tool_schemas:
            tools_desc = json.dumps(tool_schemas, indent=2)
            messages.append({
                "role": "system",
                "content": f"Available tools:\n{tools_desc}"
            })

        # Add history (last N messages)
        messages.extend(self._history[-(self.max_history * 2):])

        # Add current user message
        messages.append({"role": "user", "content": user_text})

        return messages

    def chat(self, user_text: str, tool_schemas: list[dict] | None = None) -> dict:
        """Send a chat request to the LLM."""
        start = time.perf_counter()
        messages = self._build_messages(user_text, tool_schemas)

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=tool_schemas,
                options=self.options,
            )
            latency = time.perf_counter() - start
            emit_event(EventType.TOOL_CALL, tool="llm_chat", args={"messages": len(messages)}, latency=latency)
            return response
        except Exception as e:
            latency = time.perf_counter() - start
            logger.exception("LLM chat failed")
            emit_event(EventType.ERROR, source="llm", error=str(e), latency=latency)
            raise

    def add_to_history(self, role: str, content: str, tool_calls: list | None = None) -> None:
        """Add a message to history."""
        msg = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self._history.append(msg)
        # Trim history
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