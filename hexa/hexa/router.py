"""Hybrid router: fast regex path + LLM path with tools."""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from hexa.config import get_config
from hexa.events import emit_event, EventType
from hexa.llm import get_llm
from hexa.memory import get_memory
from hexa.tools import get_registry

logger = logging.getLogger(__name__)

# Fast-path regex patterns (ordered by priority)
FAST_PATTERNS = [
    # Stop/cancel
    (re.compile(r"^\s*(stop|cancel|quit|exit)\s*$", re.I), "stop"),
    # Time
    (re.compile(r"^\s*(what time is it|what's the time|current time|time now)\s*$", re.I), "get_time"),
    # System status
    (re.compile(r"^\s*(how is (my|the) (laptop|computer|system)|system status|resource usage|cpu|ram|battery)\s*$", re.I), "system_status"),
    # Open app
    (re.compile(r"^\s*open\s+(\w+)\s*$", re.I), "open_app"),
    # Open multiple sites
    (re.compile(r"^\s*open\s+([\w\s,]+(?:and\s+\w+)?)\s*$", re.I), "open_tabs"),
    # Search on site
    (re.compile(r"^\s*(search|look up|find)\s+(.+?)\s+on\s+(\w+)\s*$", re.I), "search"),
]

# Sites that can be opened (for fast-path disambiguation)
SITE_KEYWORDS = {"youtube", "gmail", "github", "chatgpt", "claude", "whatsapp", "localhost"}


class Router:
    """Routes commands to fast path or LLM path."""

    def __init__(self) -> None:
        self.llm = get_llm()
        self.registry = get_registry()
        self._load_fast_path_apps()

    def _load_fast_path_apps(self) -> None:
        """Load app names from config for fast-path matching."""
        config = get_config()
        self.app_names = set(config.apps.keys())
        self.site_names = set(config.sites.keys())

    def route(self, text: str) -> str:
        """Route a text command and return the response."""
        text = text.strip()
        if not text:
            return "I didn't catch that."

        # Try fast path first
        fast_result = self._try_fast_path(text)
        if fast_result is not None:
            return fast_result

        # Fall back to LLM path
        return self._llm_path(text)

    def _try_fast_path(self, text: str) -> str | None:
        """Try to match fast-path patterns."""
        for pattern, tool_name in FAST_PATTERNS:
            match = pattern.match(text)
            if match:
                logger.info("Fast path matched: %s -> %s", tool_name, text[:50])
                return self._execute_fast_path(tool_name, match, text)

        return None

    def _execute_fast_path(self, tool_name: str, match: re.Match, original_text: str) -> str:
        """Execute a fast-path tool."""
        try:
            if tool_name == "stop":
                emit_event(EventType.INTERRUPT, reason="user_command")
                return "Stopped."

            elif tool_name == "get_time":
                return self.registry.call("get_time")

            elif tool_name == "system_status":
                return self.registry.call("system_status")

            elif tool_name == "open_app":
                app_name = match.group(1).lower()
                # Verify it's in our whitelist
                if app_name in self.app_names:
                    return self.registry.call("open_app", name=app_name)
                return f"Unknown app: {app_name}. Available: {', '.join(sorted(self.app_names))}"

            elif tool_name == "open_tabs":
                # Parse comma-separated and "and" separated sites
                sites_text = match.group(1)
                # Split by comma or "and"
                sites = re.split(r"\s*,\s*|\s+and\s+", sites_text, flags=re.I)
                sites = [s.strip().lower() for s in sites if s.strip()]

                # Filter to known sites
                valid_sites = [s for s in sites if s in self.site_names]
                if not valid_sites:
                    return f"No valid sites. Available: {', '.join(sorted(self.site_names))}"

                return self.registry.call("open_tabs", sites=valid_sites)

            elif tool_name == "search":
                query = match.group(2).strip()
                site = match.group(3).lower().strip()
                return self.registry.call("search", site=site, query=query)

        except Exception as e:
            logger.exception("Fast path execution failed")
            return f"Error: {e}"

        return None

    def _llm_path(self, text: str) -> str:
        """Route through LLM with tool calling."""
        start = time.perf_counter()
        tool_schemas = self.registry.get_schemas()

        # First LLM call
        response = self.llm.chat(text, tool_schemas)

        # Add user message to history
        self.llm.add_to_history("user", text)

        # Handle tool calls (max 3 rounds)
        for round_num in range(3):
            message = response.get("message", {})
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                # No tool calls, final response
                reply = message.get("content", "").strip()
                self.llm.add_to_history("assistant", reply)
                latency = time.perf_counter() - start
                emit_event(EventType.REPLY, text=reply, source="llm", latency=latency)
                return reply

            # Execute tool calls
            self.llm.add_to_history("assistant", "", tool_calls)

            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name")
                args = func.get("arguments", {})

                logger.info("LLM tool call: %s(%s)", name, args)
                emit_event(EventType.TOOL_CALL, tool=name, args=args)

                try:
                    # Check permission
                    tool = self.registry.get_tool(name)
                    if tool and tool.permission == "confirm":
                        # In a real implementation, ask user for confirmation
                        # For now, auto-confirm with a note
                        logger.info("Tool %s requires confirmation (auto-confirmed)", name)
                        emit_event(EventType.TOOL_CALL, tool=name, args=args, confirm=True)

                    if tool and tool.permission == "blocked":
                        result = f"Tool {name} is blocked."
                    else:
                        result = self.registry.call(name, **args)

                    logger.info("Tool result: %s", str(result)[:100])
                    emit_event(EventType.TOOL_RESULT, tool=name, result=str(result))

                except Exception as e:
                    logger.exception("Tool call failed: %s", name)
                    result = f"Error: {e}"
                    emit_event(EventType.ERROR, source="tool", tool=name, error=str(e))

                # Add tool result to history
                self.llm.add_to_history("tool", str(result), tool_call_id=tc.get("id", ""))

            # Next LLM call with tool results
            response = self.llm.chat("", tool_schemas)

        # Max rounds reached
        reply = "Max tool rounds reached."
        self.llm.add_to_history("assistant", reply)
        latency = time.perf_counter() - start
        emit_event(EventType.REPLY, text=reply, source="llm", latency=latency)
        return reply


# Global router instance
_router: Router | None = None


def get_router() -> Router:
    """Get the global router instance."""
    global _router
    if _router is None:
        _router = Router()
    return _router