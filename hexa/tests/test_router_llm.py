"""Tests for the router's LLM path: the complete native tool-calling cycle.

The LLM (external dependency) is mocked at the `get_llm` boundary; the router
logic under test is real: tool-call validation against the registry, tool
execution, tool-result feedback round-trip, and error fallbacks.

Router -> LLM contract (see hexa/llm.py):
  chat(user_text, tool_schemas)          # 2 positional args
  add_to_history(role, content, tool_calls=None)
The tool envelope (`_wire_tools`) is applied inside the real LLMClient and is
covered by test_llm.py; here the router passes raw registry schemas.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hexa.tools import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    """A fresh registry with real, in-memory tools."""
    reg = ToolRegistry()

    @reg.register
    def get_time() -> str:
        """Get the current date and time."""
        return "It's 10:00."

    @reg.register
    def open_app(name: str) -> str:
        """Open an application from the whitelist."""
        if name not in {"notepad", "calculator"}:
            raise ValueError("Unknown app")
        return f"Opened {name}."

    @reg.register
    def boom() -> str:
        """A tool that always fails."""
        raise RuntimeError("tool exploded")

    return reg


@pytest.fixture
def router_factory(registry: ToolRegistry):
    """Builds a Router with the LLM boundary mocked and a real registry."""
    from hexa.router import Router

    with patch("hexa.router.get_llm") as mock_get_llm, \
         patch("hexa.router.get_registry", return_value=registry), \
         patch("hexa.router.get_config") as mock_cfg:
        mock_cfg.return_value = MagicMock(
            apps={"notepad": "notepad.exe", "calculator": "calc.exe"},
            sites={"youtube": "https://www.youtube.com", "github": "https://github.com"},
        )
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        router = Router()
        yield router, mock_llm


def _resp(content: str = "", tool_calls: list | None = None) -> MagicMock:
    """Fake ollama ChatResponse-shaped object."""
    return MagicMock(
        message=MagicMock(content=content, tool_calls=tool_calls or [], role="assistant")
    )


def _history_of(mock_llm: MagicMock) -> list[tuple]:
    """All add_to_history calls as (role, content, tool_calls) triples."""
    out = []
    for c in mock_llm.add_to_history.call_args_list:
        role = c.args[0]
        content = c.args[1] if len(c.args) > 1 else c.kwargs.get("content")
        tool_calls = c.args[2] if len(c.args) > 2 else c.kwargs.get("tool_calls")
        out.append((role, content, tool_calls))
    return out


def _tool_messages(mock_llm: MagicMock) -> list[str]:
    return [content for role, content, _ in _history_of(mock_llm) if role == "tool"]


class TestFinalResponse:
    def test_plain_chat_no_tool_call(self, router_factory) -> None:
        router, mock_llm = router_factory
        mock_llm.chat.return_value = _resp("Hello! How can I help?")
        assert router.route("hi") == "Hello! How can I help?"
        # "hi" must NOT trigger a tool call and must hit the LLM (not fast path)
        mock_llm.chat.assert_called_once()
        assert mock_llm.chat.call_args.args[0] == "hi"
        # history records the user turn and the final assistant reply
        hist = _history_of(mock_llm)
        assert hist[0] == ("user", "hi", None)
        assert hist[-1] == ("assistant", "Hello! How can I help?", None)

    def test_empty_registry_still_chats(self, router_factory, registry) -> None:
        """Normal chat must work when no tools exist (empty registry)."""
        router, mock_llm = router_factory
        registry._tools.clear()
        mock_llm.chat.return_value = _resp("Hello!")
        assert router.route("hello there") == "Hello!"
        assert mock_llm.chat.call_args.args[1] == []  # no tool schemas sent


class TestToolCycle:
    def test_complete_cycle_tool_result_final(self, router_factory) -> None:
        """user -> LLM(tool_call) executes -> result fed back -> LLM final reply."""
        router, mock_llm = router_factory
        mock_llm.chat.side_effect = [
            _resp("", [{"function": {"name": "get_time", "arguments": {}}}]),
            _resp("It's 10:00 right now."),
        ]
        reply = router.route("what time is it right now please")

        assert reply == "It's 10:00 right now."
        assert mock_llm.chat.call_count == 2
        # The second call is the follow-up with the tool result in history
        assert mock_llm.chat.call_args_list[1].args[0] == ""

        # Full native tool-calling history sequence
        hist = _history_of(mock_llm)
        assert hist[0] == ("user", "what time is it right now please", None)
        assert hist[1] == ("assistant", "", [
            {"function": {"name": "get_time", "arguments": {}}}
        ])
        assert hist[2] == ("tool", "It's 10:00.", None)
        assert hist[3] == ("assistant", "It's 10:00 right now.", None)

    def test_multiple_sequential_tool_calls(self, router_factory) -> None:
        """One response carrying several tool calls executes all of them."""
        router, mock_llm = router_factory
        mock_llm.chat.side_effect = [
            _resp("", [
                {"function": {"name": "open_app", "arguments": {"name": "notepad"}}},
                {"function": {"name": "get_time", "arguments": {}}},
            ]),
            _resp("Opened notepad; time is 10:00."),
        ]
        # "please open ..." avoids the fast-path 'open X' regex so we exercise the LLM path
        reply = router.route("please open notepad and tell me the time")
        assert reply == "Opened notepad; time is 10:00."

        tool_msgs = _tool_messages(mock_llm)
        assert tool_msgs == ["Opened notepad.", "It's 10:00."]

    def test_invalid_tool_name_rejected(self, router_factory) -> None:
        """LLM inventing a non-registered tool must be blocked, never executed."""
        router, mock_llm = router_factory
        mock_llm.chat.side_effect = [
            _resp("", [{"function": {"name": "rm_rf_everything", "arguments": {}}}]),
            _resp("Sorry, I can't do that."),
        ]
        reply = router.route("please delete everything")

        assert reply == "Sorry, I can't do that."
        tool_msgs = _tool_messages(mock_llm)
        assert "unknown tool 'rm_rf_everything'" in tool_msgs[0].lower()

    def test_invalid_arguments_rejected(self, router_factory) -> None:
        """Wrong-named arguments reaching the registry signature are caught."""
        router, mock_llm = router_factory
        # open_app(name: str) called with a wrong-named argument dict
        # ("kick off ..." avoids the fast-path 'open X' regex)
        mock_llm.chat.side_effect = [
            _resp("", [{"function": {"name": "open_app", "arguments": {"appname": "notepad"}}}]),
            _resp("Which app did you mean?"),
        ]
        router.route("kick off notepad for me")
        tool_msgs = _tool_messages(mock_llm)
        assert "invalid arguments" in tool_msgs[0].lower()

    def test_non_dict_arguments_rejected(self, router_factory) -> None:
        """Scalar arguments instead of an object must be rejected safely."""
        router, mock_llm = router_factory
        mock_llm.chat.side_effect = [
            _resp("", [{"function": {"name": "open_app", "arguments": "notepad"}}]),
            _resp("Let me retry."),
        ]
        router.route("please open notepad")
        tool_msgs = _tool_messages(mock_llm)
        assert "invalid arguments" in tool_msgs[0].lower()

    def test_tool_execution_failure_returned_to_llm(self, router_factory) -> None:
        """Tool failures do NOT crash the router; error text feeds the next LLM round."""
        router, mock_llm = router_factory
        mock_llm.chat.side_effect = [
            _resp("", [{"function": {"name": "boom", "arguments": {}}}]),
            _resp("The tool failed; nothing was done."),
        ]
        reply = router.route("do the boom")
        assert reply == "The tool failed; nothing was done."
        tool_msgs = _tool_messages(mock_llm)
        assert "tool exploded" in tool_msgs[0]

    def test_blocked_tool_not_executed(self, router_factory, registry) -> None:
        """Blocked tools are never invoked and produce a 'blocked' result."""
        router, mock_llm = router_factory
        registry.set_permissions({"boom": "blocked"})
        mock_llm.chat.side_effect = [
            _resp("", [{"function": {"name": "boom", "arguments": {}}}]),
            _resp("That action is not allowed."),
        ]
        router.route("do the boom")
        tool_msgs = _tool_messages(mock_llm)
        assert "blocked" in tool_msgs[0].lower()
        # boom must not have been invoked (no 'tool exploded' error text)
        assert "tool exploded" not in tool_msgs[0]

    def test_rounds_capped_after_three(self, router_factory) -> None:
        """A tool-calling loop must stop after 3 rounds."""
        router, mock_llm = router_factory
        tool_call_resp = _resp("", [{"function": {"name": "get_time", "arguments": {}}}])
        mock_llm.chat.side_effect = [tool_call_resp for _ in range(4)]  # initial + 3 follow-ups
        assert router.route("loop forever please") == "Max tool rounds reached."
        assert mock_llm.chat.call_count == 4

    def test_malformed_empty_output_fails_safely(self, router_factory) -> None:
        router, mock_llm = router_factory
        mock_llm.chat.return_value = _resp("", tool_calls=[])
        reply = router.route("hi")
        assert "couldn't process" in reply.lower()


class TestLLMFailures:
    def test_http_400_returns_readable_error(self, router_factory) -> None:
        from hexa.llm import LLMResponseError
        router, mock_llm = router_factory
        mock_llm.chat.side_effect = LLMResponseError("parser failed", status_code=400)
        reply = router.route("hi")
        assert "LLM error" in reply and "400" in reply

    def test_ollama_unavailable_returns_readable_error(self, router_factory) -> None:
        from hexa.llm import LLMConnectionError
        router, mock_llm = router_factory
        mock_llm.chat.side_effect = LLMConnectionError(
            "Cannot reach Ollama at http://localhost:11434"
        )
        reply = router.route("hi")
        assert "LLM unavailable" in reply and "localhost" in reply

    def test_error_mid_tool_loop_can_be_caught(self, router_factory) -> None:
        from hexa.llm import LLMResponseError
        router, mock_llm = router_factory
        mock_llm.chat.side_effect = [
            _resp("", [{"function": {"name": "get_time", "arguments": {}}}]),
            LLMResponseError("kaboom", status_code=500),
        ]
        reply = router.route("time now please")
        assert "LLM error" in reply

    def test_unexpected_error_returns_readable_message(self, router_factory) -> None:
        """LLMError (what llm.py converts unexpected failures into) is readable text."""
        from hexa.llm import LLMError
        router, mock_llm = router_factory
        mock_llm.chat.side_effect = LLMError("something unexpected")
        reply = router.route("hi")
        assert reply.startswith("LLM error:")
        assert "something unexpected" in reply


class TestFastPathStillPreferred:
    def test_fast_path_bypasses_llm(self, router_factory) -> None:
        router, mock_llm = router_factory
        reply = router.route("what time is it")  # exact fast-path match
        mock_llm.chat.assert_not_called()
        assert "10:00" in reply

    def test_stop_never_calls_llm(self, router_factory) -> None:
        router, mock_llm = router_factory
        assert router.route("stop") == "Stopped."
        mock_llm.chat.assert_not_called()

    def test_unknown_app_stays_fast_path(self, router_factory) -> None:
        router, mock_llm = router_factory
        assert "Unknown app" in router.route("open xray123")
        mock_llm.chat.assert_not_called()
