"""Unit tests for the LLM client: tool wire format, errors, history.

Mocks `ollama.Client` (external service) — the client-side logic under test
is real: envelope construction, exception mapping, message normalization.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import ollama
import pytest

import hexa.llm as llm_module  # noqa: F401  (module import under test)
from hexa.llm import (
    LLMClient,
    LLMConnectionError,
    LLMError,
    LLMResponseError,
    _wire_tools,
    extract_message,
)


@pytest.fixture
def client() -> LLMClient:
    """LLMClient with a mocked ollama.Client and isolated config."""
    with patch("hexa.llm.ollama.Client") as mock_client_cls, \
         patch("hexa.llm.get_config") as mock_cfg:
        mock_cfg.return_value = MagicMock(
            llm=MagicMock(
                model="hexa-brain",
                host="http://localhost:11434",
                num_ctx=3072,
                temperature=0.2,
                keep_alive=-1,
                max_history=6,
                system_prompt="You are Hexa.",
            )
        )
        instance = LLMClient()
    instance.client = MagicMock()
    return instance


def _resp(content: str = "", tool_calls: list | None = None) -> SimpleNamespace:
    """Build a fake ollama ChatResponse-shaped object."""
    msg = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(message=msg)


def _tcall(name: str, args: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=args or {})
    )


class TestWireTools:
    def test_plain_tools_none(self) -> None:
        assert _wire_tools(None) is None
        assert _wire_tools([]) is None

    def test_openai_envelope(self) -> None:
        schemas = [{"name": "get_time", "description": "d", "parameters": {"type": "object", "properties": {}}}]
        wire = _wire_tools(schemas)
        assert wire == [{"type": "function", "function": schemas[0]}]


class TestExtractMessage:
    def test_object_response_plain_text(self) -> None:
        msg = extract_message(_resp("Hello"))
        assert msg["content"] == "Hello"
        assert msg["tool_calls"] == []
        assert msg["role"] == "assistant"

    def test_object_response_tool_calls_normalized(self) -> None:
        fake = _resp("", [_tcall("open_app", {"name": "notepad"})])
        msg = extract_message(fake)
        assert msg["tool_calls"] == [{"function": {"name": "open_app", "arguments": {"name": "notepad"}}}]

    def test_dict_response(self) -> None:
        fake = {"message": {"role": "assistant", "content": "hi", "tool_calls": []}}
        msg = extract_message(fake)
        assert msg["content"] == "hi"

    def test_none_message_is_safe(self) -> None:
        msg = extract_message(SimpleNamespace(message=None))
        assert msg == {"role": "assistant", "content": "", "tool_calls": []}

    def test_non_dict_arguments_falls_back(self) -> None:
        fake = _resp("", [_tcall("open_app", "bad")])
        msg = extract_message(fake)
        assert msg["tool_calls"][0]["function"]["arguments"] == {}

    def test_unnamed_tool_call_dropped(self) -> None:
        fake = _resp("", [SimpleNamespace(function=SimpleNamespace(name=None, arguments={}))])
        msg = extract_message(fake)
        assert msg["tool_calls"] == []


class TestChat:
    def test_plain_chat_without_tools(self, client: LLMClient) -> None:
        client.client.chat.return_value = _resp("Hello!")
        out = client.chat("hi")  # tool_schemas=None
        assert client.client.chat.call_args.kwargs["tools"] is None
        # messages contain exactly one user turn
        messages = client.client.chat.call_args.kwargs["messages"]
        assert messages[-1] == {"role": "user", "content": "hi"}

    def test_chat_wraps_tools_in_envelope(self, client: LLMClient) -> None:
        client.client.chat.return_value = _resp("Hello!")
        schemas = [{"name": "get_time", "description": "d", "parameters": {"type": "object", "properties": {}}}]
        client.chat("hi", schemas)
        sent_tools = client.client.chat.call_args.kwargs["tools"]
        assert sent_tools[0]["type"] == "function"
        assert sent_tools[0]["function"]["name"] == "get_time"

    def test_keep_alive_is_toplevel_kwarg(self, client: LLMClient) -> None:
        """keep_alive (=-1) must be its own kwarg, never inside options."""
        client.client.chat.return_value = _resp("Hello!")
        client.chat("hi")
        kwargs = client.client.chat.call_args.kwargs
        assert kwargs["keep_alive"] == -1
        assert "keep_alive" not in kwargs["options"]

    def test_http_400_maps_to_response_error(self, client: LLMClient) -> None:
        err = ollama.ResponseError("template parser failed", 400)
        client.client.chat.side_effect = err
        with pytest.raises(LLMResponseError) as exc_info:
            client.chat("hi")
        assert exc_info.value.status_code == 400

    def test_ollama_unavailable_maps_to_connection_error(self, client: LLMClient) -> None:
        client.client.chat.side_effect = httpx.ConnectError("connection refused")
        with pytest.raises(LLMConnectionError):
            client.chat("hi")

    def test_timeout_maps_to_connection_error(self, client: LLMClient) -> None:
        client.client.chat.side_effect = httpx.ConnectTimeout("timed out")
        with pytest.raises(LLMConnectionError):
            client.chat("hi")

    def test_unexpected_error_maps_to_base_llm_error(self, client: LLMClient) -> None:
        client.client.chat.side_effect = RuntimeError("boom")
        with pytest.raises(LLMError) as exc_info:
            client.chat("hi")
        assert not isinstance(exc_info.value, (LLMConnectionError, LLMResponseError))


class TestHistory:
    def test_add_and_trim(self, client: LLMClient) -> None:
        for i in range(20):
            client.add_to_history("user", f"msg{i}")
        # max_history=6 -> 12 entries kept
        assert len(client._history) == 12
        assert client._history[-1]["content"] == "msg19"

    def test_clear(self, client: LLMClient) -> None:
        client.add_to_history("user", "x")
        client.clear_history()
        assert client._history == []

    def test_tool_round_history_sequence(self, client: LLMClient) -> None:
        """Simulates the native tool-calling round: user -> assistant(tool_calls) -> tool -> next."""
        client.client.chat.return_value = _resp("final answer")
        schemas = [{"name": "get_time", "description": "d", "parameters": {"type": "object", "properties": {}}}]

        response = _resp("", [_tcall("get_time")])
        client.client.chat.side_effect = [response, _resp("The time is 10:00.")]

        # Mimic router flow
        client.chat("what time is it", schemas)
        client.add_to_history("user", "what time is it")
        client.add_to_history("assistant", "", [{"function": {"name": "get_time", "arguments": {}}}])
        client.add_to_history("tool", "It's 10:00.")
        client.chat("", schemas)

        second_call_messages = client.client.chat.call_args_list[1].kwargs["messages"]
        roles = [m["role"] for m in second_call_messages]
        assert roles == ["system", "system", "user", "assistant", "tool"]
        # An empty user turn must NOT be appended on follow-up calls
        assert second_call_messages[-1]["role"] == "tool"

    def test_build_messages_skips_empty_user_text(self, client: LLMClient) -> None:
        client.add_to_history("tool", "result")
        messages = client._build_messages("", None)
        assert messages[-1]["role"] == "tool"
