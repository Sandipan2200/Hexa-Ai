"""Tests for tools and registry."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from hexa.tools import ToolRegistry, tool, get_registry
from hexa.config import Config, AudioConfig, WakeWordConfig, HotkeyConfig, STTConfig, TTSConfig, LLMConfig, MemoryConfig, APIConfig


@pytest.fixture
def registry() -> ToolRegistry:
    """Create a fresh tool registry."""
    reg = ToolRegistry()
    reg.set_permissions({
        "safe_tool": "safe",
        "confirm_tool": "confirm",
        "blocked_tool": "blocked",
    })
    return reg


def test_registry_register(registry: ToolRegistry) -> None:
    """Test tool registration and schema generation."""
    @registry.register
    def greet(name: str, age: int = 25) -> str:
        """Greet a person."""
        return f"Hello {name}, age {age}"

    tool = registry.get_tool("greet")
    assert tool is not None
    assert tool.name == "greet"
    assert tool.description == "Greet a person."
    assert tool.permission == "safe"  # default

    # Check schema
    schema = tool.parameters
    assert schema["type"] == "object"
    assert "name" in schema["properties"]
    assert schema["properties"]["name"]["type"] == "string"
    assert "age" in schema["properties"]
    assert schema["properties"]["age"]["type"] == "integer"
    assert schema["required"] == ["name"]


def test_registry_call_safe(registry: ToolRegistry) -> None:
    """Test calling a safe tool."""
    @registry.register
    def add(a: int, b: int) -> int:
        return a + b

    result = registry.call("add", a=2, b=3)
    assert result == 5


def test_registry_call_blocked(registry: ToolRegistry) -> None:
    """Test calling a blocked tool raises error."""
    @registry.register
    def dangerous() -> str:
        return "boom"

    # The tool should have permission "blocked" from the fixture
    # But we need to set it explicitly
    registry._tools["dangerous"].permission = "blocked"

    with pytest.raises(PermissionError):
        registry.call("dangerous")


def test_registry_call_confirm(registry: ToolRegistry) -> None:
    """Test calling a confirm tool works (auto-confirms in test)."""
    @registry.register
    def confirm_tool() -> str:
        return "confirmed"

    result = registry.call("confirm_tool")
    assert result == "confirmed"


def test_registry_get_schemas_excludes_blocked(registry: ToolRegistry) -> None:
    """Test get_schemas excludes blocked tools."""
    @registry.register
    def safe_tool() -> str:
        return "safe"

    @registry.register
    def blocked_tool() -> str:
        return "blocked"

    schemas = registry.get_schemas()
    names = [s["name"] for s in schemas]
    assert "safe_tool" in names
    assert "blocked_tool" not in names


def test_type_hints_conversion(registry: ToolRegistry) -> None:
    """Test various type hints convert to JSON schema."""
    from typing import Optional, List

    def complex_types(
        required_str: str,
        optional_int: Optional[int] = None,
        list_str: List[str] = None,  # Use None default to avoid mutable default
        flag: bool = False,
    ) -> str:
        return "ok"

    # Manually set annotations since we can't use decorator with forward refs easily
    complex_types.__annotations__ = {
        "required_str": str,
        "optional_int": Optional[int],
        "list_str": List[str],
        "flag": bool,
        "return": str,
    }

    registry.register(complex_types)
    tool = registry.get_tool("complex_types")
    props = tool.parameters["properties"]

    assert props["required_str"]["type"] == "string"
    assert props["optional_int"]["type"] == "integer"
    assert props["list_str"]["type"] == "array"
    assert props["flag"]["type"] == "boolean"


# Tool function tests with mocks

def test_open_app_tool() -> None:
    """Test open_app tool with pre-populated index (no auto-discovery)."""
    from hexa.tools.apps import open_app
    from hexa.config import get_config
    from hexa.tools.app_discovery import DiscoveredApp

    # Mock config
    import hexa.tools.apps
    original_get_config = hexa.tools.apps.get_config

    class MockConfig:
        apps = {"notepad": "notepad.exe", "calc": "calc.exe"}

    hexa.tools.apps.get_config = lambda: MockConfig()

    # Pre-populate the app index to avoid auto-discovery
    from hexa.tools.app_index import get_app_index
    index = get_app_index()
    index.clear()
    index.bulk_upsert({
        "notepad": DiscoveredApp(
            app_id="notepad",
            display_name="Notepad",
            normalized_name="notepad",
            executable=r"C:\Windows\System32\notepad.exe",
            aumid=None,
            source="system32",
            aliases=["notepad", "text editor"],
            last_discovered=0,
        )
    })

    # Mock both Path.exists and subprocess.Popen
    with patch("pathlib.Path.exists", return_value=True), \
         patch("subprocess.Popen") as mock_popen:
        result = open_app("notepad")
        assert "opened notepad" in result.lower()  # Case-insensitive since display name is "Notepad"
        # New implementation uses list args, no shell=True
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        assert args[0][0].endswith("\\notepad.exe")
        assert "shell" not in kwargs or kwargs.get("shell") is False

    # Unknown app
    result = open_app("unknown")
    assert "Unknown app" in result

    hexa.tools.apps.get_config = original_get_config


def test_open_app_auto_discovery() -> None:
    """Test open_app runs discovery automatically when index is empty."""
    from hexa.tools.apps import open_app
    import hexa.tools.apps
    original_get_config = hexa.tools.apps.get_config

    class MockConfig:
        apps = {"notepad": "notepad.exe", "calc": "calc.exe"}

    hexa.tools.apps.get_config = lambda: MockConfig()

    # Clear the app index to simulate fresh start
    from hexa.tools.app_index import get_app_index
    index = get_app_index()
    index.clear()
    assert index.count() == 0

    # Mock discovery to return a proper DiscoveredApp
    from hexa.tools.app_discovery import DiscoveredApp
    with patch("hexa.tools.app_discovery.get_discovery") as mock_get_discovery, \
         patch("pathlib.Path.exists", return_value=True), \
         patch("subprocess.Popen") as mock_popen:

        mock_discovery = MagicMock()
        mock_discovery.discover_all.return_value = {
            "notepad": DiscoveredApp(
                app_id="notepad",
                display_name="Notepad",
                normalized_name="notepad",
                executable=r"C:\Windows\System32\notepad.exe",
                aumid=None,
                source="system32",
                aliases=["notepad", "text editor"],
                last_discovered=0,
            )
        }
        mock_get_discovery.return_value = mock_discovery

        result = open_app("notepad")
        assert "Opened Notepad" in result or "Opened notepad" in result
        mock_popen.assert_called_once()

    hexa.tools.apps.get_config = original_get_config


def test_open_tabs_tool() -> None:
    """Test open_tabs tool."""
    from hexa.tools.browser import open_tabs
    from hexa.config import get_config

    import hexa.tools.browser
    original_get_config = hexa.tools.browser.get_config

    class MockConfig:
        sites = {"youtube": "https://youtube.com", "github": "https://github.com"}

    hexa.tools.browser.get_config = lambda: MockConfig()

    with patch("webbrowser.open_new_tab") as mock_open:
        result = open_tabs(["youtube", "github"])
        assert "Opened 2 tab(s)" in result
        assert mock_open.call_count == 2

    # Unknown sites
    result = open_tabs(["unknown"])
    assert "Unknown site" in result
    assert "Available:" in result

    hexa.tools.browser.get_config = original_get_config


def test_search_tool() -> None:
    """Test search tool."""
    from hexa.tools.browser import search

    import hexa.tools.browser
    original_get_config = hexa.tools.browser.get_config

    class MockConfig:
        search_templates = {
            "google": "https://google.com/search?q={query}",
            "youtube": "https://youtube.com/results?search_query={query}",
        }

    hexa.tools.browser.get_config = lambda: MockConfig()

    with patch("webbrowser.open_new_tab") as mock_open:
        result = search("google", "test query")
        assert "Searching google for 'test query'" in result
        mock_open.assert_called_once()
        # Check URL encoding
        call_args = mock_open.call_args[0][0]
        assert "test+query" in call_args or "test%20query" in call_args

    hexa.tools.browser.get_config = original_get_config


def test_system_tools() -> None:
    """Test system tools."""
    from hexa.tools.system import get_time, system_status

    # get_time returns a formatted string
    result = get_time()
    assert ":" in result  # Has time
    assert "PM" in result or "AM" in result

    # system_status returns CPU/RAM info
    result = system_status()
    assert "CPU:" in result
    assert "RAM:" in result


def test_memory_tools() -> None:
    """Test memory tools."""
    import tempfile
    from hexa.memory import Memory
    from hexa.tools.notes import remember, recall

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/test.db"
        memory = Memory(db_path)

        import hexa.tools.notes
        original_get_memory = hexa.tools.notes.get_memory
        hexa.tools.notes.get_memory = lambda: memory

        # Remember
        result = remember("My exam is on Friday")
        assert "Remembered" in result

        # Recall
        result = recall("exam")
        assert "exam is on Friday" in result

        # Not found
        result = recall("nothing")
        assert "Nothing found" in result

        hexa.tools.notes.get_memory = original_get_memory


def test_timer_tool() -> None:
    """Test set_timer tool."""
    from hexa.tools.timers import set_timer

    # Just verify it returns a confirmation message
    result = set_timer(1, "test")
    assert "Timer set for 1 seconds" in result
    assert "test" in result

    # Invalid
    result = set_timer(0)
    assert "positive" in result.lower()
