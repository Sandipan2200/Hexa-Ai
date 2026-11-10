"""Tests for application discovery and index."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hexa.tools.app_discovery import AppDiscovery, DiscoveredApp, get_discovery
from hexa.tools.app_index import AppIndex, get_app_index


class TestAppDiscovery:
    def test_discovery_initialization(self) -> None:
        """Test AppDiscovery can be instantiated."""
        discovery = AppDiscovery()
        assert discovery is not None

    def test_slugify(self) -> None:
        """Test slugify helper."""
        discovery = AppDiscovery()
        assert discovery._slugify("Visual Studio Code") == "visual-studio-code"
        assert discovery._slugify("Google Chrome") == "google-chrome"
        assert discovery._slugify("Notepad++") == "notepad"

    def test_normalize_name(self) -> None:
        """Test name normalization."""
        discovery = AppDiscovery()
        assert discovery._normalize_name("Visual Studio Code") == "visual studio code"
        assert discovery._normalize_name("Google Chrome") == "google chrome"
        assert discovery._normalize_name("Notepad++") == "notepad"

    def test_is_noise(self) -> None:
        """Test noise filtering."""
        discovery = AppDiscovery()
        assert discovery._is_noise("Uninstall Notepad") is True
        assert discovery._is_noise("Notepad Help") is True
        assert discovery._is_noise("Notepad") is False
        assert discovery._is_noise("Visual Studio Code") is False

    def test_discovered_app_post_init(self) -> None:
        """Test DiscoveredApp auto-generates normalized_name and app_id."""
        app = DiscoveredApp(
            app_id="",
            display_name="Visual Studio Code",
            normalized_name="",  # Will be auto-generated
            executable="/path/to/code.exe",
        )
        assert app.normalized_name == "visual studio code"
        assert app.app_id == "visual-studio-code"


class TestAppIndex:
    @pytest.fixture
    def temp_db(self) -> Path:
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            return Path(f.name)

    @pytest.fixture
    def index(self, temp_db: Path) -> AppIndex:
        """Create an AppIndex with temp database."""
        return AppIndex(temp_db)

    @pytest.fixture
    def sample_apps(self) -> dict[str, DiscoveredApp]:
        """Create sample discovered apps for testing."""
        return {
            "visual-studio-code": DiscoveredApp(
                app_id="visual-studio-code",
                display_name="Visual Studio Code",
                normalized_name="visual studio code",
                executable="/path/to/code.exe",
                aliases=["code", "vscode", "editor"],
                source="start_menu",
            ),
            "google-chrome": DiscoveredApp(
                app_id="google-chrome",
                display_name="Google Chrome",
                normalized_name="google chrome",
                executable="/path/to/chrome.exe",
                aliases=["chrome", "browser"],
                source="start_menu",
            ),
            "notepad": DiscoveredApp(
                app_id="notepad",
                display_name="Notepad",
                normalized_name="notepad",
                executable=r"C:\Windows\System32\notepad.exe",
                aliases=["text editor", "editor"],
                source="system32",
            ),
        }

    def test_upsert_and_get(self, index: AppIndex, sample_apps: dict[str, DiscoveredApp]) -> None:
        """Test upserting and retrieving apps."""
        for app in sample_apps.values():
            index.upsert_app(app)

        retrieved = index.get_app("visual-studio-code")
        assert retrieved is not None
        assert retrieved.display_name == "Visual Studio Code"
        assert retrieved.executable == "/path/to/code.exe"
        assert retrieved.aliases == ["code", "vscode", "editor"]

    def test_bulk_upsert(self, index: AppIndex, sample_apps: dict[str, DiscoveredApp]) -> None:
        """Test bulk upsert."""
        count = index.bulk_upsert(sample_apps)
        assert count == 3
        assert index.count() == 3

    def test_get_all(self, index: AppIndex, sample_apps: dict[str, DiscoveredApp]) -> None:
        """Test getting all apps."""
        index.bulk_upsert(sample_apps)
        all_apps = index.get_all()
        assert len(all_apps) == 3

    def test_get_by_source(self, index: AppIndex, sample_apps: dict[str, DiscoveredApp]) -> None:
        """Test filtering by source."""
        index.bulk_upsert(sample_apps)
        start_menu = index.get_by_source("start_menu")
        assert len(start_menu) == 2
        system32 = index.get_by_source("system32")
        assert len(system32) == 1

    def test_clear(self, index: AppIndex, sample_apps: dict[str, DiscoveredApp]) -> None:
        """Test clearing the index."""
        index.bulk_upsert(sample_apps)
        cleared = index.clear()
        assert cleared == 3
        assert index.count() == 0

    def test_find_best_match_exact_id(self, index: AppIndex, sample_apps: dict[str, DiscoveredApp]) -> None:
        """Test exact app_id match."""
        index.bulk_upsert(sample_apps)
        matches = index.find_best_match("visual-studio-code")
        assert len(matches) == 1
        assert matches[0].app_id == "visual-studio-code"

    def test_find_best_match_exact_display(self, index: AppIndex, sample_apps: dict[str, DiscoveredApp]) -> None:
        """Test exact display_name match."""
        index.bulk_upsert(sample_apps)
        matches = index.find_best_match("Visual Studio Code")
        assert len(matches) == 1
        assert matches[0].app_id == "visual-studio-code"

    def test_find_best_match_normalized(self, index: AppIndex, sample_apps: dict[str, DiscoveredApp]) -> None:
        """Test normalized name match."""
        index.bulk_upsert(sample_apps)
        matches = index.find_best_match("visual studio code")
        assert len(matches) == 1
        assert matches[0].app_id == "visual-studio-code"

    def test_find_best_match_alias(self, index: AppIndex, sample_apps: dict[str, DiscoveredApp]) -> None:
        """Test alias match."""
        index.bulk_upsert(sample_apps)
        matches = index.find_best_match("code")
        assert len(matches) == 1
        assert matches[0].app_id == "visual-studio-code"

        matches = index.find_best_match("chrome")
        assert len(matches) == 1
        assert matches[0].app_id == "google-chrome"

    def test_find_best_match_prefix(self, index: AppIndex, sample_apps: dict[str, DiscoveredApp]) -> None:
        """Test prefix match."""
        index.bulk_upsert(sample_apps)
        matches = index.find_best_match("visual")
        assert len(matches) >= 1
        assert matches[0].app_id == "visual-studio-code"

    def test_find_best_match_fuzzy(self, index: AppIndex, sample_apps: dict[str, DiscoveredApp]) -> None:
        """Test fuzzy match."""
        index.bulk_upsert(sample_apps)
        matches = index.find_best_match("vizual studio")  # typo
        assert len(matches) >= 1
        assert matches[0].app_id == "visual-studio-code"

    def test_find_best_match_no_match(self, index: AppIndex, sample_apps: dict[str, DiscoveredApp]) -> None:
        """Test no match returns empty list."""
        index.bulk_upsert(sample_apps)
        matches = index.find_best_match("nonexistentapp")
        assert matches == []

    def test_resolve_ambiguous(self, index: AppIndex, sample_apps: dict[str, DiscoveredApp]) -> None:
        """Test ambiguous match resolution."""
        index.bulk_upsert(sample_apps)
        # "visual" could match "visual studio code"
        matches = index.resolve_ambiguous("visual", max_results=3)
        assert len(matches) >= 1
        assert matches[0].app_id == "visual-studio-code"

    def test_preferred_apps(self, index: AppIndex, sample_apps: dict[str, DiscoveredApp]) -> None:
        """Test setting and getting preferred apps."""
        index.bulk_upsert(sample_apps)

        # Set preferred
        assert index.set_preferred("browser", "google-chrome") is True
        assert index.set_preferred("editor", "visual-studio-code") is True

        # Get preferred
        browser = index.get_preferred("browser")
        assert browser is not None
        assert browser.app_id == "google-chrome"

        editor = index.get_preferred("editor")
        assert editor is not None
        assert editor.app_id == "visual-studio-code"

        # Invalid app_id
        assert index.set_preferred("browser", "nonexistent") is False

        # Get all preferred
        all_prefs = index.get_all_preferred()
        assert "browser" in all_prefs
        assert "editor" in all_prefs

    def test_indexed_app_hashable(self, index: AppIndex, sample_apps: dict[str, DiscoveredApp]) -> None:
        """Test IndexedApp is hashable (can be used in sets)."""
        index.bulk_upsert(sample_apps)
        apps = index.get_all()
        app_set = set(apps)
        assert len(app_set) == 3


class TestDiscoveryIntegration:
    """Integration tests using the real discovery (may be slow)."""

    def test_discovery_runs_without_error(self) -> None:
        """Test that discovery runs and returns apps."""
        discovery = AppDiscovery()
        apps = discovery.discover_all()
        # Should find at least some apps on any Windows system
        assert isinstance(apps, dict)
        # On a real system, should find > 0 apps
        # But in CI might be 0, so just check it runs

    def test_system32_discovery(self) -> None:
        """Test system32 common apps are discovered."""
        discovery = AppDiscovery()
        discovery._discover_system32_common()
        apps = discovery._apps

        # Should find notepad, calc, etc.
        assert "notepad" in apps
        assert apps["notepad"].display_name == "Notepad"
        assert apps["notepad"].source == "system32"

        assert "calculator" in apps
        assert apps["calculator"].display_name == "Calculator"


class TestGlobalInstances:
    """Test global instance getters."""

    def test_get_discovery(self) -> None:
        """Test get_discovery returns singleton."""
        d1 = get_discovery()
        d2 = get_discovery()
        assert d1 is d2

    def test_get_app_index(self) -> None:
        """Test get_app_index returns singleton."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        try:
            i1 = get_app_index(db_path)
            i2 = get_app_index(db_path)
            assert i1 is i2
        finally:
            db_path.unlink(missing_ok=True)