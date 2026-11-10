"""App launching tools with automatic discovery.

Supports:
- Legacy whitelist from config.yaml (backward compatible)
- Discovered application index (auto-discovered from Windows)
- Intelligent matching with aliases and fuzzy fallback
- Safe launching (no shell=True, validates executables)
- UWP / Microsoft Store app support
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from hexa.tools import tool
from hexa.config import get_config
from hexa.tools.app_index import get_app_index, IndexedApp

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _resolve_app(name: str) -> IndexedApp | None:
    """Resolve an app name to an IndexedApp using the discovery index.

    Falls back to legacy config.yaml whitelist if not in index.
    """
    index = get_app_index()

    # Try intelligent matching on the index
    matches = index.find_best_match(name)

    # If exactly one match, use it
    if len(matches) == 1:
        return matches[0]

    # If multiple matches, check for exact app_id or display_name match
    name_lower = name.lower().strip()
    for app in matches:
        if app.app_id.lower() == name_lower or app.display_name.lower() == name_lower:
            return app

    # If still ambiguous, return None (caller should ask for disambiguation)
    if len(matches) > 1:
        logger.info("Ambiguous app request '%s': %s", name, [a.display_name for a in matches])
        return None

    # Fallback: check legacy config.yaml whitelist
    config = get_config()
    if name_lower in config.apps:
        # Create a synthetic IndexedApp for legacy entries
        cmd = config.apps[name_lower]
        return IndexedApp(
            app_id=name_lower,
            display_name=name_lower,
            normalized_name=name_lower,
            publisher=None,
            version=None,
            executable=cmd if cmd.endswith(".exe") else None,
            aumid=None,
            source="config",
            aliases=[],
            last_discovered=0,
        )

    return None


def _launch_app(app: IndexedApp) -> str:
    """Launch an application safely (no shell=True)."""
    cmd = app.launch_command()
    if not cmd:
        return f"Cannot launch {app.display_name}: no executable or AUMID"

    # Validate executable exists (for non-UWP apps)
    if app.executable and not Path(app.executable).exists():
        return f"Executable not found: {app.executable}"

    try:
        logger.info("Launching app: %s (%s)", app.display_name, cmd)
        # Use Popen with list args, no shell=True
        subprocess.Popen(cmd)
        return f"Opened {app.display_name}"
    except FileNotFoundError:
        logger.exception("Executable not found: %s", cmd[0])
        return f"Failed to open {app.display_name}: executable not found"
    except PermissionError:
        logger.exception("Permission denied: %s", cmd[0])
        return f"Failed to open {app.display_name}: permission denied"
    except Exception as e:
        logger.exception("Failed to launch app: %s", app.display_name)
        return f"Failed to open {app.display_name}: {e}"


# ------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------
@tool
def open_app(name: str) -> str:
    """Open an application by name.

    Uses the discovered application index with intelligent matching:
    - Exact name, normalized name, or alias match
    - Fuzzy fallback for typos
    - Falls back to legacy config.yaml whitelist

    If multiple apps match closely, returns a disambiguation message.
    """
    app = _resolve_app(name)

    if app is None:
        # Try to find close matches for helpful error
        index = get_app_index()
        suggestions = index.resolve_ambiguous(name, max_results=5)
        if suggestions:
            names = ", ".join(f'"{a.display_name}"' for a in suggestions)
            return f"Ambiguous: '{name}' could be {names}. Please be more specific."
        return f"Unknown app: {name}. Try 'refresh apps' to discover installed applications."

    return _launch_app(app)


@tool
def refresh_apps(force: bool = False) -> str:
    """Refresh the application index by re-scanning Windows sources.

    Discovers from: Start Menu, Registry, App Paths, Microsoft Store.
    Runs in background; may take 5-15 seconds.

    Args:
        force: If True, clears the index before re-discovering.
    """
    from hexa.tools.app_discovery import get_discovery

    index = get_app_index()
    discovery = get_discovery()

    if force:
        cleared = index.clear()
        logger.info("Cleared %d apps from index (force refresh)", cleared)

    # Run discovery
    discovered = discovery.discover_all()

    if not discovered:
        return "No applications discovered. Check Start Menu and Registry permissions."

    # Persist to index
    count = index.bulk_upsert(discovered)

    return f"Discovered and indexed {count} applications from {len(set(a.source for a in discovered.values()))} sources."


@tool
def list_apps(source: str | None = None, limit: int = 50) -> str:
    """List discovered applications.

    Args:
        source: Filter by discovery source (start_menu, registry, app_paths, store, config)
        limit: Maximum number of apps to return
    """
    index = get_app_index()

    if source:
        apps = index.get_by_source(source)
    else:
        apps = index.get_all(limit=limit)

    if not apps:
        return "No applications indexed. Try 'refresh apps' first."

    lines = [f"Discovered applications ({len(apps)}):"]
    for app in apps[:limit]:
        src = f"[{app.source}]"
        exe = f" -> {app.executable}" if app.executable else f" (UWP: {app.aumid})"
        aliases = f" (aliases: {', '.join(app.aliases)})" if app.aliases else ""
        lines.append(f"  {app.display_name} {src}{exe}{aliases}")

    return "\n".join(lines)


@tool
def set_preferred_app(category: str, app_name: str) -> str:
    """Set a preferred application for a generic category.

    Categories: browser, editor, terminal, music, video, pdf, image, email

    Example: set_preferred_app("browser", "chrome")
    """
    index = get_app_index()

    # Resolve the app name
    app = _resolve_app(app_name)
    if not app:
        return f"Unknown app: {app_name}"

    valid_categories = {"browser", "editor", "terminal", "music", "video", "pdf", "image", "email"}
    if category not in valid_categories:
        return f"Invalid category: {category}. Valid: {', '.join(sorted(valid_categories))}"

    if index.set_preferred(category, app.app_id):
        return f"Set {app.display_name} as preferred {category}."
    return f"Failed to set preferred {category}."


@tool
def get_preferred_apps() -> str:
    """Show all preferred applications for generic categories."""
    index = get_app_index()
    prefs = index.get_all_preferred()

    if not prefs:
        return "No preferred apps set. Use 'set preferred app' to configure."

    lines = ["Preferred applications:"]
    for category, app in sorted(prefs.items()):
        lines.append(f"  {category}: {app.display_name}")

    return "\n".join(lines)