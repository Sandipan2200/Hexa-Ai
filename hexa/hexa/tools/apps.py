"""App launching tools."""
from __future__ import annotations

import logging
import subprocess
from hexa.tools import tool
from hexa.config import get_config

logger = logging.getLogger(__name__)


@tool
def open_app(name: str) -> str:
    """Open an application from the whitelist. Unknown names are rejected."""
    config = get_config()
    apps = config.apps

    name_lower = name.lower().strip()
    if name_lower not in apps:
        available = ", ".join(sorted(apps.keys()))
        logger.warning("Unknown app requested: %s (available: %s)", name, available)
        return f"Unknown app: {name}. Available: {available}"

    cmd = apps[name_lower]
    try:
        logger.info("Opening app: %s (%s)", name, cmd)
        subprocess.Popen(cmd, shell=True)
        return f"Opened {name}"
    except Exception as e:
        logger.exception("Failed to open app: %s", name)
        return f"Failed to open {name}: {e}"