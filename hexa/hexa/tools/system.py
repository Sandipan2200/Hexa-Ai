"""System info tools."""
from __future__ import annotations

import logging
import time
from hexa.tools import tool

logger = logging.getLogger(__name__)


@tool
def get_time() -> str:
    """Get the current date and time."""
    now = time.localtime()
    return time.strftime("It's %I:%M %p on %A, %B %d, %Y", now)


@tool
def system_status() -> str:
    """Get CPU, RAM, and battery status."""
    try:
        import psutil
    except ImportError:
        return "psutil not available"

    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory()
    ram_percent = ram.percent
    ram_used_gb = ram.used / (1024**3)
    ram_total_gb = ram.total / (1024**3)

    battery_info = ""
    battery = psutil.sensors_battery()
    if battery:
        battery_info = f", Battery: {battery.percent:.0f}%{' (charging)' if battery.power_plugged else ''}"

    return f"CPU: {cpu:.0f}%, RAM: {ram_percent:.0f}% ({ram_used_gb:.1f}/{ram_total_gb:.1f} GB){battery_info}"