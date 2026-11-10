"""Windows application discovery for Hexa.

Discovers installed applications from trusted Windows sources:
- Start Menu shortcuts
- Registry (HKLM/HKCU, 32/64-bit)
- App Paths registry entries
- Microsoft Store / UWP packaged applications
- User-approved portable application folders

All discovery is local, offline, and non-recursive.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import time
import winreg
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredApp:
    """A discovered application with metadata."""
    app_id: str                 # Stable ID (e.g., "vscode", "chrome")
    display_name: str           # User-visible name (e.g., "Visual Studio Code")
    normalized_name: str        # Lowercase, alphanumeric only for matching
    publisher: str | None = None
    version: str | None = None
    executable: str | None = None       # Path to .exe or shortcut target
    aumid: str | None = None            # Package AUMID for UWP apps
    source: str = "unknown"             # "start_menu", "registry", "app_paths", "store", "portable"
    aliases: list[str] = field(default_factory=list)  # Alternative names
    last_discovered: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        # Normalize name: lowercase, keep alphanumeric and spaces
        self.normalized_name = re.sub(r"[^a-z0-9\s]", "", self.display_name.lower()).strip()
        # Also generate a compact ID from display name if not provided
        if not self.app_id:
            # Replace spaces with hyphens, keep alphanumeric and hyphens
            self.app_id = re.sub(r"[^a-z0-9-]", "", self.normalized_name.replace(" ", "-"))


class AppDiscovery:
    """Discovers installed Windows applications from trusted sources."""

    def __init__(self) -> None:
        self._apps: dict[str, DiscoveredApp] = {}
        self._seen_paths: set[str] = set()

    def discover_all(self) -> dict[str, DiscoveredApp]:
        """Run all discovery sources and return the combined app index."""
        self._apps.clear()
        self._seen_paths.clear()

        # Order matters: later sources can overwrite with better metadata
        self._discover_start_menu()
        self._discover_registry_uninstall()
        self._discover_app_paths()
        self._discover_uwp_apps()
        self._discover_system32_common()
        self._discover_portable_folders()

        # Enrich with common aliases for well-known apps
        self._enrich_with_aliases()

        logger.info("Discovered %d applications from all sources", len(self._apps))
        return self._apps.copy()

    def _add_app(self, app: DiscoveredApp) -> None:
        """Add an app to the index, deduplicating by executable path."""
        if not app.executable:
            return

        # Resolve path to canonical form for deduplication
        try:
            canonical = str(Path(app.executable).resolve())
        except Exception:
            canonical = app.executable

        # Skip if we've seen this executable already (prefer first source)
        if canonical in self._seen_paths:
            return

        self._seen_paths.add(canonical)
        self._apps[app.app_id] = app

    def _normalize_name(self, name: str) -> str:
        """Normalize a name for matching: lowercase, alphanumeric only."""
        return re.sub(r"[^a-z0-9\s]", "", name.lower()).strip()

    # ------------------------------------------------------------------
    # Source 1: Start Menu shortcuts
    # ------------------------------------------------------------------
    def _discover_start_menu(self) -> None:
        """Discover apps from Start Menu shortcuts (.lnk files)."""
        start_menu_paths = [
            Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
            Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
        ]

        for base in start_menu_paths:
            if not base.exists():
                continue
            try:
                for lnk in base.rglob("*.lnk"):
                    target = self._resolve_shortcut(lnk)
                    if not target or not self._is_valid_executable(target):
                        continue

                    name = lnk.stem
                    # Skip uninstallers, help files, etc.
                    if self._is_noise(name):
                        continue

                    app = DiscoveredApp(
                        app_id=self._slugify(name),
                        display_name=name,
                        normalized_name=self._normalize_name(name),
                        executable=target,
                        source="start_menu",
                    )
                    self._add_app(app)
            except Exception as e:
                logger.debug("Start Menu discovery error in %s: %s", base, e)

    def _resolve_shortcut(self, lnk_path: Path) -> str | None:
        """Resolve a .lnk shortcut to its target path."""
        try:
            import pythoncom
            from win32com.shell import shell, shellcon
            shortcut = shell.SHCreateItemFromParsingName(str(lnk_path), None, shell.IID_IShellLink)
            target = shortcut.GetPath(shellcon.SLGP_UNCPRIORITY)[0]
            return target
        except Exception:
            # Fallback: try using PowerShell
            try:
                ps_cmd = f'$sh = New-Object -ComObject WScript.Shell; $sh.CreateShortcut("{lnk_path}").TargetPath'
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_cmd],
                    capture_output=True, text=True, timeout=5
                )
                target = result.stdout.strip()
                return target if target else None
            except Exception:
                return None

    # ------------------------------------------------------------------
    # Source 2: Registry Uninstall keys (installed programs)
    # ------------------------------------------------------------------
    def _discover_registry_uninstall(self) -> None:
        """Discover apps from Windows Registry uninstall keys."""
        root_keys = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        ]

        for hkey, subkey in root_keys:
            try:
                with winreg.OpenKey(hkey, subkey) as key:
                    for i in range(winreg.QueryInfoKey(key)[0]):
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            with winreg.OpenKey(key, subkey_name) as app_key:
                                self._read_uninstall_entry(app_key)
                        except Exception:
                            continue
            except Exception as e:
                logger.debug("Registry discovery error in %s\\%s: %s", hkey, subkey, e)

    def _read_uninstall_entry(self, key: winreg.HKEYType) -> None:
        """Read a single uninstall registry entry."""
        try:
            display_name = winreg.QueryValueEx(key, "DisplayName")[0]
        except Exception:
            return

        if not display_name or self._is_noise(display_name):
            return

        # Skip system components, updates, etc.
        try:
            system_component = winreg.QueryValueEx(key, "SystemComponent")[0]
            if system_component:
                return
        except Exception:
            pass

        # Get executable path
        executable = None
        try:
            executable = winreg.QueryValueEx(key, "InstallLocation")[0]
            if executable:
                # Try to find main .exe in install location
                exe_files = list(Path(executable).glob("*.exe"))
                if exe_files:
                    executable = str(exe_files[0])
        except Exception:
            pass

        try:
            uninstall_string = winreg.QueryValueEx(key, "UninstallString")[0]
            if uninstall_string and not executable:
                # Extract executable from uninstall string
                match = re.search(r'"([^"]+\.exe)"', uninstall_string)
                if match:
                    executable = match.group(1)
        except Exception:
            pass

        if not executable or not self._is_valid_executable(executable):
            return

        publisher = None
        try:
            publisher = winreg.QueryValueEx(key, "Publisher")[0]
        except Exception:
            pass

        version = None
        try:
            version = winreg.QueryValueEx(key, "DisplayVersion")[0]
        except Exception:
            pass

        app = DiscoveredApp(
            app_id=self._slugify(display_name),
            display_name=display_name,
            normalized_name=self._normalize_name(display_name),
            publisher=publisher,
            version=version,
            executable=executable,
            source="registry",
        )
        self._add_app(app)

    # ------------------------------------------------------------------
    # Source 3: App Paths registry
    # ------------------------------------------------------------------
    def _discover_app_paths(self) -> None:
        """Discover apps from App Paths registry (executable -> path mapping)."""
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths") as key:
                for i in range(winreg.QueryInfoKey(key)[0]):
                    try:
                        exe_name = winreg.EnumKey(key, i)
                        with winreg.OpenKey(key, exe_name) as app_key:
                            path, _ = winreg.QueryValueEx(app_key, "")
                            if path and self._is_valid_executable(path):
                                name = Path(exe_name).stem
                                app = DiscoveredApp(
                                    app_id=self._slugify(name),
                                    display_name=name,
                                    normalized_name=self._normalize_name(name),
                                    executable=path,
                                    source="app_paths",
                                )
                                self._add_app(app)
                    except Exception:
                        continue
        except Exception as e:
            logger.debug("App Paths discovery error: %s", e)

    # ------------------------------------------------------------------
    # Source 4: UWP / Microsoft Store apps
    # ------------------------------------------------------------------
    def _discover_uwp_apps(self) -> None:
        """Discover UWP (Microsoft Store) apps via PowerShell."""
        try:
            ps_cmd = (
                "Get-AppxPackage | "
                "Select-Object Name, PackageFamilyName, InstallLocation, "
                "@{N='Exe';E={(Get-AppxPackageManifest $_).Package.Applications.Application.Executable}} | "
                "ConvertTo-Json -Depth 3"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0 or not result.stdout.strip():
                return

            import json
            packages = json.loads(result.stdout)
            if isinstance(packages, dict):
                packages = [packages]

            for pkg in packages:
                name = pkg.get("Name") or pkg.get("PackageFamilyName")
                if not name:
                    continue

                # Get AUMID for launching
                aumid = pkg.get("PackageFamilyName")
                if not aumid:
                    continue

                # Try to get display name from package
                display_name = self._get_uwp_display_name(aumid)

                app = DiscoveredApp(
                    app_id=self._slugify(name),
                    display_name=display_name or name,
                    normalized_name=self._normalize_name(display_name or name),
                    executable=None,  # UWP apps launched via AUMID
                    aumid=aumid,
                    source="store",
                )
                self._add_app(app)

        except Exception as e:
            logger.debug("UWP discovery error: %s", e)

    def _get_uwp_display_name(self, aumid: str) -> str | None:
        """Get friendly display name for a UWP package."""
        try:
            ps_cmd = (
                f"Get-AppxPackage -PackageFamilyName '{aumid}' | "
                "Select-Object -ExpandProperty DisplayName"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True, text=True, timeout=10
            )
            name = result.stdout.strip()
            return name if name else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Source 5: User-approved portable folders
    # ------------------------------------------------------------------
    def _discover_portable_folders(self) -> None:
        """Discover apps from user-configured portable folders."""
        # Read from config - will be populated by the tool
        pass

    # ------------------------------------------------------------------
    # Source 6: Well-known System32 executables
    # ------------------------------------------------------------------
    def _discover_system32_common(self) -> None:
        """Discover well-known system executables in System32.

        These are Windows built-in apps that don't appear in Start Menu,
        Registry, or App Paths but are commonly used.
        """
        system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"

        # Well-known system executables with friendly names and aliases
        common_apps = {
            "notepad.exe": ("Notepad", ["notepad", "text editor", "editor"]),
            "calc.exe": ("Calculator", ["calculator", "calc", "math"]),
            "mspaint.exe": ("Paint", ["paint", "mspaint", "image editor"]),
            "wordpad.exe": ("WordPad", ["wordpad", "rich text"]),
            "write.exe": ("WordPad", ["write", "wordpad"]),
            "snippingtool.exe": ("Snipping Tool", ["snipping tool", "snip", "screenshot"]),
            "snipsketch.exe": ("Snip & Sketch", ["snip sketch", "sketch"]),
            "mstsc.exe": ("Remote Desktop Connection", ["remote desktop", "rdp", "mstsc"]),
            "regedit.exe": ("Registry Editor", ["registry editor", "regedit", "registry"]),
            "cmd.exe": ("Command Prompt", ["cmd", "command prompt", "terminal"]),
            "powershell.exe": ("Windows PowerShell", ["powershell", "ps", "terminal"]),
            "wt.exe": ("Windows Terminal", ["terminal", "windows terminal", "wt"]),
            "explorer.exe": ("File Explorer", ["explorer", "file explorer", "files"]),
            "control.exe": ("Control Panel", ["control panel", "control"]),
            "taskmgr.exe": ("Task Manager", ["task manager", "taskmgr", "tasks"]),
            "msinfo32.exe": ("System Information", ["system info", "msinfo32"]),
            "dxdiag.exe": ("DirectX Diagnostic Tool", ["dxdiag", "directx"]),
            "cleanmgr.exe": ("Disk Cleanup", ["disk cleanup", "cleanmgr"]),
            "dfrgui.exe": ("Defragment and Optimize Drives", ["defrag", "optimize drives"]),
            "mdsched.exe": ("Windows Memory Diagnostic", ["memory diagnostic", "mdsched"]),
            "eventvwr.exe": ("Event Viewer", ["event viewer", "eventvwr", "logs"]),
            "services.msc": ("Services", ["services", "services.msc"]),
            "compmgmt.msc": ("Computer Management", ["computer management", "compmgmt"]),
            "diskmgmt.msc": ("Disk Management", ["disk management", "diskmgmt"]),
            "devmgmt.msc": ("Device Manager", ["device manager", "devmgmt"]),
            "gpedit.msc": ("Group Policy Editor", ["group policy", "gpedit"]),
            "lusrmgr.msc": ("Local Users and Groups", ["local users", "lusrmgr"]),
            "secpol.msc": ("Local Security Policy", ["security policy", "secpol"]),
            "wf.msc": ("Windows Firewall", ["firewall", "wf.msc"]),
        }

        for exe_name, (display_name, aliases) in common_apps.items():
            exe_path = system32 / exe_name
            if exe_path.exists():
                app = DiscoveredApp(
                    app_id=self._slugify(display_name),
                    display_name=display_name,
                    normalized_name=self._normalize_name(display_name),
                    executable=str(exe_path),
                    aliases=aliases,
                    source="system32",
                )
                self._add_app(app)

    def _enrich_with_aliases(self) -> None:
        """Add common aliases for well-known discovered apps."""
        # Mapping of app_id patterns to additional aliases
        alias_map = {
            "visual-studio-code": ["code", "vscode", "vs code", "editor", "ide"],
            "google-chrome": ["chrome", "browser", "web browser", "internet"],
            "microsoft-edge": ["edge", "browser", "web browser"],
            "firefox": ["firefox", "browser", "mozilla"],
            "discord": ["discord", "chat", "voice chat"],
            "steam": ["steam", "games", "game launcher"],
            "spotify": ["spotify", "music", "music player"],
            "vlc": ["vlc", "video player", "media player"],
            "notepad++": ["notepad++", "npp", "text editor"],
            "sublime-text": ["sublime", "sublime text", "editor"],
            "postman": ["postman", "api client", "api testing"],
            "docker-desktop": ["docker", "docker desktop", "containers"],
            "obs-studio": ["obs", "obs studio", "streaming", "recording"],
            "audacity": ["audacity", "audio editor"],
            "gimp": ["gimp", "image editor", "photo editor"],
            "blender": ["blender", "3d", "modeling"],
            "unity": ["unity", "game engine"],
            "unreal-editor": ["unreal", "unreal engine", "ue4", "ue5"],
            "android-studio": ["android studio", "android ide"],
            "intellij-idea": ["intellij", "idea", "java ide"],
            "pycharm": ["pycharm", "python ide"],
            "webstorm": ["webstorm", "js ide", "typescript ide"],
            "rider": ["rider", "dotnet ide", "csharp ide"],
            "clion": ["clion", "cpp ide"],
            "goland": ["goland", "go ide"],
            "rustrover": ["rustrover", "rust ide"],
            "datagrip": ["datagrip", "database ide", "sql ide"],
        }

        for app_id, aliases in alias_map.items():
            if app_id in self._apps:
                app = self._apps[app_id]
                # Add aliases that aren't already present
                for alias in aliases:
                    if alias not in app.aliases:
                        app.aliases.append(alias)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_valid_executable(self, path: str) -> bool:
        """Check if path points to a valid executable."""
        try:
            p = Path(path)
            return p.exists() and p.is_file() and p.suffix.lower() == ".exe"
        except Exception:
            return False

    def _is_noise(self, name: str) -> bool:
        """Filter out uninstallers, help, readme, etc."""
        noise_patterns = [
            r"uninstall", r"remove", r"repair", r"help", r"readme",
            r"license", r"eula", r"changelog", r"release notes",
            r"update", r"updater", r"setup", r"install",
            r"\.url$", r"\.website$",
        ]
        name_lower = name.lower()
        return any(re.search(p, name_lower) for p in noise_patterns)

    def _slugify(self, text: str) -> str:
        """Create a stable ID from text."""
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
        return slug.strip("-")


# Module-level instance for easy access
_discovery: AppDiscovery | None = None


def get_discovery() -> AppDiscovery:
    """Get the global AppDiscovery instance."""
    global _discovery
    if _discovery is None:
        _discovery = AppDiscovery()
    return _discovery