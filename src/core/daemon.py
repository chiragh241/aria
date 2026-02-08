"""Daemon / service mode — run Aria as a background system service.

Supports:
  - macOS: launchd plist
  - Linux: systemd unit file
  - Generic: PID file management for any Unix

CLI usage:
  aria --daemon install   # Install as system service
  aria --daemon start     # Start the service
  aria --daemon stop      # Stop the service
  aria --daemon status    # Check if running
  aria --daemon uninstall # Remove the service
"""

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)

PID_FILE = Path("data/aria.pid")
SERVICE_NAME = "com.aria.assistant"


def get_platform() -> str:
    """Detect the service manager platform."""
    if sys.platform == "darwin":
        return "launchd"
    elif sys.platform.startswith("linux"):
        # Check if systemd is available
        if Path("/run/systemd/system").exists():
            return "systemd"
        return "generic"
    return "generic"


# --- PID file management ---

def write_pid() -> None:
    """Write current process PID to file."""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def read_pid() -> int | None:
    """Read PID from file, or None if not running."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        # Check if process is still alive
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        PID_FILE.unlink(missing_ok=True)
        return None


def remove_pid() -> None:
    """Remove PID file."""
    PID_FILE.unlink(missing_ok=True)


# --- Service status ---

def is_running() -> bool:
    """Check if Aria daemon is running."""
    return read_pid() is not None


def get_status() -> dict[str, Any]:
    """Get daemon status info."""
    pid = read_pid()
    platform = get_platform()
    installed = _is_installed(platform)
    return {
        "running": pid is not None,
        "pid": pid,
        "platform": platform,
        "installed": installed,
        "pid_file": str(PID_FILE),
    }


# --- Service installation ---

def _get_aria_command() -> str:
    """Get the command to run Aria."""
    # If installed as a package, use the entry point
    if Path(sys.prefix, "bin", "aria").exists():
        return str(Path(sys.prefix, "bin", "aria"))
    # Otherwise use python -m
    return f"{sys.executable} -m src.main"


def _get_project_dir() -> str:
    """Get the project root directory."""
    return str(Path(__file__).resolve().parent.parent.parent)


def _is_installed(platform: str) -> bool:
    """Check if the service is installed."""
    if platform == "launchd":
        plist = Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_NAME}.plist"
        return plist.exists()
    elif platform == "systemd":
        unit = Path.home() / ".config" / "systemd" / "user" / "aria.service"
        return unit.exists()
    return False


def install_service() -> dict[str, Any]:
    """Install Aria as a system service."""
    platform = get_platform()

    if platform == "launchd":
        return _install_launchd()
    elif platform == "systemd":
        return _install_systemd()
    else:
        return {"success": False, "error": "Unsupported platform. Use PID-based management."}


def _install_launchd() -> dict[str, Any]:
    """Install as a macOS launchd service (user agent)."""
    aria_cmd = _get_aria_command()
    project_dir = _get_project_dir()
    log_dir = Path(project_dir) / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{SERVICE_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{aria_cmd.split()[0]}</string>
        {"".join(f"<string>{a}</string>" for a in aria_cmd.split()[1:])}
        <string>--skip-setup</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{project_dir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir}/aria-daemon.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/aria-daemon.err</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin:{Path(sys.executable).parent}</string>
    </dict>
</dict>
</plist>"""

    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_NAME}.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist_content)

    logger.info("Installed launchd service", path=str(plist_path))
    return {
        "success": True,
        "platform": "launchd",
        "path": str(plist_path),
        "message": f"Installed. Start with: launchctl load {plist_path}",
    }


def _install_systemd() -> dict[str, Any]:
    """Install as a systemd user service."""
    aria_cmd = _get_aria_command()
    project_dir = _get_project_dir()

    unit_content = f"""[Unit]
Description=Aria Personal AI Assistant
After=network.target

[Service]
Type=simple
WorkingDirectory={project_dir}
ExecStart={aria_cmd} --skip-setup
Restart=on-failure
RestartSec=10
Environment=PATH=/usr/local/bin:/usr/bin:/bin:{Path(sys.executable).parent}

[Install]
WantedBy=default.target
"""

    unit_path = Path.home() / ".config" / "systemd" / "user" / "aria.service"
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(unit_content)

    # Reload systemd
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)

    logger.info("Installed systemd service", path=str(unit_path))
    return {
        "success": True,
        "platform": "systemd",
        "path": str(unit_path),
        "message": "Installed. Start with: systemctl --user start aria",
    }


def uninstall_service() -> dict[str, Any]:
    """Uninstall the system service."""
    platform = get_platform()

    if platform == "launchd":
        plist = Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_NAME}.plist"
        # Unload first
        subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
        plist.unlink(missing_ok=True)
        return {"success": True, "message": "Launchd service removed."}
    elif platform == "systemd":
        subprocess.run(["systemctl", "--user", "stop", "aria"], capture_output=True)
        subprocess.run(["systemctl", "--user", "disable", "aria"], capture_output=True)
        unit = Path.home() / ".config" / "systemd" / "user" / "aria.service"
        unit.unlink(missing_ok=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        return {"success": True, "message": "Systemd service removed."}

    return {"success": False, "error": "No service installed."}


def start_service() -> dict[str, Any]:
    """Start the daemon service."""
    platform = get_platform()

    if platform == "launchd":
        plist = Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_NAME}.plist"
        if not plist.exists():
            return {"success": False, "error": "Service not installed. Run: aria --daemon install"}
        result = subprocess.run(["launchctl", "load", str(plist)], capture_output=True, text=True)
        if result.returncode == 0:
            return {"success": True, "message": "Aria daemon started via launchd."}
        return {"success": False, "error": result.stderr.strip()}
    elif platform == "systemd":
        result = subprocess.run(
            ["systemctl", "--user", "start", "aria"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return {"success": True, "message": "Aria daemon started via systemd."}
        return {"success": False, "error": result.stderr.strip()}

    return {"success": False, "error": "Unsupported platform for service management."}


def stop_service() -> dict[str, Any]:
    """Stop the daemon service."""
    # Try PID file first
    pid = read_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            remove_pid()
            return {"success": True, "message": f"Sent SIGTERM to PID {pid}."}
        except ProcessLookupError:
            remove_pid()

    platform = get_platform()
    if platform == "launchd":
        plist = Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_NAME}.plist"
        subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
        return {"success": True, "message": "Aria daemon stopped via launchd."}
    elif platform == "systemd":
        subprocess.run(["systemctl", "--user", "stop", "aria"], capture_output=True)
        return {"success": True, "message": "Aria daemon stopped via systemd."}

    return {"success": False, "error": "Daemon not running."}


def handle_daemon_command(action: str) -> None:
    """Handle --daemon CLI sub-commands."""
    actions = {
        "install": install_service,
        "uninstall": uninstall_service,
        "start": start_service,
        "stop": stop_service,
        "status": lambda: get_status(),
    }

    func = actions.get(action)
    if not func:
        print(f"Unknown daemon action: {action}")
        print("Usage: aria --daemon [install|uninstall|start|stop|status]")
        sys.exit(1)

    result = func()
    if isinstance(result, dict):
        if result.get("success") is False:
            print(f"Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)
        for key, value in result.items():
            if key not in ("success",):
                print(f"  {key}: {value}")
    sys.exit(0)
