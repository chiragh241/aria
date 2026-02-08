"""Step 2: Communication channels configuration."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.cli.detection import SystemDetector
from src.cli.styles import (
    ERROR,
    ICON_ARROW,
    ICON_CHECK,
    ICON_CROSS,
    ICON_INFO,
    ICON_WARN,
    MUTED,
    PANEL_BORDER,
    PANEL_BOX,
    PRIMARY,
    SUCCESS,
    WARNING,
    step_title,
    status_icon,
)

if TYPE_CHECKING:
    from src.cli.detection import DetectionResults
    from src.cli.wizard import WizardState


def run_step(console: Console, state: WizardState, detection: DetectionResults) -> bool:
    """Run the channels configuration step.

    Returns True if completed, False to go back.
    """
    console.print()
    console.print(
        Panel(
            f"[{PRIMARY}]Select which messaging channels Aria should use.\n"
            f"[{MUTED}]The Web UI is always enabled. Add Slack and/or WhatsApp for mobile access.",
            title=step_title(2, "Channels"),
            border_style=PANEL_BORDER,
            box=PANEL_BOX,
        )
    )

    # Web is always enabled, show that
    console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Web UI [dim](always enabled)[/dim]")
    console.print()

    state.channels_web = True

    # ── Slack ──────────────────────────────────────────────────────────────
    enable_slack = questionary.confirm(
        "Enable Slack channel?",
        default=state.channels_slack,
    ).ask()

    if enable_slack is None:
        return False

    state.channels_slack = enable_slack

    if state.channels_slack:
        if not _configure_slack(console, state, detection):
            return False

    # ── WhatsApp ───────────────────────────────────────────────────────────
    enable_whatsapp = questionary.confirm(
        "Enable WhatsApp channel?",
        default=state.channels_whatsapp,
    ).ask()

    if enable_whatsapp is None:
        return False

    state.channels_whatsapp = enable_whatsapp

    if state.channels_whatsapp:
        if not _configure_whatsapp(console, state, detection):
            return False

    _show_summary(console, state)
    return True


def _configure_slack(
    console: Console, state: WizardState, detection: DetectionResults
) -> bool:
    """Configure Slack integration - collect tokens and validate."""
    console.print()
    console.print(
        Panel(
            f"[{PRIMARY}]Slack Setup[/{PRIMARY}]\n\n"
            f"[{MUTED}]To connect Aria to Slack, you need a Slack app with:\n"
            f"  {ICON_ARROW} Bot Token Scopes: chat:write, app_mentions:read, im:history, channels:history\n"
            f"  {ICON_ARROW} Socket Mode enabled with an App-Level Token\n\n"
            f"  Create a Slack app at: https://api.slack.com/apps",
            border_style=PANEL_BORDER,
            box=PANEL_BOX,
        )
    )

    # Bot token
    bot_token = questionary.password(
        "Slack Bot Token (xoxb-...):",
        default=state.slack_bot_token if state.slack_bot_token else "",
    ).ask()

    if bot_token is None:
        return False

    if not bot_token:
        console.print(f"  [{WARNING}]{ICON_WARN}[/{WARNING}] No bot token provided")
        skip = questionary.confirm(
            "Skip Slack setup for now?",
            default=True,
        ).ask()
        if skip or skip is None:
            state.channels_slack = False
            console.print(f"  [{MUTED}]{ICON_ARROW} Slack disabled")
            return True

    if bot_token and not bot_token.startswith("xoxb-"):
        console.print(f"  [{WARNING}]{ICON_WARN}[/{WARNING}] Token doesn't look like a bot token (expected xoxb-...)")
        proceed = questionary.confirm(
            "Use this token anyway?",
            default=False,
        ).ask()
        if not proceed:
            state.channels_slack = False
            return True

    # App token
    app_token = questionary.password(
        "Slack App-Level Token (xapp-...):",
        default=state.slack_app_token if state.slack_app_token else "",
    ).ask()

    if app_token is None:
        return False

    if not app_token:
        console.print(f"  [{WARNING}]{ICON_WARN}[/{WARNING}] No app token provided")
        skip = questionary.confirm(
            "Skip Slack setup for now?",
            default=True,
        ).ask()
        if skip or skip is None:
            state.channels_slack = False
            console.print(f"  [{MUTED}]{ICON_ARROW} Slack disabled")
            return True

    if app_token and not app_token.startswith("xapp-"):
        console.print(f"  [{WARNING}]{ICON_WARN}[/{WARNING}] Token doesn't look like an app token (expected xapp-...)")

    # Validate credentials
    if bot_token and app_token:
        console.print(f"  [{MUTED}]Validating Slack credentials...")
        valid, info = SystemDetector.check_slack_credentials(bot_token, app_token)
        if valid:
            console.print(
                f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Connected to workspace: {info}"
            )
        else:
            console.print(
                f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] Validation failed: {info}"
            )
            proceed = questionary.confirm(
                "Save these tokens anyway? (you can fix them later in the web UI)",
                default=False,
            ).ask()
            if not proceed:
                state.channels_slack = False
                console.print(f"  [{MUTED}]{ICON_ARROW} Slack disabled")
                return True

    state.slack_bot_token = bot_token
    state.slack_app_token = app_token
    console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Slack configured")
    return True


def _configure_whatsapp(
    console: Console, state: WizardState, detection: DetectionResults
) -> bool:
    """Configure WhatsApp bridge — install deps, start bridge, show QR."""
    console.print()
    console.print(
        Panel(
            f"[{PRIMARY}]WhatsApp Setup[/{PRIMARY}]\n\n"
            f"[{MUTED}]Aria uses a WhatsApp Web bridge (Node.js) to connect.\n"
            f"  We'll set it up now and show you a QR code to scan.",
            border_style=PANEL_BORDER,
            box=PANEL_BOX,
        )
    )

    bridge_dir = Path("whatsapp-bridge")

    # Check Node.js
    if not detection.node.installed:
        console.print(
            f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] Node.js is required but not installed"
        )
        console.print(f"  [{MUTED}]  Install from https://nodejs.org/ then re-run setup")

        skip = questionary.confirm(
            "Skip WhatsApp setup for now?",
            default=True,
        ).ask()

        if skip or skip is None:
            state.channels_whatsapp = False
            return True
    else:
        console.print(
            f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Node.js {detection.node.version} found"
        )

    # Check bridge directory exists
    if not bridge_dir.exists():
        console.print(
            f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] WhatsApp bridge not found at {bridge_dir}"
        )
        state.channels_whatsapp = False
        return True

    # Bridge port
    port = questionary.text(
        "WhatsApp bridge port:",
        default=str(state.whatsapp_bridge_port),
        validate=lambda val: val.isdigit() and 1024 <= int(val) <= 65535,
    ).ask()

    if port is None:
        return False

    state.whatsapp_bridge_port = int(port)

    # Collect allowed phone numbers (allowlist)
    console.print()
    console.print(
        f"  [{MUTED}]{ICON_INFO} For security, only your phone number will be allowed to\n"
        f"  [{MUTED}]  communicate with Aria via WhatsApp. Enter your full number\n"
        f"  [{MUTED}]  with country code (e.g. 919882278774 for +91-9882278774)."
    )
    phone = questionary.text(
        "Your WhatsApp phone number (with country code, e.g. +919882278774):",
        validate=lambda val: len(val.strip().lstrip("+").replace(" ", "").replace("-", "")) >= 7
        and val.strip().lstrip("+").replace(" ", "").replace("-", "").isdigit()
        or "Enter a valid phone number with country code (e.g. +919882278774 or 919882278774)",
    ).ask()

    if phone is None:
        return False

    cleaned = phone.strip().replace(" ", "").replace("-", "").lstrip("+")
    if cleaned:
        state.whatsapp_allowed_numbers = [cleaned]
        console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Only {cleaned} can message Aria on WhatsApp")
    else:
        console.print(f"  [{WARNING}]{ICON_WARN}[/{WARNING}] No number set — all numbers will be allowed")
        state.whatsapp_allowed_numbers = []

    # Install bridge dependencies if needed
    bridge_deps = detection.node.extra.get("bridge_deps_installed", False)
    if not bridge_deps:
        console.print(f"\n  [{MUTED}]{ICON_INFO} Installing WhatsApp bridge dependencies...")
        try:
            result = subprocess.run(
                ["npm", "install"],
                cwd=str(bridge_dir),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Dependencies installed")
            else:
                console.print(
                    f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] npm install failed: {result.stderr[:200]}"
                )
                console.print(f"  [{MUTED}]  You can try manually: cd {bridge_dir} && npm install")
                state.channels_whatsapp = False
                return True
        except subprocess.TimeoutExpired:
            console.print(f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] npm install timed out")
            state.channels_whatsapp = False
            return True
        except Exception as e:
            console.print(f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] Error: {e}")
            state.channels_whatsapp = False
            return True
    else:
        console.print(
            f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Bridge dependencies already installed"
        )

    # Mark as not needing install later (we just did it)
    state.whatsapp_needs_install = False

    # Ask to connect now
    connect_now = questionary.confirm(
        "Connect WhatsApp now? (starts bridge and shows QR code to scan)",
        default=True,
    ).ask()

    if not connect_now:
        console.print(
            f"\n  [{MUTED}]{ICON_INFO} You can connect later — the QR code will appear\n"
            f"  [{MUTED}]  when Aria starts with WhatsApp enabled."
        )
        return True

    # Start the bridge and wait for QR code
    _start_bridge_and_show_qr(console, bridge_dir, state.whatsapp_bridge_port)

    return True


def _start_bridge_and_show_qr(console: Console, bridge_dir: Path, port: int) -> None:
    """Start the WhatsApp bridge, display the QR code, wait for auth."""
    import signal as sig

    console.print(f"\n  [{MUTED}]{ICON_INFO} Starting WhatsApp bridge on port {port}...")

    env = dict(__import__("os").environ)
    env["PORT"] = str(port)

    try:
        proc = subprocess.Popen(
            ["node", "index.js"],
            cwd=str(bridge_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )
    except Exception as e:
        console.print(f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] Failed to start bridge: {e}")
        return

    console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Bridge process started (pid {proc.pid})")
    console.print()

    # Wait for the bridge HTTP server to be ready
    console.print(f"  [{MUTED}]Waiting for bridge to initialize...")
    bridge_ready = False
    for _ in range(30):
        time.sleep(1)
        try:
            import urllib.request
            req = urllib.request.urlopen(f"http://localhost:{port}/status", timeout=2)
            data = json.loads(req.read().decode())
            if data.get("ready"):
                console.print(
                    f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] WhatsApp already authenticated!"
                )
                _stop_bridge(proc)
                return
            bridge_ready = True
            break
        except Exception:
            # Check if process died
            if proc.poll() is not None:
                stderr = proc.stderr.read() if proc.stderr else ""
                console.print(
                    f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] Bridge process exited unexpectedly"
                )
                if stderr:
                    console.print(f"  [{MUTED}]{stderr[:300]}")
                return
            continue

    if not bridge_ready:
        console.print(f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] Bridge did not start in time")
        _stop_bridge(proc)
        return

    # Poll for QR code
    console.print(f"  [{MUTED}]Waiting for QR code...")
    qr_shown = False
    for _ in range(30):
        time.sleep(1)
        try:
            import urllib.request
            req = urllib.request.urlopen(f"http://localhost:{port}/qr", timeout=2)
            data = json.loads(req.read().decode())
            qr_data = data.get("qr")
            if qr_data:
                console.print()
                console.print(
                    Panel(
                        f"[{PRIMARY}]Scan this QR code with your phone:[/{PRIMARY}]\n\n"
                        f"  1. Open WhatsApp on your phone\n"
                        f"  2. Go to Settings {ICON_ARROW} Linked Devices\n"
                        f"  3. Tap 'Link a Device'\n"
                        f"  4. Point your phone camera at the QR code below",
                        title=f"[{PRIMARY}]WhatsApp QR Code[/{PRIMARY}]",
                        border_style=PANEL_BORDER,
                        box=PANEL_BOX,
                    )
                )
                # Use qrcode-terminal style output via the bridge's stdout
                # The bridge already prints the QR to stdout via qrcode-terminal
                # But we also display it from the API data
                try:
                    import qrcode  # type: ignore
                    qr = qrcode.QRCode(border=1)
                    qr.add_data(qr_data)
                    qr.make(fit=True)
                    qr.print_ascii(invert=True)
                except ImportError:
                    # Fallback: print the raw QR data and tell user to look at bridge terminal
                    console.print(
                        f"\n  [{MUTED}]The QR code should be visible in the bridge output above."
                    )
                    console.print(
                        f"  [{MUTED}]If not, open http://localhost:{port}/qr in your browser."
                    )
                qr_shown = True
                break
            msg = data.get("message", "")
            if "authenticated" in msg.lower():
                console.print(
                    f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] WhatsApp already authenticated!"
                )
                _stop_bridge(proc)
                return
        except Exception:
            pass

    if not qr_shown:
        console.print(
            f"  [{WARNING}]{ICON_WARN}[/{WARNING}] Could not retrieve QR code from bridge"
        )
        _stop_bridge(proc)
        return

    # Wait for authentication
    console.print(f"\n  [{MUTED}]Waiting for you to scan the QR code...")
    authenticated = False
    for _ in range(120):  # wait up to 2 minutes
        time.sleep(2)
        try:
            import urllib.request
            req = urllib.request.urlopen(f"http://localhost:{port}/status", timeout=2)
            data = json.loads(req.read().decode())
            if data.get("ready"):
                authenticated = True
                break
        except Exception:
            pass
        # Check if process died
        if proc.poll() is not None:
            break

    if authenticated:
        console.print(
            f"\n  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] WhatsApp connected successfully!"
        )
        console.print(
            f"  [{MUTED}]  The session is saved — you won't need to scan again unless you log out."
        )
    else:
        console.print(
            f"\n  [{WARNING}]{ICON_WARN}[/{WARNING}] WhatsApp not authenticated yet."
        )
        console.print(
            f"  [{MUTED}]  The QR code will appear again when Aria starts."
        )

    _stop_bridge(proc)


def _stop_bridge(proc: subprocess.Popen) -> None:
    """Stop the bridge process gracefully."""
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass


def _show_summary(console: Console, state: WizardState) -> None:
    """Show channel configuration summary."""
    table = Table(
        title="Channel Configuration",
        show_header=True,
        header_style=f"bold {PRIMARY}",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Channel", style=MUTED)
    table.add_column("Status")
    table.add_column("Details", style=MUTED)

    table.add_row("Web UI", f"[{SUCCESS}]Enabled[/{SUCCESS}]", "Always on")

    if state.channels_slack:
        token_status = "tokens saved" if state.slack_bot_token else "tokens needed"
        table.add_row("Slack", f"[{SUCCESS}]Enabled[/{SUCCESS}]", token_status)
    else:
        table.add_row("Slack", f"[{MUTED}]Disabled", "")

    if state.channels_whatsapp:
        port_info = f"bridge port {state.whatsapp_bridge_port}"
        table.add_row("WhatsApp", f"[{SUCCESS}]Enabled[/{SUCCESS}]", port_info)
    else:
        table.add_row("WhatsApp", f"[{MUTED}]Disabled", "")

    console.print()
    console.print(table)
