"""Step 4: Dashboard port and credentials configuration."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.cli.detection import SystemDetector
from src.cli.styles import (
    ICON_CHECK,
    ICON_INFO,
    ICON_WARN,
    MUTED,
    PANEL_BORDER,
    PANEL_BOX,
    PRIMARY,
    SUCCESS,
    WARNING,
    step_title,
)

if TYPE_CHECKING:
    from src.cli.detection import DetectionResults
    from src.cli.wizard import WizardState


def run_step(console: Console, state: WizardState, detection: DetectionResults) -> bool:
    """Run the dashboard configuration step.

    Returns True if completed, False to go back.
    """
    console.print()
    console.print(
        Panel(
            f"[{PRIMARY}]Configure the web dashboard that lets you chat with Aria,\n"
            f"manage approvals, and monitor activity.\n"
            f"[{MUTED}]Accessible at http://localhost:<port> after startup.",
            title=step_title(4, "Dashboard"),
            border_style=PANEL_BORDER,
            box=PANEL_BOX,
        )
    )

    # Port selection
    default_port = "8080"
    port_str = questionary.text(
        "Dashboard port:",
        default=default_port,
        validate=lambda val: _validate_port(val),
    ).ask()

    if port_str is None:
        return False

    port = int(port_str)

    # Check port availability
    if not SystemDetector.check_port_available(port):
        console.print(
            f"  [{WARNING}]{ICON_WARN}[/{WARNING}] Port {port} is currently in use."
        )
        use_anyway = questionary.confirm(
            "Use this port anyway? (The current process may need to be stopped first)",
            default=False,
        ).ask()
        if not use_anyway:
            alt_port = questionary.text(
                "Choose a different port:",
                default="8081",
                validate=lambda val: _validate_port(val),
            ).ask()
            if alt_port is None:
                return False
            port = int(alt_port)

    state.dashboard_port = port
    console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Dashboard will run on port {port}")

    # JWT secret
    console.print(f"\n  [{MUTED}]{ICON_INFO} Generating secure JWT secret...")
    state.jwt_secret = secrets.token_urlsafe(32)
    console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] JWT secret generated")

    # Admin password
    admin_password = questionary.password(
        "Set admin password for the dashboard:",
        default="",
    ).ask()

    if admin_password is None:
        return False

    if not admin_password:
        admin_password = secrets.token_urlsafe(12)
        console.print(
            f"  [{MUTED}]{ICON_INFO} Auto-generated password: {admin_password}"
        )
        console.print(
            f"  [{MUTED}]  (This will be saved in your .env file)"
        )

    state.admin_password = admin_password

    _show_summary(console, state)
    return True


def _validate_port(val: str) -> bool | str:
    """Validate port number input."""
    if not val.isdigit():
        return "Port must be a number"
    port = int(val)
    if port < 1024:
        return "Port must be >= 1024 (non-privileged)"
    if port > 65535:
        return "Port must be <= 65535"
    return True


def _show_summary(console: Console, state: WizardState) -> None:
    """Show dashboard configuration summary."""
    table = Table(
        title="Dashboard Configuration",
        show_header=True,
        header_style=f"bold {PRIMARY}",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Setting", style=MUTED)
    table.add_column("Value")

    table.add_row("Port", str(state.dashboard_port))
    table.add_row("URL", f"http://localhost:{state.dashboard_port}")
    table.add_row("JWT secret", "[dim]generated (saved to .env)[/dim]")
    table.add_row("Admin password", "[dim]set (saved to .env)[/dim]")

    console.print()
    console.print(table)
