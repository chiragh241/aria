"""Step 5: Security profile selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.cli.styles import (
    ERROR,
    ICON_SHIELD,
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


# Security profile descriptions
PROFILES = {
    "paranoid": {
        "label": "Paranoid",
        "desc": "Maximum security - every action requires your approval",
        "read_files": "Approve",
        "write_files": "Approve",
        "delete_files": "Approve",
        "shell_cmds": "Approve",
        "messages": "Approve",
        "web": "Approve",
        "calendar": "Approve",
        "skills": "Approve",
    },
    "balanced": {
        "label": "Balanced (Recommended)",
        "desc": "Safe defaults with approval for destructive actions",
        "read_files": "Auto*",
        "write_files": "Notify",
        "delete_files": "Approve",
        "shell_cmds": "Approve",
        "messages": "Approve",
        "web": "Auto*",
        "calendar": "Auto/Notify",
        "skills": "Approve",
    },
    "trusted": {
        "label": "Trusted",
        "desc": "Minimal friction - only blocks dangerous operations",
        "read_files": "Auto",
        "write_files": "Auto",
        "delete_files": "Notify",
        "shell_cmds": "Notify",
        "messages": "Notify",
        "web": "Auto",
        "calendar": "Auto",
        "skills": "Notify",
    },
}


def run_step(console: Console, state: WizardState, detection: DetectionResults) -> bool:
    """Run the security profile selection step.

    Returns True if completed, False to go back.
    """
    console.print()
    console.print(
        Panel(
            f"[{PRIMARY}]Choose a security profile that controls when Aria asks for permission.\n"
            f"[{MUTED}]You can change this later in config/settings.yaml or the dashboard.",
            title=step_title(5, "Security Profile"),
            border_style=PANEL_BORDER,
            box=PANEL_BOX,
        )
    )

    # Show comparison table
    _show_comparison(console)

    # Select profile
    choices = [
        questionary.Choice(
            f"{PROFILES['paranoid']['label']} - {PROFILES['paranoid']['desc']}",
            value="paranoid",
        ),
        questionary.Choice(
            f"{PROFILES['balanced']['label']} - {PROFILES['balanced']['desc']}",
            value="balanced",
        ),
        questionary.Choice(
            f"{PROFILES['trusted']['label']} - {PROFILES['trusted']['desc']}",
            value="trusted",
        ),
    ]

    profile = questionary.select(
        "Security profile:",
        choices=choices,
        default="balanced",
    ).ask()

    if profile is None:
        return False

    state.security_profile = profile

    console.print(
        f"\n  [{PRIMARY}]{ICON_SHIELD}[/{PRIMARY}] Security profile set to "
        f"[bold]{PROFILES[profile]['label']}[/bold]"
    )
    console.print(
        f"  [{MUTED}]* Auto actions are restricted to safe paths/domains"
    )

    return True


def _show_comparison(console: Console) -> None:
    """Show the security profile comparison table."""
    table = Table(
        title="Security Profile Comparison",
        show_header=True,
        header_style=f"bold {PRIMARY}",
        box=None,
        padding=(0, 1),
    )
    table.add_column("Action", style=MUTED, min_width=14)
    table.add_column(f"[{ERROR}]Paranoid[/{ERROR}]", justify="center", min_width=10)
    table.add_column(f"[{WARNING}]Balanced[/{WARNING}]", justify="center", min_width=12)
    table.add_column(f"[{SUCCESS}]Trusted[/{SUCCESS}]", justify="center", min_width=10)

    rows = [
        ("Read files", "read_files"),
        ("Write files", "write_files"),
        ("Delete files", "delete_files"),
        ("Shell commands", "shell_cmds"),
        ("Send messages", "messages"),
        ("Web requests", "web"),
        ("Calendar", "calendar"),
        ("Create skills", "skills"),
    ]

    for label, key in rows:
        table.add_row(
            label,
            _color_action(PROFILES["paranoid"][key]),
            _color_action(PROFILES["balanced"][key]),
            _color_action(PROFILES["trusted"][key]),
        )

    console.print(table)
    console.print()


def _color_action(action: str) -> str:
    """Color an action string based on its type."""
    lower = action.lower()
    if lower.startswith("approve"):
        return f"[{ERROR}]{action}[/{ERROR}]"
    if lower.startswith("notify"):
        return f"[{WARNING}]{action}[/{WARNING}]"
    if lower.startswith("auto"):
        return f"[{SUCCESS}]{action}[/{SUCCESS}]"
    return action
