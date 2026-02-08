"""Step 6: Browser automation configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.cli.detection import SystemDetector
from src.cli.styles import (
    ICON_ARROW,
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
    status_icon,
)

if TYPE_CHECKING:
    from src.cli.detection import DetectionResults
    from src.cli.wizard import WizardState


def run_step(console: Console, state: WizardState, detection: DetectionResults) -> bool:
    """Run the browser configuration step.

    Returns True if completed, False to go back.
    """
    console.print()
    console.print(
        Panel(
            f"[{PRIMARY}]Choose how Aria accesses the web for browsing and search.\n"
            f"[{MUTED}]Playwright automates a real browser; Brave API provides search results.",
            title=step_title(6, "Browser"),
            border_style=PANEL_BORDER,
            box=PANEL_BOX,
        )
    )

    # Show detection
    _show_detection_status(console, detection)

    # Choose browser mode
    choices = [
        questionary.Choice(
            "Playwright (full browser automation)",
            value="playwright",
        ),
        questionary.Choice(
            "Brave Search API (search results only, no page interaction)",
            value="brave",
        ),
        questionary.Choice(
            "None (disable web browsing)",
            value="none",
        ),
    ]

    mode = questionary.select(
        "Web browsing method:",
        choices=choices,
        default="playwright",
    ).ask()

    if mode is None:
        return False

    state.browser_mode = mode

    if mode == "playwright":
        _configure_playwright(console, state, detection)
    elif mode == "brave":
        if not _configure_brave(console, state, detection):
            return False

    _show_summary(console, state)
    return True


def _show_detection_status(console: Console, detection: DetectionResults) -> None:
    """Show browser-related detection status."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Status", width=3)
    table.add_column("Item")

    # Playwright
    if detection.playwright.installed:
        has_chromium = detection.playwright.extra.get("has_chromium", False)
        if has_chromium:
            table.add_row(status_icon(True), "Playwright installed with Chromium")
        else:
            table.add_row(
                status_icon(False),
                f"[{WARNING}]Playwright installed but Chromium not found",
            )
    else:
        table.add_row(
            status_icon(False),
            f"[{MUTED}]Playwright not installed",
        )

    # Brave API key
    if detection.brave_key.installed:
        table.add_row(status_icon(True), "Brave Search API key found")
    else:
        table.add_row(
            status_icon(False),
            f"[{MUTED}]No Brave Search API key found",
        )

    console.print(table)
    console.print()


def _configure_playwright(
    console: Console, state: WizardState, detection: DetectionResults
) -> None:
    """Configure Playwright browser automation."""
    if not detection.playwright.installed:
        console.print(
            f"  [{MUTED}]{ICON_INFO} Playwright package is already in pyproject.toml dependencies."
        )

    has_chromium = detection.playwright.extra.get("has_chromium", False)
    if not has_chromium:
        console.print(
            f"  [{WARNING}]{ICON_WARN}[/{WARNING}] Chromium browser not installed for Playwright."
        )
        install = questionary.confirm(
            "Install Chromium after setup? (playwright install chromium)",
            default=True,
        ).ask()
        state.playwright_install_chromium = bool(install)
        if state.playwright_install_chromium:
            console.print(
                f"  [{MUTED}]{ICON_ARROW} Will run: playwright install chromium"
            )
    else:
        console.print(
            f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Chromium is ready for Playwright"
        )


def _configure_brave(
    console: Console, state: WizardState, detection: DetectionResults
) -> bool:
    """Configure Brave Search API."""
    if detection.brave_key.installed:
        console.print(
            f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Brave API key found in environment"
        )
        use_existing = questionary.confirm(
            "Use the existing Brave API key?",
            default=True,
        ).ask()
        if use_existing is None:
            return False
        if use_existing:
            state.brave_api_key_source = "existing"
            return True

    api_key = questionary.password(
        "Enter your Brave Search API key:",
    ).ask()

    if api_key is None:
        return False

    if api_key:
        console.print(f"  [{MUTED}]Validating Brave API key...", end="")
        valid, msg = SystemDetector.validate_brave_key(api_key)
        if valid:
            console.print(
                f"\r  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] API key is valid          "
            )
        else:
            console.print(
                f"\r  [{WARNING}]{ICON_WARN}[/{WARNING}] Validation failed: {msg}          "
            )

        state.brave_api_key = api_key

    return True


def _show_summary(console: Console, state: WizardState) -> None:
    """Show browser configuration summary."""
    table = Table(
        title="Browser Configuration",
        show_header=True,
        header_style=f"bold {PRIMARY}",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Setting", style=MUTED)
    table.add_column("Value")

    table.add_row("Mode", state.browser_mode)

    if state.browser_mode == "playwright":
        table.add_row(
            "Install Chromium",
            "Yes" if state.playwright_install_chromium else "Already installed",
        )
    elif state.browser_mode == "brave":
        table.add_row("API key", "configured" if state.brave_api_key else "from environment")

    console.print()
    console.print(table)
