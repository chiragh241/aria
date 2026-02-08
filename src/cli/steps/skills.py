"""Step 7: Skills selection and dependency management."""

from __future__ import annotations

import subprocess
import sys
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
    ICON_GEAR,
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

# ── Skill definitions with dependency info ─────────────────────────────────────

SKILL_GROUPS = {
    "Core": {
        "filesystem": {
            "label": "Filesystem Operations",
            "pip": [],
            "system": [],
            "post_install": [],
            "default": True,
        },
        "shell": {
            "label": "Shell Commands",
            "pip": [],
            "system": [],
            "post_install": [],
            "default": True,
        },
    },
    "Web & Browser": {
        "browser": {
            "label": "Web Browser (Playwright)",
            "pip": ["playwright"],
            "system": [],
            "post_install": ["playwright install chromium"],
            "default": True,
        },
    },
    "Media": {
        "tts": {
            "label": "Text-to-Speech",
            "pip": ["edge-tts"],
            "system": [],
            "post_install": [],
            "default": True,
        },
        "stt": {
            "label": "Speech-to-Text (Whisper)",
            "pip": ["openai-whisper"],
            "system": ["ffmpeg"],
            "post_install": [],
            "default": True,
        },
        "image": {
            "label": "Image Processing",
            "pip": ["pillow"],
            "system": [],
            "post_install": [],
            "default": True,
        },
        "video": {
            "label": "Video Processing",
            "pip": ["ffmpeg-python"],
            "system": ["ffmpeg"],
            "post_install": [],
            "default": True,
        },
        "documents": {
            "label": "Document Processing",
            "pip": ["reportlab", "python-docx", "pypdf"],
            "system": [],
            "post_install": [],
            "default": True,
        },
    },
    "Communication": {
        "calendar": {
            "label": "Calendar (Google)",
            "pip": ["google-api-python-client", "google-auth-oauthlib"],
            "system": [],
            "post_install": [],
            "default": False,
        },
        "email": {
            "label": "Email (SMTP/IMAP)",
            "pip": ["aiosmtplib", "aioimaplib"],
            "system": [],
            "post_install": [],
            "default": False,
        },
        "sms": {
            "label": "SMS (Twilio)",
            "pip": ["twilio"],
            "system": [],
            "post_install": [],
            "default": False,
        },
    },
    "Integrations": {
        "notion": {
            "label": "Notion",
            "pip": [],
            "system": [],
            "post_install": [],
            "default": False,
        },
        "todoist": {
            "label": "Todoist",
            "pip": [],
            "system": [],
            "post_install": [],
            "default": False,
        },
        "linear": {
            "label": "Linear",
            "pip": [],
            "system": [],
            "post_install": [],
            "default": False,
        },
        "spotify": {
            "label": "Spotify",
            "pip": [],
            "system": [],
            "post_install": [],
            "default": False,
        },
    },
}


def run_step(console: Console, state: WizardState, detection: DetectionResults) -> bool:
    """Run the skills selection and dependency check step.

    Returns True if completed, False to go back.
    """
    console.print()
    console.print(
        Panel(
            f"[{PRIMARY}]Select which skills (capabilities) Aria should have.\n"
            f"[{MUTED}]Skills with missing dependencies can be installed automatically.",
            title=step_title(7, "Skills"),
            border_style=PANEL_BORDER,
            box=PANEL_BOX,
        )
    )

    # Build choices grouped by category
    all_choices = []
    for group_name, skills in SKILL_GROUPS.items():
        all_choices.append(questionary.Separator(f"--- {group_name} ---"))
        for skill_id, info in skills.items():
            all_choices.append(
                questionary.Choice(
                    title=info["label"],
                    value=skill_id,
                    checked=info["default"],
                )
            )

    selected = questionary.checkbox(
        "Select skills to enable:",
        choices=all_choices,
    ).ask()

    if selected is None:
        return False

    state.enabled_skills = selected
    state.enabled_integrations = [s for s in selected if s in ("notion", "todoist", "linear", "spotify")]

    # Check dependencies for selected skills
    _check_dependencies(console, state, detection, selected)

    _show_summary(console, state)
    return True


def _check_dependencies(
    console: Console,
    state: WizardState,
    detection: DetectionResults,
    selected: list[str],
) -> None:
    """Check and offer to install missing dependencies."""
    # Gather all required packages
    all_pip: list[str] = []
    all_system: list[str] = []
    all_post: list[str] = []

    for skill_id in selected:
        info = _get_skill_info(skill_id)
        if info:
            all_pip.extend(info["pip"])
            all_system.extend(info["system"])
            all_post.extend(info["post_install"])

    # Deduplicate
    all_pip = list(dict.fromkeys(all_pip))
    all_system = list(dict.fromkeys(all_system))
    all_post = list(dict.fromkeys(all_post))

    if not all_pip and not all_system:
        console.print(f"\n  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] No additional dependencies needed")
        return

    # Check pip packages
    pip_status = SystemDetector.check_python_packages(all_pip) if all_pip else {}

    # Check system packages
    system_status: dict[str, bool] = {}
    if "ffmpeg" in all_system:
        system_status["ffmpeg"] = detection.ffmpeg.installed

    # Show dependency table
    console.print()
    table = Table(
        title="Dependency Status",
        show_header=True,
        header_style=f"bold {PRIMARY}",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Package", style=MUTED)
    table.add_column("Type")
    table.add_column("Status")

    missing_pip = []
    missing_system = []

    for pkg, installed in pip_status.items():
        table.add_row(
            pkg,
            "pip",
            f"[{SUCCESS}]{ICON_CHECK} installed[/{SUCCESS}]"
            if installed
            else f"[{WARNING}]{ICON_CROSS} missing[/{WARNING}]",
        )
        if not installed:
            missing_pip.append(pkg)

    for pkg, installed in system_status.items():
        table.add_row(
            pkg,
            "system",
            f"[{SUCCESS}]{ICON_CHECK} installed[/{SUCCESS}]"
            if installed
            else f"[{WARNING}]{ICON_CROSS} missing[/{WARNING}]",
        )
        if not installed:
            missing_system.append(pkg)

    console.print(table)

    # Offer to install missing pip packages
    if missing_pip:
        console.print()
        install = questionary.confirm(
            f"Install {len(missing_pip)} missing pip package(s)?",
            default=True,
        ).ask()

        if install:
            _install_pip_packages(console, missing_pip)

    # Show system package instructions
    if missing_system:
        console.print(f"\n  [{WARNING}]{ICON_WARN}[/{WARNING}] Missing system packages:")
        for pkg in missing_system:
            if pkg == "ffmpeg":
                console.print(f"  [{MUTED}]  {ICON_ARROW} macOS: brew install ffmpeg")
                console.print(f"  [{MUTED}]  {ICON_ARROW} Ubuntu: sudo apt install ffmpeg")
                console.print(f"  [{MUTED}]  {ICON_ARROW} Windows: choco install ffmpeg")

        install_sys = questionary.confirm(
            "Attempt to install missing system packages now?",
            default=False,
        ).ask()

        if install_sys:
            _install_system_packages(console, missing_system)

    # Post-install commands
    if all_post:
        state.post_install_commands = all_post
        console.print(f"\n  [{MUTED}]{ICON_INFO} Post-install commands will run after setup:")
        for cmd in all_post:
            console.print(f"  [{MUTED}]  {ICON_ARROW} {cmd}")


def _install_pip_packages(console: Console, packages: list[str]) -> None:
    """Install missing pip packages."""
    for pkg in packages:
        console.print(f"  [{MUTED}]{ICON_GEAR} Installing {pkg}...", end="")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                console.print(f"\r  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] {pkg} installed          ")
            else:
                console.print(
                    f"\r  [{ERROR}]{ICON_CROSS}[/{ERROR}] {pkg} failed: {result.stderr.strip()[:80]}          "
                )
        except subprocess.TimeoutExpired:
            console.print(f"\r  [{ERROR}]{ICON_CROSS}[/{ERROR}] {pkg} timed out          ")
        except Exception as e:
            console.print(f"\r  [{ERROR}]{ICON_CROSS}[/{ERROR}] {pkg} error: {e}          ")


def _install_system_packages(console: Console, packages: list[str]) -> None:
    """Attempt to install system packages via brew or apt."""
    import platform
    import shutil

    system = platform.system()

    for pkg in packages:
        if system == "Darwin" and shutil.which("brew"):
            cmd = ["brew", "install", pkg]
        elif system == "Linux" and shutil.which("apt"):
            cmd = ["sudo", "apt", "install", "-y", pkg]
        else:
            console.print(
                f"  [{WARNING}]{ICON_WARN}[/{WARNING}] Cannot auto-install {pkg} on this system"
            )
            continue

        console.print(f"  [{MUTED}]{ICON_GEAR} Running: {' '.join(cmd)}...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] {pkg} installed")
            else:
                console.print(
                    f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] {pkg} install failed"
                )
        except Exception as e:
            console.print(f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] {pkg} error: {e}")


def _get_skill_info(skill_id: str) -> dict | None:
    """Look up skill info across all groups."""
    for _group_name, skills in SKILL_GROUPS.items():
        if skill_id in skills:
            return skills[skill_id]
    return None


def _show_summary(console: Console, state: WizardState) -> None:
    """Show skills configuration summary."""
    table = Table(
        title="Enabled Skills",
        show_header=True,
        header_style=f"bold {PRIMARY}",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Skill", style=MUTED)
    table.add_column("Status")

    for group_name, skills in SKILL_GROUPS.items():
        for skill_id, info in skills.items():
            if skill_id in state.enabled_skills:
                table.add_row(
                    info["label"],
                    f"[{SUCCESS}]Enabled[/{SUCCESS}]",
                )
            else:
                table.add_row(
                    info["label"],
                    f"[{MUTED}]Disabled",
                )

    console.print()
    console.print(table)
