"""Step 3: Execution environment configuration (Docker vs Local)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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


def run_step(console: Console, state: WizardState, detection: DetectionResults) -> bool:
    """Run the execution environment configuration step.

    Returns True if completed, False to go back.
    """
    console.print()
    console.print(
        Panel(
            f"[{PRIMARY}]Choose how Aria runs and executes commands.\n"
            f"[{MUTED}]Docker provides isolation; Local gives direct access to your system.",
            title=step_title(3, "Execution Environment"),
            border_style=PANEL_BORDER,
            box=PANEL_BOX,
        )
    )

    # ── Deployment Mode ───────────────────────────────────────────────────
    docker_available = detection.docker.installed and detection.docker.running

    console.print(
        f"\n  [{PRIMARY}]Deployment Mode[/{PRIMARY}]"
    )
    console.print(
        f"  [{MUTED}]How should Aria run? Docker deploys all services (Aria, Redis,\n"
        f"  [{MUTED}]Ollama, WhatsApp bridge) as containers. Local runs directly on your machine."
    )

    deploy_choices = []
    deploy_choices.append(
        questionary.Choice(
            "Local - Run Aria directly on this machine",
            value="local",
        )
    )
    if docker_available:
        deploy_choices.append(
            questionary.Choice(
                "Docker - Run everything in containers (auto-managed)",
                value="docker",
            )
        )
    else:
        deploy_choices.append(
            questionary.Choice(
                "Docker - Not available (Docker not running)",
                value="docker",
                disabled="Docker not detected/running",
            )
        )

    deploy_mode = questionary.select(
        "Deployment mode:",
        choices=deploy_choices,
        default=state.deployment_mode,
    ).ask()

    if deploy_mode is None:
        return False

    state.deployment_mode = deploy_mode

    # If Docker deployment, sandbox is automatically docker too
    if state.deployment_mode == "docker":
        state.sandbox_mode = "docker"
        console.print(
            f"\n  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Docker deployment selected — "
            f"all services will be containerized."
        )
        console.print(
            f"  [{MUTED}]  Aria will run: docker compose up -d --build"
        )
        _show_summary(console, state)
        return True

    console.print(
        f"\n  [{MUTED}]{ICON_ARROW} Local deployment — now configure command execution sandbox:"
    )

    # Show detection results
    _show_detection_status(console, detection)

    # Show comparison table
    _show_comparison(console)

    # Choose mode
    docker_available = detection.docker.installed and detection.docker.running

    choices = []
    if docker_available:
        choices.append(
            questionary.Choice(
                "Docker (Sandboxed) - Recommended",
                value="docker",
            )
        )
    else:
        choices.append(
            questionary.Choice(
                "Docker (Sandboxed) - Not available (Docker not running)",
                value="docker",
                disabled="Docker not detected/running",
            )
        )

    choices.append(
        questionary.Choice(
            "Local (Direct) - Commands run on your machine",
            value="local",
        )
    )

    mode = questionary.select(
        "Execution environment:",
        choices=choices,
        default="docker" if docker_available else "local",
    ).ask()

    if mode is None:
        return False

    state.sandbox_mode = mode

    if mode == "docker":
        _configure_docker(console, state, detection)
    else:
        _configure_local(console, state)

    _show_summary(console, state)
    return True


def _show_detection_status(console: Console, detection: DetectionResults) -> None:
    """Show Docker detection status."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Status", width=3)
    table.add_column("Item")

    if detection.docker.installed:
        if detection.docker.running:
            has_image = detection.docker.extra.get("has_sandbox_image", False)
            image_info = ", sandbox image ready" if has_image else ""
            table.add_row(
                status_icon(True),
                f"Docker is installed and running{image_info}",
            )
        else:
            table.add_row(
                status_icon(False),
                f"[{WARNING}]Docker installed but daemon not running",
            )
    else:
        table.add_row(
            status_icon(False),
            f"[{MUTED}]Docker not installed",
        )

    console.print(table)
    console.print()


def _show_comparison(console: Console) -> None:
    """Show a comparison table of Docker vs Local execution."""
    table = Table(
        title="Environment Comparison",
        show_header=True,
        header_style=f"bold {PRIMARY}",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Feature", style=MUTED)
    table.add_column("Docker (Sandboxed)")
    table.add_column("Local (Direct)")

    table.add_row(
        "Isolation",
        f"[{SUCCESS}]Full isolation[/{SUCCESS}]",
        f"[{WARNING}]No isolation[/{WARNING}]",
    )
    table.add_row(
        "File access",
        "Mounted paths only",
        "Full system access",
    )
    table.add_row(
        "Network",
        "Disabled by default",
        "Full network access",
    )
    table.add_row(
        "Performance",
        "Slight overhead",
        f"[{SUCCESS}]Native speed[/{SUCCESS}]",
    )
    table.add_row(
        "Setup",
        "Requires Docker",
        "No extra setup",
    )

    console.print(table)
    console.print()


def _configure_docker(
    console: Console, state: WizardState, detection: DetectionResults
) -> None:
    """Configure Docker sandbox settings."""
    has_image = detection.docker.extra.get("has_sandbox_image", False)

    if not has_image:
        console.print(
            f"  [{MUTED}]{ICON_INFO} The sandbox Docker image needs to be built."
        )
        build = questionary.confirm(
            "Build sandbox image after setup?",
            default=True,
        ).ask()
        state.docker_build_image = bool(build)
        if state.docker_build_image:
            console.print(
                f"  [{MUTED}]{ICON_ARROW} Will run: docker build -f docker/Dockerfile.sandbox -t aria-sandbox:latest ."
            )
    else:
        console.print(
            f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Sandbox image already built"
        )

    # Memory limit
    memory = questionary.text(
        "Container memory limit:",
        default="512m",
    ).ask()
    state.docker_memory = memory or "512m"

    # CPU limit
    cpu = questionary.text(
        "Container CPU limit:",
        default="1.0",
        validate=lambda v: _is_float(v),
    ).ask()
    state.docker_cpu = float(cpu) if cpu else 1.0

    # Trusted paths to mount
    console.print(f"\n  [{MUTED}]{ICON_INFO} Trusted paths will be mounted into the container.")
    paths = questionary.text(
        "Trusted paths (comma-separated):",
        default="~/Documents, ~/Projects, ~/Downloads",
    ).ask()

    if paths:
        state.trusted_paths = [p.strip() for p in paths.split(",") if p.strip()]


def _configure_local(console: Console, state: WizardState) -> None:
    """Configure local execution settings."""
    console.print(
        f"\n  [{WARNING}]{ICON_WARN}[/{WARNING}] Local mode: commands will run directly on your machine."
    )
    console.print(
        f"  [{MUTED}]The security profile (Step 5) controls which commands need approval."
    )

    # Trusted paths
    paths = questionary.text(
        "Trusted file paths (comma-separated):",
        default="~/Documents, ~/Projects, ~/Downloads",
    ).ask()

    if paths:
        state.trusted_paths = [p.strip() for p in paths.split(",") if p.strip()]


def _show_summary(console: Console, state: WizardState) -> None:
    """Show environment configuration summary."""
    table = Table(
        title="Execution Environment",
        show_header=True,
        header_style=f"bold {PRIMARY}",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Setting", style=MUTED)
    table.add_column("Value")

    table.add_row("Deployment", state.deployment_mode)
    if state.deployment_mode != "docker":
        table.add_row("Sandbox mode", state.sandbox_mode)
        if state.sandbox_mode == "docker":
            table.add_row("Memory limit", state.docker_memory)
            table.add_row("CPU limit", str(state.docker_cpu))
            table.add_row("Build image", str(state.docker_build_image))
        table.add_row("Trusted paths", ", ".join(state.trusted_paths))

    console.print()
    console.print(table)


def _is_float(val: str) -> bool | str:
    """Validate that a string is a valid float."""
    try:
        float(val)
        return True
    except ValueError:
        return "Please enter a valid number"
