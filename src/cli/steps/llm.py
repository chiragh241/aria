"""Step 1: LLM Provider selection and configuration."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.cli.detection import SystemDetector
from src.cli.hardware import (
    best_downloaded_model,
    detect_hardware,
    format_hardware_summary,
    get_suggested_models,
    model_supports_tools,
    recommend_ollama_model,
)
from src.cli.styles import (
    ERROR,
    ICON_ARROW,
    ICON_CHECK,
    ICON_CROSS,
    ICON_GEAR,
    ICON_STAR,
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
    """Run the LLM provider configuration step.

    Returns True if the step completed successfully, False to go back.
    """
    console.print()
    console.print(
        Panel(
            f"[{PRIMARY}]Choose how Aria connects to large language models.\n"
            f"[{MUTED}]Hybrid mode uses a local model for simple tasks and Claude for complex ones.",
            title=step_title(1, "LLM Provider"),
            border_style=PANEL_BORDER,
            box=PANEL_BOX,
        )
    )

    # Show current detection status
    _show_detection_status(console, detection)

    # Choose provider mode
    choices = [
        questionary.Choice(
            title="Both - Hybrid (Recommended)",
            value="hybrid",
        ),
        questionary.Choice(
            title="Anthropic Claude only",
            value="anthropic",
        ),
        questionary.Choice(
            title="Ollama only (local, no API key needed)",
            value="ollama",
        ),
    ]

    provider = questionary.select(
        "Which LLM provider(s) would you like to use?",
        choices=choices,
        default="hybrid",
    ).ask()

    if provider is None:
        return False

    state.llm_provider = provider

    # Configure Anthropic if needed
    if provider in ("anthropic", "hybrid"):
        if not _configure_anthropic(console, state, detection):
            return False

    # Configure Ollama if needed
    if provider in ("ollama", "hybrid"):
        if not _configure_ollama(console, state, detection):
            return False

    # Show summary
    _show_summary(console, state)
    return True


def _show_detection_status(console: Console, detection: DetectionResults) -> None:
    """Show what was auto-detected for LLM providers."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Status", width=3)
    table.add_column("Item")

    # Anthropic
    if detection.anthropic_key.installed:
        source = detection.anthropic_key.extra.get("source", "unknown")
        table.add_row(
            status_icon(True),
            f"Anthropic API key found ({source})",
        )
    else:
        table.add_row(
            status_icon(False),
            f"[{MUTED}]No Anthropic API key detected",
        )

    # Ollama
    if detection.ollama.installed:
        models = detection.ollama.extra.get("models", [])
        model_info = f", {len(models)} model(s)" if models else ""
        running = "running" if detection.ollama.running else "not running"
        table.add_row(
            status_icon(True),
            f"Ollama installed ({running}{model_info})",
        )
    else:
        table.add_row(
            status_icon(False),
            f"[{MUTED}]Ollama not installed",
        )

    console.print(table)
    console.print()


def _configure_anthropic(
    console: Console, state: WizardState, detection: DetectionResults
) -> bool:
    """Configure Anthropic API access."""
    console.print(f"\n[{PRIMARY}]Anthropic Configuration[/{PRIMARY}]")

    # Determine auth method
    has_key = detection.anthropic_key.installed
    source = detection.anthropic_key.extra.get("source", "")

    if source == "claude_code":
        console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Claude Code auth detected at ~/.claude/")
        use_existing = questionary.confirm(
            "Use existing Claude Code authentication?",
            default=True,
        ).ask()
        if use_existing is None:
            return False
        if use_existing:
            state.anthropic_auth = "claude_code"
            return True

    if has_key and source in ("environment", ".env"):
        console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] API key found in {source}")
        use_existing = questionary.confirm(
            "Use the existing API key?",
            default=True,
        ).ask()
        if use_existing is None:
            return False
        if use_existing:
            state.anthropic_auth = "existing_key"
            return True

    # Need to enter a key
    auth_method = questionary.select(
        "How would you like to authenticate with Anthropic?",
        choices=[
            questionary.Choice("Enter API key", value="api_key"),
            questionary.Choice("Use Claude Code auth (~/.claude/)", value="claude_code"),
        ],
    ).ask()

    if auth_method is None:
        return False

    if auth_method == "api_key":
        api_key = questionary.password(
            "Enter your Anthropic API key (sk-ant-...):",
        ).ask()

        if api_key is None:
            return False

        if not api_key.startswith("sk-ant-"):
            console.print(f"  [{WARNING}]{ICON_WARN}[/{WARNING}] Key doesn't start with 'sk-ant-' - it may be invalid")

        # Validate the key
        console.print(f"  [{MUTED}]Validating API key...", end="")
        valid, msg = SystemDetector.validate_anthropic_key(api_key)
        if valid:
            console.print(f"\r  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] API key is valid          ")
            state.anthropic_api_key = api_key
            state.anthropic_auth = "api_key"
        else:
            console.print(f"\r  [{ERROR}]{ICON_CROSS}[/{ERROR}] Validation failed: {msg}          ")
            proceed = questionary.confirm(
                "Use this key anyway?",
                default=False,
            ).ask()
            if not proceed:
                return False
            state.anthropic_api_key = api_key
            state.anthropic_auth = "api_key"
    else:
        state.anthropic_auth = "claude_code"

    # Choose model
    model = questionary.select(
        "Which Claude model should Aria use?",
        choices=[
            questionary.Choice("Claude Sonnet 4 (recommended)", value="claude-sonnet-4-20250514"),
            questionary.Choice("Claude Opus 4.5", value="claude-opus-4-5-20251101"),
            questionary.Choice("Claude Haiku 3.5", value="claude-3-5-haiku-20241022"),
        ],
        default="claude-sonnet-4-20250514",
    ).ask()

    if model is None:
        return False

    state.anthropic_model = model
    return True


def _configure_ollama(
    console: Console, state: WizardState, detection: DetectionResults
) -> bool:
    """Configure Ollama local LLM with hardware-based model suggestions."""
    console.print(f"\n[{PRIMARY}]Ollama Configuration[/{PRIMARY}]")

    if not detection.ollama.installed:
        console.print(f"  [{WARNING}]{ICON_WARN}[/{WARNING}] Ollama is not installed.")
        console.print(f"  [{MUTED}]Install it from https://ollama.ai")

        skip = questionary.confirm(
            "Skip Ollama setup for now?",
            default=True,
        ).ask()

        if skip or skip is None:
            if state.llm_provider == "hybrid":
                state.ollama_enabled = False
                console.print(f"  [{MUTED}]{ICON_ARROW} Ollama will be disabled; using Anthropic only")
            return True

    if not detection.ollama.running:
        console.print(f"  [{WARNING}]{ICON_WARN}[/{WARNING}] Ollama is installed but not running.")
        console.print(f"  [{MUTED}]Start it with: ollama serve")

    # Run hardware detection and suggest models
    specs = detect_hardware()
    available_models = detection.ollama.extra.get("models", [])

    # Prefer best downloaded model if it fits and beats tier recommendation
    best_dl, best_dl_reason = best_downloaded_model(available_models, specs)
    if best_dl:
        recommended_model, reason = best_dl, best_dl_reason
    else:
        recommended_model, reason = recommend_ollama_model(specs)

    suggested = get_suggested_models(specs)

    console.print()
    console.print(
        Panel(
            f"[{PRIMARY}]Your hardware:[/{PRIMARY}] {format_hardware_summary(specs)}\n"
            f"[{SUCCESS}]{ICON_STAR} Recommended:[/{SUCCESS}] {recommended_model} — {reason}",
            title=f"{ICON_GEAR} Hardware detected",
            border_style=PANEL_BORDER,
            box=PANEL_BOX,
        )
    )
    console.print()

    # Build model choices: available + suggested (with download indicator)
    model_choices: list[questionary.Choice] = []

    for model_name, is_rec in suggested:
        is_recommended = is_rec or (model_name == recommended_model)
        if model_name in available_models:
            title = f"{model_name} {'(Recommended)' if is_recommended else ''}"
            model_choices.append(questionary.Choice(title, value=model_name))
        else:
            suffix = " (Recommended — will be downloaded)" if is_recommended else " (will be downloaded)"
            model_choices.append(
                questionary.Choice(f"{model_name}{suffix}", value=f"_pull:{model_name}")
            )

    # Add already-downloaded models not in suggested list (only tool-capable models)
    for m in available_models:
        if model_supports_tools(m) and not any(c.value == m for c in model_choices):
            title = f"{m} (Recommended)" if m == recommended_model else m
            model_choices.append(questionary.Choice(title, value=m))

    model_choices.append(questionary.Choice("Enter a different model name...", value="_pull"))

    model = questionary.select(
        "Which Ollama model should Aria use?",
        choices=model_choices,
    ).ask()

    if model is None:
        return False

    if model == "_pull":
        model = _pull_model(console)
        if not model:
            return False
        if model not in available_models:
            if _confirm_and_pull_model(console, model):
                state.ollama_model = model
            else:
                state.ollama_model = model  # Use anyway, user can pull later
    elif model.startswith("_pull:"):
        model = model.split(":", 1)[1]
        if model not in available_models:
            _confirm_and_pull_model(console, model)
        state.ollama_model = model
    else:
        state.ollama_model = model

    state.ollama_enabled = True

    # Base URL
    base_url = questionary.text(
        "Ollama base URL:",
        default="http://localhost:11434",
    ).ask()

    if base_url is None:
        return False

    state.ollama_base_url = base_url
    return True


def _confirm_and_pull_model(console: Console, model: str) -> bool:
    """Offer to download the model and run ollama pull. Returns True if pull succeeded."""
    do_pull = questionary.confirm(
        f"Download {model} now? (This may take a few minutes)",
        default=True,
    ).ask()

    if not do_pull:
        console.print(f"  [{MUTED}]{ICON_ARROW} You can pull it later with: ollama pull {model}")
        return False

    console.print(f"  [{MUTED}]{ICON_GEAR} Pulling {model}...")
    try:
        result = subprocess.run(
            ["ollama", "pull", model],
            capture_output=False,
            text=True,
            timeout=3600,
        )
        if result.returncode == 0:
            console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] {model} downloaded successfully")
            return True
        console.print(f"  [{WARNING}]{ICON_WARN}[/{WARNING}] Pull failed. Use: ollama pull {model}")
        return False
    except subprocess.TimeoutExpired:
        console.print(f"  [{WARNING}]{ICON_WARN}[/{WARNING}] Pull timed out. Use: ollama pull {model}")
        return False
    except FileNotFoundError:
        console.print(f"  [{WARNING}]{ICON_WARN}[/{WARNING}] ollama not found. Use: ollama pull {model}")
        return False


def _pull_model(console: Console) -> str | None:
    """Prompt user to enter a model name to pull."""
    model = questionary.text(
        "Enter model name to pull (e.g. llama3.2:latest, mistral, codellama):",
        default="llama3.2:latest",
    ).ask()

    if model is None:
        return None

    console.print(f"  [{MUTED}]{ICON_ARROW} You can pull this model with: ollama pull {model}")
    return model


def _show_summary(console: Console, state: WizardState) -> None:
    """Show a summary of LLM configuration."""
    table = Table(
        title="LLM Configuration",
        show_header=True,
        header_style=f"bold {PRIMARY}",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Setting", style=MUTED)
    table.add_column("Value")

    table.add_row("Provider mode", state.llm_provider)

    if state.llm_provider in ("anthropic", "hybrid"):
        table.add_row("Anthropic auth", state.anthropic_auth or "not set")
        table.add_row("Claude model", state.anthropic_model)

    if state.llm_provider in ("ollama", "hybrid"):
        table.add_row("Ollama enabled", str(state.ollama_enabled))
        table.add_row("Ollama model", state.ollama_model)
        table.add_row("Ollama URL", state.ollama_base_url)

    console.print()
    console.print(table)
