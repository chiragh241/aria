"""Main wizard orchestrator for the Aria CLI setup.

Guides the user through a 7-step interactive configuration process,
then writes settings.yaml and .env files.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.cli.ascii_art import print_banner, print_completion_banner
from src.cli.config_writer import ConfigWriter
from src.cli.detection import DetectionResults, SystemDetector
from src.cli.steps.llm import NVIDIA_DEFAULT_MODEL
from src.cli.styles import (
    ARIA_THEME,
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
    SECONDARY,
    SPINNER_COLOR,
    SPINNER_STYLE,
    SUCCESS,
    WARNING,
    status_icon,
)


@dataclass
class WizardState:
    """Holds all configuration selections from the wizard steps."""

    # Step 1: LLM
    llm_provider: str = "hybrid"  # "anthropic" | "ollama" | "hybrid" | "gemini" | "openrouter" | "nvidia"
    anthropic_auth: str = ""  # "api_key" | "claude_code" | "existing_key"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    ollama_enabled: bool = True
    ollama_model: str = "llama3.2:latest"
    ollama_base_url: str = "http://localhost:11434"
    gemini_enabled: bool = False
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    openrouter_enabled: bool = False
    openrouter_api_key: str = ""
    openrouter_model: str = "anthropic/claude-3.5-sonnet"
    openrouter_use_free: bool = False
    nvidia_enabled: bool = False
    nvidia_api_key: str = ""
    nvidia_model: str = NVIDIA_DEFAULT_MODEL

    # Step 2: Channels
    channels_web: bool = True
    channels_slack: bool = False
    channels_whatsapp: bool = False
    slack_bot_token: str = ""
    slack_app_token: str = ""
    whatsapp_bridge_port: int = 3001
    whatsapp_needs_install: bool = False
    whatsapp_allowed_numbers: list[str] = field(default_factory=list)

    # Step 3: Environment
    deployment_mode: str = "local"  # "local" | "docker"
    sandbox_mode: str = "docker"  # "docker" | "local"
    docker_build_image: bool = False
    docker_memory: str = "512m"
    docker_cpu: float = 1.0
    trusted_paths: list[str] = field(
        default_factory=lambda: ["~/Documents", "~/Projects", "~/Downloads"]
    )

    # Step 4: Dashboard
    dashboard_port: int = 8080
    jwt_secret: str = ""
    admin_password: str = ""

    # Step 5: Security
    security_profile: str = "balanced"

    # Step 6: Browser
    browser_mode: str = "playwright"  # "playwright" | "brave" | "none"
    playwright_install_chromium: bool = False
    brave_api_key: str = ""
    brave_api_key_source: str = ""  # "existing" | ""

    # Step 7: Skills
    enabled_skills: list[str] = field(
        default_factory=lambda: [
            "filesystem", "shell", "browser", "tts", "stt",
            "image", "video", "documents",
        ]
    )
    post_install_commands: list[str] = field(default_factory=list)
    npm_packages_to_install: list[str] = field(default_factory=list)

    # Step 8: Integrations (Notion, Todoist, Linear, Spotify)
    enabled_integrations: list[str] = field(default_factory=list)
    notion_api_key: str = ""
    todoist_api_key: str = ""
    linear_api_key: str = ""
    spotify_client_id: str = ""
    spotify_client_secret: str = ""


def run_wizard() -> bool:
    """Run the interactive setup wizard.

    Returns True if configuration was applied, False if cancelled.
    """
    console = Console(theme=ARIA_THEME)

    # ── Step 0: Welcome ────────────────────────────────────────────────────
    _show_welcome(console)

    # Run system detection while user reads the welcome screen
    console.print(f"  [{MUTED}]Detecting system capabilities...", end="")
    detection = SystemDetector().run_all()
    console.print(f"\r  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] System detection complete          ")
    console.print()

    # Wait for user to continue
    try:
        questionary.press_any_key_to_continue("Press Enter to begin setup...").ask()
    except (KeyboardInterrupt, EOFError):
        return False

    state = WizardState()

    # ── Steps 1-8 ──────────────────────────────────────────────────────────
    from src.cli.steps import browser, channels, dashboard, environment, integrations, llm, security, skills

    steps = [
        ("LLM Provider", llm.run_step),
        ("Channels", channels.run_step),
        ("Execution Environment", environment.run_step),
        ("Dashboard", dashboard.run_step),
        ("Security Profile", security.run_step),
        ("Browser", browser.run_step),
        ("Skills", skills.run_step),
        ("Integrations", integrations.run_step),
    ]

    current_step = 0
    while current_step < len(steps):
        step_name, step_fn = steps[current_step]

        try:
            result = step_fn(console, state, detection)
        except (KeyboardInterrupt, EOFError):
            console.print(f"\n  [{WARNING}]{ICON_WARN}[/{WARNING}] Setup interrupted")
            return False

        if result:
            current_step += 1
        else:
            # Step returned False - go back or cancel
            if current_step > 0:
                go_back = questionary.confirm(
                    "Go back to the previous step?",
                    default=True,
                ).ask()
                if go_back:
                    current_step -= 1
                    continue
            # Cancel
            cancel = questionary.confirm(
                "Cancel setup wizard?",
                default=False,
            ).ask()
            if cancel:
                console.print(f"  [{MUTED}]Setup cancelled. No changes were made.")
                return False

    # ── Summary Screen ─────────────────────────────────────────────────────
    action = _show_summary(console, state)

    if action == "cancel":
        console.print(f"  [{MUTED}]Setup cancelled. No changes were made.")
        return False

    if action == "edit":
        # Let user pick a section to re-edit
        section = questionary.select(
            "Which section would you like to edit?",
            choices=[
                questionary.Choice(name, value=i) for i, (name, _) in enumerate(steps)
            ],
        ).ask()
        if section is not None:
            step_name, step_fn = steps[section]
            try:
                step_fn(console, state, detection)
            except (KeyboardInterrupt, EOFError):
                pass
            # Show summary again recursively
            return _finalize(console, state)

    # ── Apply Configuration ────────────────────────────────────────────────
    return _finalize(console, state)


def _show_welcome(console: Console) -> None:
    """Show the welcome banner and first-time intro. Setup = fresh start; no prior name or state."""
    console.clear()
    print_banner(console, version="0.1.0")

    console.print(
        Panel(
            f"[{PRIMARY}]Welcome — this is your first-time setup.[/{PRIMARY}]\n\n"
            f"[{MUTED}]Aria is your personal AI assistant. It can help you with tasks using your files, "
            f"the web, calendar, email, and more. After setup, Aria will introduce itself and ask what to call you — "
            f"everything starts fresh.\n\n"
            f"This wizard will configure:\n"
            f"  {ICON_ARROW} LLM providers (Claude / Gemini / OpenRouter / NVIDIA / Ollama)\n"
            f"  {ICON_ARROW} Messaging channels (Web / Slack / WhatsApp)\n"
            f"  {ICON_ARROW} Execution environment (Docker / Local)\n"
            f"  {ICON_ARROW} Dashboard and security\n"
            f"  {ICON_ARROW} Browser automation and skills",
            border_style=PANEL_BORDER,
            box=PANEL_BOX,
        )
    )


def _show_summary(console: Console, state: WizardState) -> str:
    """Show a summary of all selections and let user choose next action.

    Returns: "apply", "edit", "save", or "cancel"
    """
    console.print()
    console.print(
        Panel(
            _build_summary_text(state),
            title=f"[{PRIMARY}]Configuration Summary[/{PRIMARY}]",
            border_style=PANEL_BORDER,
            box=PANEL_BOX,
        )
    )

    action = questionary.select(
        "What would you like to do?",
        choices=[
            questionary.Choice("Apply & Start Aria", value="apply"),
            questionary.Choice("Edit a section", value="edit"),
            questionary.Choice("Save config & exit (don't start)", value="save"),
            questionary.Choice("Cancel (discard all changes)", value="cancel"),
        ],
    ).ask()

    return action or "cancel"


def _build_summary_text(state: WizardState) -> str:
    """Build the summary text for the summary panel."""
    s = state
    lines = []

    # LLM
    lines.append(f"[{PRIMARY}]LLM Provider[/{PRIMARY}]")
    lines.append(f"  Mode: {s.llm_provider}")
    if s.llm_provider in ("anthropic", "hybrid"):
        lines.append(f"  Claude model: {s.anthropic_model}")
        lines.append(f"  Auth: {s.anthropic_auth or 'not configured'}")
    if s.llm_provider == "gemini":
        lines.append(f"  Gemini model: {getattr(s, 'gemini_model', 'gemini-2.0-flash')}")
    if s.llm_provider == "openrouter":
        lines.append(f"  OpenRouter model: {getattr(s, 'openrouter_model', 'anthropic/claude-3.5-sonnet')}")
        lines.append(f"  OpenRouter tier: {'free' if getattr(s, 'openrouter_use_free', False) else 'paid'}")
    if s.llm_provider == "nvidia":
        lines.append(f"  NVIDIA model: {getattr(s, 'nvidia_model', NVIDIA_DEFAULT_MODEL)}")
    if s.llm_provider in ("ollama", "hybrid"):
        lines.append(f"  Ollama: {'enabled' if s.ollama_enabled else 'disabled'}")
        if s.ollama_enabled:
            lines.append(f"  Ollama model: {s.ollama_model}")
    lines.append("")

    # Channels
    lines.append(f"[{PRIMARY}]Channels[/{PRIMARY}]")
    lines.append(f"  Web UI: enabled")
    lines.append(f"  Slack: {'enabled' if s.channels_slack else 'disabled'}")
    lines.append(f"  WhatsApp: {'enabled' if s.channels_whatsapp else 'disabled'}")
    lines.append("")

    # Environment
    lines.append(f"[{PRIMARY}]Execution[/{PRIMARY}]")
    lines.append(f"  Deployment: {s.deployment_mode}")
    if s.deployment_mode != "docker":
        lines.append(f"  Sandbox: {s.sandbox_mode}")
        lines.append(f"  Trusted paths: {', '.join(s.trusted_paths)}")
    lines.append("")

    # Dashboard
    lines.append(f"[{PRIMARY}]Dashboard[/{PRIMARY}]")
    lines.append(f"  Port: {s.dashboard_port}")
    lines.append(f"  URL: http://localhost:{s.dashboard_port}")
    lines.append("")

    # Security
    lines.append(f"[{PRIMARY}]Security[/{PRIMARY}]")
    lines.append(f"  Profile: {s.security_profile}")
    lines.append("")

    # Browser
    lines.append(f"[{PRIMARY}]Browser[/{PRIMARY}]")
    lines.append(f"  Mode: {s.browser_mode}")
    lines.append("")

    # Skills
    lines.append(f"[{PRIMARY}]Skills[/{PRIMARY}]")
    lines.append(f"  Enabled: {', '.join(s.enabled_skills)}")

    return "\n".join(lines)


def _finalize(console: Console, state: WizardState) -> bool:
    """Write configuration files and run post-install commands."""
    console.print()
    console.print(f"  [{MUTED}]{ICON_GEAR} Writing configuration files...")

    # Write configs
    writer = ConfigWriter(state)
    try:
        written = writer.write_all()
        for path in written:
            console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] {path}")
        console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Cleared previous chat history and memory for fresh start")
    except Exception as e:
        console.print(f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] Failed to write config: {e}")
        return False

    # Run post-install commands
    if state.post_install_commands:
        console.print(f"\n  [{MUTED}]{ICON_GEAR} Running post-install commands...")
        for cmd in state.post_install_commands:
            console.print(f"  [{MUTED}]{ICON_ARROW} {cmd}")
            try:
                result = subprocess.run(
                    cmd.split(),
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Done")
                else:
                    console.print(
                        f"  [{WARNING}]{ICON_WARN}[/{WARNING}] Command exited with code {result.returncode}"
                    )
            except Exception as e:
                console.print(f"  [{WARNING}]{ICON_WARN}[/{WARNING}] Error: {e}")

    # Install npm packages for selected skills (e.g. onecontext-ai)
    if getattr(state, "npm_packages_to_install", []):
        console.print(f"\n  [{MUTED}]{ICON_GEAR} Installing npm packages...")
        for pkg in state.npm_packages_to_install:
            console.print(f"  [{MUTED}]{ICON_ARROW} npm i -g {pkg}")
            try:
                result = subprocess.run(
                    ["npm", "i", "-g", pkg],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode == 0:
                    console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] {pkg} installed")
                else:
                    console.print(
                        f"  [{WARNING}]{ICON_WARN}[/{WARNING}] {pkg} failed: {result.stderr[:100]}"
                    )
            except Exception as e:
                console.print(f"  [{WARNING}]{ICON_WARN}[/{WARNING}] {pkg} error: {e}")

    # Docker deployment: build and start all containers
    if state.deployment_mode == "docker":
        console.print(f"\n  [{MUTED}]{ICON_GEAR} Setting up Docker deployment...")

        # Check Docker is available
        try:
            result = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                console.print(f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] docker compose not available")
                console.print(f"  [{MUTED}]  Install Docker Compose and try again.")
                return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            console.print(f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] Docker not found on PATH")
            return False

        # Build and start containers
        console.print(f"  [{MUTED}]{ICON_GEAR} Building and starting Docker containers...")
        console.print(f"  [{MUTED}]  This may take a few minutes on first run...")
        try:
            result = subprocess.run(
                ["docker", "compose", "up", "-d", "--build"],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Docker containers started")
            else:
                console.print(f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] docker compose failed:")
                console.print(f"  [{MUTED}]{result.stderr[:500]}")
                return False
        except subprocess.TimeoutExpired:
            console.print(f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] Docker build timed out (10 min)")
            return False

        # Pull Ollama model if Ollama is enabled
        if state.ollama_enabled:
            model = state.ollama_model
            console.print(f"\n  [{MUTED}]{ICON_GEAR} Pulling Ollama model: {model}...")
            console.print(f"  [{MUTED}]  This may take a while for large models...")
            try:
                result = subprocess.run(
                    ["docker", "exec", "aria-ollama", "ollama", "pull", model],
                    capture_output=True, text=True, timeout=600,
                )
                if result.returncode == 0:
                    console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Model {model} pulled")
                else:
                    console.print(
                        f"  [{WARNING}]{ICON_WARN}[/{WARNING}] Model pull failed — "
                        f"you can pull it manually: docker exec aria-ollama ollama pull {model}"
                    )
            except subprocess.TimeoutExpired:
                console.print(
                    f"  [{WARNING}]{ICON_WARN}[/{WARNING}] Model pull timed out — "
                    f"run manually: docker exec aria-ollama ollama pull {model}"
                )
            except Exception as e:
                console.print(f"  [{WARNING}]{ICON_WARN}[/{WARNING}] Model pull error: {e}")

        # Docker deployment is done — show completion and skip local startup
        _show_completion(console, state)
        return True

    # Build docker image if requested
    if state.docker_build_image:
        console.print(f"\n  [{MUTED}]{ICON_GEAR} Building sandbox Docker image...")
        try:
            result = subprocess.run(
                ["docker", "build", "-f", "docker/Dockerfile.sandbox",
                 "-t", "aria-sandbox:latest", "."],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Sandbox image built")
            else:
                console.print(
                    f"  [{WARNING}]{ICON_WARN}[/{WARNING}] Image build failed (you can build it later)"
                )
        except Exception as e:
            console.print(f"  [{WARNING}]{ICON_WARN}[/{WARNING}] Image build error: {e}")

    # Install WhatsApp bridge deps if needed
    if state.whatsapp_needs_install:
        bridge_dir = Path("whatsapp-bridge")
        if bridge_dir.exists():
            console.print(f"\n  [{MUTED}]{ICON_GEAR} Installing WhatsApp bridge dependencies...")
            try:
                result = subprocess.run(
                    ["npm", "install"],
                    cwd=str(bridge_dir),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode == 0:
                    console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Bridge dependencies installed")
                else:
                    console.print(
                        f"  [{WARNING}]{ICON_WARN}[/{WARNING}] npm install failed"
                    )
            except Exception as e:
                console.print(f"  [{WARNING}]{ICON_WARN}[/{WARNING}] Error: {e}")

    # ── Build Web UI Frontend ─────────────────────────────────────────────
    _build_frontend(console)

    # ── Completion Screen ──────────────────────────────────────────────────
    _show_completion(console, state)
    return True


def _build_frontend(console: Console) -> None:
    """Build the web UI frontend (npm install + build).

    This is a required step — retries on failure so the dashboard is usable.
    """
    frontend_dir = Path(__file__).parent.parent / "web" / "frontend"
    dist_index = frontend_dir / "dist" / "index.html"

    if not frontend_dir.exists():
        console.print(
            f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] Frontend source not found at {frontend_dir}"
        )
        return

    # Check if already built and up-to-date
    if dist_index.exists():
        # Check if source files are newer than the build
        src_dir = frontend_dir / "src"
        if src_dir.exists():
            newest_src = max(
                (f.stat().st_mtime for f in src_dir.rglob("*") if f.is_file()),
                default=0,
            )
            build_time = dist_index.stat().st_mtime
            if newest_src <= build_time:
                console.print(
                    f"\n  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Web UI already built and up-to-date"
                )
                return

    # npm is required
    try:
        subprocess.run(["npm", "--version"], capture_output=True, check=True, timeout=10)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        console.print(
            f"\n  [{ERROR}]{ICON_CROSS}[/{ERROR}] npm is required to build the web UI but was not found.\n"
            f"  [{MUTED}]Install Node.js from https://nodejs.org/ then run:\n"
            f"  [{MUTED}]  cd {frontend_dir} && npm install && npm run build"
        )
        return

    # Attempt build, retry once on failure
    for attempt in range(1, 3):
        if attempt > 1:
            console.print(f"\n  [{MUTED}]{ICON_GEAR} Retrying frontend build (attempt {attempt})...")

        # npm install
        console.print(f"\n  [{MUTED}]{ICON_GEAR} Installing web UI dependencies...")
        try:
            result = subprocess.run(
                ["npm", "install"],
                cwd=str(frontend_dir),
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode != 0:
                console.print(
                    f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] npm install failed: {result.stderr[:300]}"
                )
                continue
            console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Dependencies installed")
        except subprocess.TimeoutExpired:
            console.print(f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] npm install timed out")
            continue
        except Exception as e:
            console.print(f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] Error: {e}")
            continue

        # npm run build
        console.print(f"  [{MUTED}]{ICON_GEAR} Building web UI...")
        try:
            result = subprocess.run(
                ["npm", "run", "build"],
                cwd=str(frontend_dir),
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode == 0:
                console.print(f"  [{SUCCESS}]{ICON_CHECK}[/{SUCCESS}] Web UI built successfully")
                return
            else:
                console.print(
                    f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] Build failed: {result.stderr[:300]}"
                )
        except subprocess.TimeoutExpired:
            console.print(f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] Build timed out")
        except Exception as e:
            console.print(f"  [{ERROR}]{ICON_CROSS}[/{ERROR}] Build error: {e}")

    # Both attempts failed
    console.print(
        f"\n  [{ERROR}]{ICON_CROSS}[/{ERROR}] Could not build the web UI.\n"
        f"  [{MUTED}]You can build it manually later:\n"
        f"  [{MUTED}]  cd {frontend_dir} && npm install && npm run build"
    )


def _show_completion(console: Console, state: WizardState) -> None:
    """Show the setup completion screen with next steps."""
    print_completion_banner(console)

    dashboard_url = f"http://localhost:{state.dashboard_port}"

    if state.deployment_mode == "docker":
        tips = [
            f"[{SUCCESS}]Aria is running in Docker containers![/{SUCCESS}]",
            "",
            f"[{PRIMARY}]Dashboard:[/{PRIMARY}]     {dashboard_url}",
            f"[{PRIMARY}]View logs:[/{PRIMARY}]     docker compose logs -f aria",
            f"[{PRIMARY}]Stop:[/{PRIMARY}]          docker compose down",
            f"[{PRIMARY}]Restart:[/{PRIMARY}]       docker compose restart aria",
            f"[{PRIMARY}]Re-run setup:[/{PRIMARY}]  python -m src.main --setup",
            "",
            f"[{MUTED}]Next time you run Aria, it will automatically start Docker.[/{MUTED}]",
            f"[{MUTED}]Use --local flag to run without Docker.[/{MUTED}]",
        ]
    else:
        tips = [
            f"[{SUCCESS}]Aria will now start and open the web dashboard.[/{SUCCESS}]",
            "",
            f"[{PRIMARY}]Dashboard:[/{PRIMARY}]   {dashboard_url}",
            f"[{PRIMARY}]Re-run setup:[/{PRIMARY}] python -m src.main --setup",
            "",
            f"[{MUTED}]You can manage all settings (LLM, channels, security, skills,[/{MUTED}]",
            f"[{MUTED}]browser, etc.) from the Settings page in the web UI.[/{MUTED}]",
        ]

    if state.channels_slack:
        tips.append(
            f"\n[{PRIMARY}]Slack:[/{PRIMARY}]       Invite the Aria bot to your channels"
        )

    if state.channels_whatsapp:
        tips.append(
            f"[{PRIMARY}]WhatsApp:[/{PRIMARY}]    Start the bridge and scan the QR code"
        )

    if state.deployment_mode != "docker" and state.sandbox_mode == "docker" and state.docker_build_image:
        tips.append(
            f"[{PRIMARY}]Sandbox:[/{PRIMARY}]     Docker image built and ready"
        )

    console.print(
        Panel(
            "\n".join(tips),
            title=f"[{SUCCESS}]Setup Complete[/{SUCCESS}]",
            border_style=SUCCESS,
            box=PANEL_BOX,
        )
    )
    console.print()
