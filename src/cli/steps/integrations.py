"""Step 8: Integrations (Notion, Todoist, Linear, Spotify) - API keys."""

from __future__ import annotations

from typing import TYPE_CHECKING

import questionary
from rich.console import Console
from rich.panel import Panel

from src.cli.styles import MUTED, PANEL_BORDER, PANEL_BOX, PRIMARY, step_title

if TYPE_CHECKING:
    from src.cli.detection import DetectionResults
    from src.cli.wizard import WizardState


INTEGRATIONS = {
    "notion": {"label": "Notion", "env_key": "NOTION_API_KEY", "prompt": "Notion integration secret (optional, press Enter to skip):"},
    "todoist": {"label": "Todoist", "env_key": "TODOIST_API_KEY", "prompt": "Todoist API token (optional, press Enter to skip):"},
    "linear": {"label": "Linear", "env_key": "LINEAR_API_KEY", "prompt": "Linear API key (optional, press Enter to skip):"},
    "spotify": {"label": "Spotify", "env_keys": ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET"), "prompt": "Spotify credentials (optional)"},
}


def run_step(console: Console, state: WizardState, detection: DetectionResults) -> bool:
    """Run the integrations API key step.

    Returns True if completed, False to go back.
    """
    console.print()
    console.print(
        Panel(
            f"[{PRIMARY}]Add API keys for integrations enabled in the previous step.\n"
            f"[{MUTED}]You can add these later in the web Settings > Integrations tab.",
            title=step_title(8, "Integrations"),
            border_style=PANEL_BORDER,
            box=PANEL_BOX,
        )
    )

    integrations_to_config = [i for i in state.enabled_integrations if i in INTEGRATIONS]
    if not integrations_to_config:
        console.print(f"\n  [{MUTED}]No integrations selected. Skipping.")
        return True

    for name in integrations_to_config:
        info = INTEGRATIONS[name]
        if name == "spotify":
            state.spotify_client_id = questionary.password(info["prompt"] + " Client ID:").ask() or ""
            state.spotify_client_secret = questionary.password("Client Secret:").ask() or ""
        else:
            key = questionary.password(info["prompt"]).ask() or ""
            setattr(state, f"{name}_api_key", key)

    return True
