"""ASCII art banner for the Aria CLI wizard."""

from rich.console import Console
from rich.text import Text

# Aria banner in block letters
ARIA_BANNER = r"""
     _          ___          _
    / \        |  _ \       (_)        / \
   / _ \       | |_) |       _        / _ \
  / ___ \      |  _ <       | |      / ___ \
 / /   \ \     | | \ \      | |     / /   \ \
/_/     \_\    |_|  \_\     |_|    /_/     \_\
"""

TAGLINE = "Personal AI Assistant"
VERSION_FMT = "v{version}"


def print_banner(console: Console, version: str = "0.1.0") -> None:
    """Print the Aria ASCII art banner with gradient colors."""
    lines = ARIA_BANNER.strip("\n").split("\n")

    # Gradient from bright_magenta to bright_cyan
    gradient_colors = [
        "bright_magenta",
        "bright_magenta",
        "magenta",
        "blue",
        "bright_cyan",
        "bright_cyan",
    ]

    for i, line in enumerate(lines):
        color = gradient_colors[i % len(gradient_colors)]
        styled = Text(line, style=color)
        console.print(styled, highlight=False)

    # Tagline centered
    console.print()
    console.print(
        f"  [bright_cyan]{TAGLINE}[/bright_cyan]  "
        f"[dim]({VERSION_FMT.format(version=version)})[/dim]",
        justify="center",
    )
    console.print()


def print_completion_banner(console: Console) -> None:
    """Print a small completion banner."""
    console.print()
    text = Text("  Aria is ready  ", style="bold bright_green on grey23")
    console.print(text, justify="center")
    console.print()
