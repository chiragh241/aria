"""Theme, colors, and visual constants for the Aria CLI wizard."""

from rich.box import ROUNDED, SIMPLE
from rich.style import Style
from rich.theme import Theme

# ── Color palette ──────────────────────────────────────────────────────────────

PRIMARY = "bright_cyan"
SECONDARY = "bright_magenta"
SUCCESS = "bright_green"
WARNING = "bright_yellow"
ERROR = "bright_red"
MUTED = "dim white"
ACCENT = "bright_blue"

# ── Rich theme ─────────────────────────────────────────────────────────────────

ARIA_THEME = Theme(
    {
        "info": Style(color="bright_cyan"),
        "success": Style(color="bright_green", bold=True),
        "warning": Style(color="bright_yellow"),
        "error": Style(color="bright_red", bold=True),
        "muted": Style(color="white", dim=True),
        "header": Style(color="bright_cyan", bold=True),
        "accent": Style(color="bright_magenta"),
        "key": Style(color="bright_cyan"),
        "value": Style(color="white"),
        "step": Style(color="bright_magenta", bold=True),
    }
)

# ── Unicode icons ──────────────────────────────────────────────────────────────

ICON_CHECK = "\u2713"       # ✓
ICON_CROSS = "\u2717"       # ✗
ICON_ARROW = "\u25b6"       # ▶
ICON_WARN = "\u26a0"        # ⚠
ICON_INFO = "\u24d8"        # ⓘ
ICON_GEAR = "\u2699"        # ⚙
ICON_BULLET = "\u2022"      # •
ICON_SHIELD = "\u25c6"      # ◆
ICON_STAR = "\u2605"        # ★
ICON_DOT = "\u25cf"         # ●

# ── Box styles ─────────────────────────────────────────────────────────────────

PANEL_BOX = ROUNDED
TABLE_BOX = SIMPLE
PANEL_BORDER = PRIMARY

# ── Spinner ────────────────────────────────────────────────────────────────────

SPINNER_STYLE = "dots"
SPINNER_COLOR = PRIMARY

# ── Step labels ────────────────────────────────────────────────────────────────

STEP_LABELS = [
    "LLM Provider",
    "Channels",
    "Execution Environment",
    "Dashboard",
    "Security Profile",
    "Browser",
    "Skills",
]


def step_title(number: int, label: str) -> str:
    """Format a step title like: Step 1 of 7 ▶ LLM Provider"""
    total = len(STEP_LABELS)
    return f"Step {number} of {total} {ICON_ARROW} {label}"


def status_icon(ok: bool) -> str:
    """Return a colored check or cross mark."""
    if ok:
        return f"[{SUCCESS}]{ICON_CHECK}[/{SUCCESS}]"
    return f"[{ERROR}]{ICON_CROSS}[/{ERROR}]"
