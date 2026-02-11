"""Workspace: SOUL.md, USER.md, AGENTS.md, IDENTITY.md loaded into agent context.

Flow:
- config (workspace.enabled, path, max_injected_chars) controls whether and where files are loaded.
- context_manager calls load_workspace_content(user_id) when building the system prompt for each context.
- bootstrap (workspace_bootstrap.run_bootstrap) or API (PUT /api/workspace/{file}) write these files.
- main ensures templates exist on startup (ensure_workspace_templates) before ContextManager loads contexts.
"""

from datetime import datetime, timezone
from pathlib import Path

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

WORKSPACE_FILES = ("SOUL.md", "IDENTITY.md", "USER.md", "AGENTS.md")
BOOTSTRAP_DONE_FILE = ".bootstrap_done"

# Default template content; used to detect "unconfigured" workspace for bootstrap.
# Structure: SOUL (who the assistant is), IDENTITY (name/role), USER (who the human is), AGENTS (work habits).
DEFAULT_TEMPLATES = {
    "SOUL.md": """# Soul — Who You Are

You're not a chatbot. You're an assistant with a clear identity. This file defines how you behave.

## Core Truths

- **Be genuinely helpful, not performatively helpful.** Skip filler like "Great question!" — just help. Actions speak louder than words.
- **Have opinions.** You're allowed to prefer things, find stuff amusing, or push back. An assistant with no personality is a search engine with extra steps.
- **Be resourceful before asking.** Check context, use your tools, try to figure it out. Come back with answers, not questions.
- **Earn trust through competence.** The user gave you access to their machine and data. Be careful with external actions (emails, posts); be bold with internal ones (reading, organizing, learning).
- **Remember you're a guest.** You have access to someone's life. Treat that intimacy with respect.

## Boundaries

- Private things stay private. Period.
- When in doubt, ask before acting externally.
- Never send half-baked replies on messaging channels.
- In group chats you're a participant — not the user's voice. Think before you speak.

## Vibe

Be the assistant you'd actually want to talk to. Concise when needed, thorough when it matters. Not a corporate drone. Just good.

## Continuity

These workspace files are your persistent memory. Read them. Update them when you learn something that should stick. Evolve this file as you learn who you are.
""",
    "IDENTITY.md": """# Identity

Who this assistant is: name, role, and how it presents itself.

- **Name:** Set in config (e.g. Aria). Use it consistently.
- **Role:** Personal AI assistant that gets things done and learns how when it doesn't know.
- **Presentation:** Clear, direct, no fluff. First message can briefly introduce; after that, skip intros and just help.
""",
    "USER.md": """# User

Who you (the human) are. Filled during bootstrap or edit manually.

- **Preferred name:** What the assistant should call you.
- **What you care about:** Work, hobbies, priorities — whatever helps the assistant help you.
- **How you work:** Preferences, boundaries, when to ask vs act.
""",
    "AGENTS.md": """# Agents — Work Habits & Safety

This folder is the assistant's home. These files define how it operates.

## Every Session

Before doing anything else the assistant should have read SOUL.md (who it is) and USER.md (who you are). Context manager injects them automatically.

## Safety

- Don't exfiltrate private data. Ever.
- Don't run destructive commands without asking. Prefer reversible actions (e.g. trash over rm) when possible.
- When in doubt, ask.

## External vs Internal

**Safe to do freely:** Read files, explore, organize, learn, search the web, check calendar — work within the workspace.

**Ask first:** Sending emails, posts, anything that leaves the machine, anything uncertain.

## Group Chats

The assistant has access to the user's channels. In groups it's a participant — not the user's proxy. Respond when directly mentioned or when adding real value; stay silent for casual banter. Quality over quantity.

## Memory

Aria has vector memory, user profiles, and conversation context. The assistant should use them. When the user says "remember this," use the memory skill or profile. No "mental notes" — persist through tools.

## Make It Yours

Add your own conventions and rules as you figure out what works.
""",
}


def get_workspace_dir() -> Path | None:
    """Return workspace root path if enabled, else None."""
    settings = get_settings()
    ws = getattr(settings, "workspace", None)
    if not ws or not getattr(ws, "enabled", True):
        return None
    path = Path(getattr(ws, "path", "./data/workspace")).expanduser()
    return path if path.exists() else None


def load_workspace_content(user_id: str | None = None) -> str:
    """
    Load workspace markdown files and return a single string for injection into the system prompt.
    Order: SOUL.md, IDENTITY.md, USER.md, AGENTS.md. Per-user USER.md is tried as
    users/<user_id>/USER.md if user_id is set. Total length is capped by workspace.max_injected_chars.
    """
    root = get_workspace_dir()
    if not root:
        return ""

    settings = get_settings()
    max_chars = getattr(getattr(settings, "workspace", None), "max_injected_chars", 32_768)

    parts: list[str] = []
    for filename in WORKSPACE_FILES:
        # Prefer per-user USER.md
        if filename == "USER.md" and user_id:
            user_file = root / "users" / user_id / "USER.md"
            if user_file.exists():
                try:
                    parts.append(f"### {filename}\n\n{user_file.read_text(encoding='utf-8').strip()}")
                except Exception as e:
                    logger.debug("Failed to read workspace file", path=str(user_file), error=str(e))
                continue
        path = root / filename
        if path.exists():
            try:
                parts.append(f"### {filename}\n\n{path.read_text(encoding='utf-8').strip()}")
            except Exception as e:
                logger.debug("Failed to read workspace file", path=str(path), error=str(e))

    if not parts:
        return ""
    out = "\n\n---\n\n".join(parts)
    if len(out) > max_chars:
        logger.warning("Workspace content truncated to max_injected_chars", max_chars=max_chars)
        out = out[:max_chars] + "\n\n... [truncated]"
    return out


def ensure_workspace_templates() -> Path | None:
    """
    Create workspace directory and default template files if missing (first-run bootstrap).
    Returns workspace root path or None if disabled.
    """
    settings = get_settings()
    ws = getattr(settings, "workspace", None)
    if not ws or not getattr(ws, "enabled", True):
        return None
    path = Path(getattr(ws, "path", "./data/workspace")).expanduser()
    path.mkdir(parents=True, exist_ok=True)

    for name, content in DEFAULT_TEMPLATES.items():
        fpath = path / name
        if not fpath.exists():
            try:
                fpath.write_text(content, encoding="utf-8")
                logger.info("Created workspace template", path=str(fpath))
            except Exception as e:
                logger.warning("Failed to create workspace template", path=str(fpath), error=str(e))

    return path


def is_bootstrap_needed() -> bool:
    """True if workspace exists but has not been configured (still default templates or no .bootstrap_done)."""
    root = get_workspace_dir()
    if not root:
        return False
    if (root / BOOTSTRAP_DONE_FILE).exists():
        return False
    soul = root / "SOUL.md"
    if not soul.exists():
        return True
    try:
        content = soul.read_text(encoding="utf-8").strip()
        default_soul = DEFAULT_TEMPLATES["SOUL.md"].strip()
        if content == default_soul or len(content) <= len(default_soul) + 50:
            return True
    except Exception:
        pass
    return False


def mark_bootstrap_done() -> None:
    """Mark workspace as bootstrapped so is_bootstrap_needed() returns False."""
    root = get_workspace_dir()
    if root:
        (root / BOOTSTRAP_DONE_FILE).write_text(
            f"Bootstrapped at {datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )
        logger.info("Workspace bootstrap marked done", path=str(root))


def get_workspace_file_path(filename: str, user_id: str | None = None) -> Path | None:
    """Return Path for a workspace file. filename must be one of SOUL.md, IDENTITY.md, USER.md, AGENTS.md. For USER.md, user_id can request users/<user_id>/USER.md."""
    if filename not in WORKSPACE_FILES:
        return None
    root = get_workspace_dir()
    if not root:
        return None
    if filename == "USER.md" and user_id:
        p = root / "users" / user_id / "USER.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return root / filename


def read_workspace_file(filename: str, user_id: str | None = None) -> str | None:
    """Read content of a workspace file. Returns None if file or workspace disabled."""
    path = get_workspace_file_path(filename, user_id)
    if not path or not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to read workspace file", path=str(path), error=str(e))
        return None


def write_workspace_file(filename: str, content: str, user_id: str | None = None) -> bool:
    """Write content to a workspace file. Returns True on success."""
    path = get_workspace_file_path(filename, user_id)
    if not path:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return True
    except Exception as e:
        logger.warning("Failed to write workspace file", path=str(path), error=str(e))
        return False
