"""OneContext integration — unified context for all AI agents.

OneContext (https://github.com/TheAgentContextLab/OneContext) provides:
- Record agent trajectory in sessions
- Share context across ALL channels (web, Slack, WhatsApp) so anyone can talk to it
- Load shared context to continue from the same point

When enabled, Aria syncs context with OneContext so all agents stay on the same page.
"""

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class OneContextBridge:
    """
    Bridge to OneContext for unified agent context.

    - Exports Aria's conversation context to OneContext-compatible format
    - Can invoke OneContext CLI when available (onecontext, oc)
    - Provides share/load context for cross-agent continuity
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._cli_path: str | None = None
        self._available: bool | None = None

    @property
    def enabled(self) -> bool:
        """Whether OneContext integration is enabled."""
        oc = getattr(self.settings, "onecontext", None)
        return oc.enabled if oc and hasattr(oc, "enabled") else False

    def _find_cli(self) -> str | None:
        """Find OneContext CLI (onecontext or oc)."""
        if self._cli_path is not None:
            return self._cli_path
        for cmd in ("onecontext", "onecontext-ai", "oc"):
            path = shutil.which(cmd)
            if path:
                self._cli_path = path
                return path
        self._cli_path = ""
        return None

    @property
    def cli_available(self) -> bool:
        """Whether OneContext CLI is installed."""
        if self._available is None:
            self._available = self._find_cli() is not None
        return self._available

    async def run_cli(self, *args: str, timeout: int = 30) -> tuple[bool, str]:
        """Run OneContext CLI command. Returns (success, output)."""
        cli = self._find_cli()
        if not cli:
            return False, "OneContext CLI not found. Install: npm i -g onecontext-ai"
        try:
            proc = await asyncio.create_subprocess_exec(
                cli,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
            out = (stdout or b"").decode("utf-8", errors="replace").strip()
            err = (stderr or b"").decode("utf-8", errors="replace").strip()
            if proc.returncode != 0:
                return False, err or out
            return True, out
        except asyncio.TimeoutError:
            return False, "OneContext CLI timed out"
        except Exception as e:
            return False, str(e)

    def export_context(
        self,
        context_id: str,
        channel: str,
        user_id: str,
        messages: list[dict[str, Any]],
        summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Export Aria context to OneContext-compatible JSON format."""
        return {
            "source": "aria",
            "context_id": context_id,
            "channel": channel,
            "user_id": user_id,
            "messages": messages,
            "summary": summary,
            "metadata": metadata or {},
        }

    def save_export(self, data: dict[str, Any], path: Path | None = None) -> Path:
        """Save exported context to file. Returns path."""
        base = Path(self.settings.aria.data_dir).expanduser() / "onecontext"
        base.mkdir(parents=True, exist_ok=True)
        fp = path or base / f"context_{data.get('context_id', 'export')}.json"
        fp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.debug("Exported context to OneContext format", path=str(fp))
        return fp

    def load_import(self, path: Path | str) -> dict[str, Any] | None:
        """Load context from OneContext export file."""
        p = Path(path).expanduser()
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to load OneContext import", path=str(p), error=str(e))
            return None

    async def share_context(
        self,
        context_id: str,
        channel: str,
        user_id: str,
        messages: list[dict[str, Any]],
        summary: str = "",
    ) -> tuple[bool, str]:
        """
        Share context. Saves to OneContext-compatible JSON; if CLI available, can get share link.
        Returns (success, share_link_or_path).
        """
        data = self.export_context(context_id, channel, user_id, messages, summary)
        fp = self.save_export(data)
        if self.cli_available:
            link = await self.get_shared_link(str(fp))
            if link:
                return True, link
        return True, str(fp)

    async def get_shared_link(self, export_path: str) -> str | None:
        """Get shareable link from OneContext for an export file."""
        if not self.cli_available:
            return None
        ok, out = await self.run_cli("link", "--file", export_path)
        if ok and out:
            return out.strip()
        return None
