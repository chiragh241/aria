"""Context skill — share and load unified agent context (OneContext integration)."""

from pathlib import Path
from typing import Any

from ..base import BaseSkill, SkillResult
from ...integrations.onecontext import OneContextBridge
from ...utils.config import get_settings
from ...utils.logging import get_logger

logger = get_logger(__name__)


class ContextSkill(BaseSkill):
    """Share and load context so all agents stay on the same page."""

    name = "context"
    description = "Share and load unified agent context (OneContext integration)"
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._bridge = OneContextBridge()
        self._context_manager = None
        self._coordinator = None

    def set_coordinator(self, coordinator: Any) -> None:
        """Set the agent coordinator for shared context."""
        self._coordinator = coordinator

    def set_context_manager(self, cm: Any) -> None:
        """Set the context manager for exporting conversation."""
        self._context_manager = cm

    def _register_capabilities(self) -> None:
        self.register_capability(
            name="share_context",
            description="Share current conversation/agent context. Returns a shareable path or OneContext link.",
            parameters={
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "user_id": {"type": "string"},
                    "summary": {"type": "string", "description": "Optional summary to include"},
                },
            },
        )
        self.register_capability(
            name="load_context",
            description="Load shared context from file path or export. Use to continue from where another agent left off.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to context JSON file or OneContext export"},
                },
                "required": ["path"],
            },
        )
        self.register_capability(
            name="get_shared_context",
            description="Get the current shared context (tasks, findings) from the agent coordinator.",
            parameters={"type": "object", "properties": {}},
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        from datetime import datetime, timezone

        start = datetime.now(timezone.utc)
        if capability == "share_context":
            return await self._share_context(start, **kwargs)
        if capability == "load_context":
            return await self._load_context(start, **kwargs)
        if capability == "get_shared_context":
            return self._get_shared_context(start, **kwargs)
        return self._error_result(f"Unknown capability: {capability}", start)

    def _get_coordinator(self, kwargs: dict[str, Any]) -> Any:
        """Get coordinator from kwargs or stored reference."""
        return kwargs.get("coordinator") or getattr(self, "_coordinator", None)

    async def _share_context(self, start: Any, **kwargs: Any) -> SkillResult:
        """Share context across ALL channels."""
        channel = kwargs.get("channel", "web")
        user_id = kwargs.get("user_id", "default")
        summary = kwargs.get("summary", "")
        if not self._bridge.enabled and not self._bridge.cli_available:
            return SkillResult(
                success=False,
                output="OneContext not configured. Enable in config/settings.yaml (onecontext.enabled: true) or install: npm i -g onecontext-ai",
                error="onecontext_unavailable",
                start_time=start,
            )
        messages: list[dict[str, Any]] = []
        channels_included: list[str] = []
        if self._context_manager:
            try:
                all_contexts = await self._context_manager.get_active_contexts()
                for ctx in all_contexts:
                    channels_included.append(f"{ctx.channel}:{ctx.user_id}")
                    for m in ctx.get_messages(max_messages=10):
                        messages.append({
                            "role": m.role,
                            "content": (m.content or "")[:400],
                            "source": f"{ctx.channel}:{ctx.user_id}",
                        })
            except Exception as e:
                logger.warning("Could not get contexts for share", error=str(e))
        coord = kwargs.get("coordinator") or getattr(self, "_coordinator", None)
        if coord and hasattr(coord, "_shared_context"):
            messages.append({
                "role": "system",
                "content": f"Shared agent context (all channels): {coord._shared_context}",
                "source": "agents",
            })
        unified_summary = summary or f"Context from channels: {', '.join(channels_included) or 'all'}"
        ok, out = await self._bridge.share_context(
            context_id="aria_unified",
            channel="*",
            user_id="*",
            messages=messages,
            summary=unified_summary,
        )
        if ok:
            return SkillResult(
                success=True,
                output=f"Context shared. Shareable: {out}",
                start_time=start,
            )
        return SkillResult(success=False, output=out, error="share_failed", start_time=start)

    async def _load_context(self, start: Any, path: str = "", **kwargs: Any) -> SkillResult:
        """Load context from file."""
        data = self._bridge.load_import(path)
        if not data:
            return SkillResult(
                success=False,
                output=f"Could not load context from {path or 'empty path'}",
                error="load_failed",
                start_time=start,
            )
        summary = data.get("summary", "")
        tasks = data.get("metadata", {}).get("tasks", [])
        return SkillResult(
            success=True,
            output=f"Loaded context. Summary: {summary} Tasks: {len(tasks)}",
            metadata={"loaded_context": data},
            start_time=start,
        )

    def _get_shared_context(self, start: Any, **kwargs: Any) -> SkillResult:
        """Get current shared context from agent coordinator."""
        coord = kwargs.get("coordinator") or getattr(self, "_coordinator", None)
        if not coord or not hasattr(coord, "_shared_context"):
            return SkillResult(
                success=True,
                output="No shared context available (no agents have run yet).",
                start_time=start,
            )
        ctx = coord._shared_context
        tasks = ctx.get("tasks", [])
        findings = ctx.get("findings", [])
        return SkillResult(
            success=True,
            output=f"Shared context: {len(tasks)} tasks, {len(findings)} findings. Tasks: {[t.get('task','') for t in tasks]}",
            metadata={"shared_context": ctx},
            start_time=start,
        )
