"""Package tracking skill."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..base import BaseSkill, SkillResult
from ...utils.config import get_settings
from ...utils.logging import get_logger

logger = get_logger(__name__)


class TrackingSkill(BaseSkill):
    """Track packages by tracking number."""

    name = "tracking"
    description = "Track packages by tracking number"
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._data_path: Path | None = None
        self._packages: list[dict[str, Any]] = []

    def _get_data_path(self) -> Path:
        if self._data_path is None:
            settings = get_settings()
            base = Path(settings.aria.data_dir).expanduser()
            self._data_path = base / "tracking.json"
            self._data_path.parent.mkdir(parents=True, exist_ok=True)
        return self._data_path

    def _load(self) -> None:
        path = self._get_data_path()
        if path.exists():
            try:
                self._packages = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._packages = []
        else:
            self._packages = []

    def _save(self) -> None:
        path = self._get_data_path()
        path.write_text(json.dumps(self._packages, indent=2), encoding="utf-8")

    async def initialize(self) -> None:
        self._load()
        await super().initialize()

    def _register_capabilities(self) -> None:
        self.register_capability(
            name="track",
            description="Track a package by tracking number",
            parameters={
                "type": "object",
                "properties": {
                    "tracking_number": {"type": "string", "description": "Package tracking number"},
                },
                "required": ["tracking_number"],
            },
        )
        self.register_capability(
            name="list_packages",
            description="List all tracked packages",
            parameters={"type": "object", "properties": {}},
        )
        self.register_capability(
            name="add_tracking",
            description="Add a tracking number to the list (for manual tracking)",
            parameters={
                "type": "object",
                "properties": {
                    "tracking_number": {"type": "string"},
                    "carrier": {"type": "string", "description": "Carrier name (optional)"},
                },
                "required": ["tracking_number"],
            },
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        start = datetime.now(timezone.utc)
        self._load()

        if capability == "track":
            return await self._track(kwargs.get("tracking_number", ""), start)
        elif capability == "list_packages":
            return await self._list_packages(start)
        elif capability == "add_tracking":
            return await self._add_tracking(
                kwargs.get("tracking_number", ""),
                kwargs.get("carrier", ""),
                start,
            )
        return self._error_result(f"Unknown capability: {capability}", start)

    async def _track(self, tracking_number: str, start: datetime) -> SkillResult:
        if not tracking_number:
            return self._error_result("Tracking number required", start)

        # Add to list if not already there
        exists = any(
            p.get("tracking_number", "").upper() == tracking_number.upper()
            for p in self._packages
        )
        if not exists:
            self._packages.append({
                "tracking_number": tracking_number,
                "carrier": "unknown",
                "added_at": datetime.now(timezone.utc).isoformat(),
            })
            self._save()

        # Real-time tracking would require API (17track, AfterShip, etc.)
        return self._success_result(
            f"Tracking number {tracking_number} is now being tracked. "
            "Check your carrier's website for the latest status.",
            start,
        )

    async def _list_packages(self, start: datetime) -> SkillResult:
        if not self._packages:
            return self._success_result("No packages being tracked.", start)
        lines = [f"- {p.get('tracking_number', '')} ({p.get('carrier', 'unknown')})" for p in self._packages]
        return self._success_result("\n".join(lines), start)

    async def _add_tracking(self, tracking_number: str, carrier: str, start: datetime) -> SkillResult:
        if not tracking_number:
            return self._error_result("Tracking number required", start)
        self._packages.append({
            "tracking_number": tracking_number,
            "carrier": carrier or "unknown",
            "added_at": datetime.now(timezone.utc).isoformat(),
        })
        self._save()
        return self._success_result(f"Added tracking: {tracking_number}", start)
