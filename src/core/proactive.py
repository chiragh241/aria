"""Proactive intelligence engine — morning briefings, follow-ups, suggestions."""

from datetime import datetime, timezone
from typing import Any

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ProactiveEngine:
    """
    Generates proactive content: morning briefings, follow-up checks, suggestions.

    Subscribes to heartbeat events and emits proactive_alert for channels to deliver.
    """

    def __init__(
        self,
        event_bus: Any = None,
        orchestrator: Any = None,
        skill_registry: Any = None,
        scheduler: Any = None,
    ) -> None:
        self.settings = get_settings()
        self._event_bus = event_bus
        self._orchestrator = orchestrator
        self._skill_registry = skill_registry
        self._scheduler = scheduler
        self._subscribed = False

    async def start(self) -> None:
        """Subscribe to heartbeat and cron events."""
        if not self._event_bus or self._subscribed:
            return
        self._event_bus.on("heartbeat", self._on_heartbeat)
        self._event_bus.on("cron_fired", self._on_cron_fired)
        self._subscribed = True
        logger.info("Proactive engine started")

    async def stop(self) -> None:
        """Unsubscribe from events."""
        if self._event_bus and self._subscribed:
            self._event_bus.off("heartbeat", self._on_heartbeat)
            self._event_bus.off("cron_fired", self._on_cron_fired)
            self._subscribed = False

    async def _on_heartbeat(self, event: Any) -> None:
        """Called on each heartbeat."""
        if not self.settings.proactive.enabled:
            return

    async def _on_cron_fired(self, event: Any) -> None:
        """Handle scheduled cron jobs (e.g. morning briefing)."""
        if not self.settings.proactive.enabled:
            return
        # Event bus passes Event object; payload is in event.data
        data = event.data if hasattr(event, "data") else event
        payload = data.get("payload") or {}
        if payload.get("type") == "morning_briefing":
            user_id = data.get("user_id") or payload.get("user_id", "default")
            channel = data.get("channel") or payload.get("channel") or self.settings.proactive.briefing_channel
            try:
                briefing = await self.generate_morning_briefing(user_id)
                if briefing:
                    await self._emit_proactive(briefing, user_id, channel)
            except Exception as e:
                logger.error("Morning briefing failed", error=str(e))

    async def generate_morning_briefing(self, user_id: str = "default") -> str:
        """Generate a morning briefing: weather, calendar, news, reminders."""
        if not self.settings.proactive.morning_briefing:
            return ""

        parts: list[str] = []
        registry = self._skill_registry

        # Weather
        if registry:
            weather_skill = registry.get_skill("weather")
            if weather_skill and weather_skill.enabled:
                try:
                    result = await weather_skill.execute("current", location="New York")
                    if result.success and result.output:
                        parts.append(f"Weather: {str(result.output)[:200]}")
                except Exception:
                    pass

        # Calendar (today's events)
        if registry:
            cal_skill = registry.get_skill("calendar")
            if cal_skill and cal_skill.enabled:
                try:
                    result = await cal_skill.execute("list_events")
                    if result.success and result.output:
                        parts.append(f"Today: {str(result.output)[:200]}")
                except Exception:
                    pass

        # News headlines
        if registry:
            news_skill = registry.get_skill("news")
            if news_skill and news_skill.enabled:
                try:
                    result = await news_skill.execute("headlines", count=3)
                    if result.success and result.output:
                        parts.append(f"News: {str(result.output)[:200]}")
                except Exception:
                    pass

        # Scheduled reminders
        if self._scheduler:
            try:
                jobs = self._scheduler.list_jobs()
                due = [j for j in jobs if j.get("next_run_at") and j.get("action") == "reminder"]
                if due:
                    parts.append(f"Reminders: {len(due)} due today")
            except Exception:
                pass

        if not parts:
            return "Good morning! No updates to report. Have a great day!"

        return "Good morning!\n\n" + "\n\n".join(parts)

    async def check_followups(self, user_id: str) -> list[str]:
        """Check for unresolved action items from recent conversations."""
        if not self.settings.proactive.follow_up_tracking:
            return []
        # Would integrate with conversation summarizer action items
        return []

    async def get_suggestions(self, user_id: str, context: str = "") -> list[str]:
        """Get context-aware suggestions."""
        if not self.settings.proactive.suggestions_enabled:
            return []
        return []

    async def _emit_proactive(
        self,
        message: str,
        user_id: str,
        channel: str = "",
    ) -> None:
        """Emit proactive_alert event for channels to deliver."""
        if self._event_bus:
            await self._event_bus.emit("proactive_alert", {
                "message": message,
                "user_id": user_id,
                "channel": channel,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, source="proactive")
