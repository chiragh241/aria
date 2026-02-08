"""Proactive intelligence engine — morning briefings, follow-ups, suggestions."""

from datetime import datetime, timezone
from typing import Any

from ..features.registry import is_feature_enabled
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
        """Called on each heartbeat. Handles context reminders and time-of-day awareness."""
        if not self.settings.proactive.enabled:
            return

        data = event.data if hasattr(event, "data") else event
        user_id = data.get("user_id", "default")
        channel = data.get("channel") or "web"
        context = data.get("context") or {}

        # Context reminders (e.g. when in Slack channel, opening repo)
        if is_feature_enabled("context_reminders"):
            reminders = await self._get_context_reminders(user_id, context)
            for msg in reminders:
                await self._emit_proactive(msg, user_id, channel)

        # Proactive suggestions (time-aware)
        if is_feature_enabled("proactive_suggestions"):
            suggestions = await self.get_suggestions(user_id, str(context))
            if suggestions:
                await self._emit_proactive(
                    "Quick suggestion: " + suggestions[0],
                    user_id,
                    channel,
                )

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
        # Scheduled reports (weekly summaries, status reports)
        elif payload.get("type") == "scheduled_report" or data.get("action") == "scheduled_report":
            if is_feature_enabled("scheduled_reports"):
                user_id = data.get("user_id") or payload.get("user_id", "default")
                channel = data.get("channel") or payload.get("channel") or "web"
                try:
                    report = await self._generate_scheduled_report(user_id, payload)
                    if report:
                        await self._emit_proactive(report, user_id, channel)
                except Exception as e:
                    logger.error("Scheduled report failed", error=str(e))

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

    async def _generate_scheduled_report(self, user_id: str, payload: dict) -> str:
        """Generate a scheduled report (weekly summary, status, etc.)."""
        report_type = payload.get("report_type", "week总结")
        parts: list[str] = []
        if self._scheduler:
            jobs = self._scheduler.list_jobs()
            active = [j for j in jobs if j.get("enabled", True)]
            parts.append(f"Scheduled jobs: {len(active)} active")
        if self._skill_registry:
            try:
                from ..core.usage_tracker import get_usage_tracker
                stats = get_usage_tracker().get_stats()
                parts.append(f"API calls this period: {stats.get('total_calls', 0)}")
                parts.append(f"Est. cost: ${stats.get('cost_estimate_usd', 0):.4f}")
            except Exception:
                pass
        if not parts:
            return f"Scheduled report ({report_type}): No data to report."
        return f"**{report_type}**\n\n" + "\n".join(parts)

    async def check_followups(self, user_id: str) -> list[str]:
        """Check for unresolved action items from recent conversations."""
        if not self.settings.proactive.follow_up_tracking:
            return []
        # Would integrate with conversation summarizer action items
        return []

    def _get_time_of_day_context(self) -> dict[str, Any]:
        """Return time-of-day context for suggestions."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday()
        is_weekend = weekday >= 5
        return {
            "hour": hour,
            "weekday": weekday,
            "is_weekend": is_weekend,
            "period": "morning" if 5 <= hour < 12 else "afternoon" if 12 <= hour < 17 else "evening",
        }

    async def _get_context_reminders(self, user_id: str, context: dict) -> list[str]:
        """Get reminders based on current context (channel, app, etc.)."""
        reminders = []
        ctx_str = str(context).lower()
        if "slack" in ctx_str or "channel" in ctx_str:
            if self._scheduler:
                jobs = self._scheduler.list_jobs()
                due = [j for j in jobs if j.get("action") == "reminder"]
                if due:
                    reminders.append(f"You have {len(due)} reminders pending.")
        return reminders

    async def get_suggestions(self, user_id: str, context: str = "") -> list[str]:
        """Get context-aware suggestions (time-of-day aware)."""
        if not self.settings.proactive.suggestions_enabled and not is_feature_enabled("proactive_suggestions"):
            return []
        suggestions = []
        tod = self._get_time_of_day_context() if is_feature_enabled("time_of_day") else {}
        period = tod.get("period", "general")

        if period == "morning":
            suggestions.append("Ready for your morning briefing? Ask me for a summary.")
        elif period == "afternoon":
            suggestions.append("Need a quick status check? I can summarize your day.")
        elif period == "evening":
            suggestions.append("Want a wrap-up of today's tasks before you sign off?")

        if self._scheduler:
            jobs = self._scheduler.list_jobs()
            meeting_reminders = [j for j in jobs if "meeting" in str(j).lower()]
            if meeting_reminders:
                suggestions.append("You have meetings scheduled. Want me to prep your agenda?")

        return suggestions[:3]

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
