"""Central registry of Aria features with enable/disable and config."""

from dataclasses import dataclass, field
from typing import Any

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureDef:
    """Definition of a feature."""

    id: str
    name: str
    description: str
    category: str
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)


# Feature registry
FEATURES: dict[str, FeatureDef] = {
    "morning_briefing": FeatureDef(
        id="morning_briefing",
        name="Morning Briefing",
        description="Scheduled summary of calendar, weather, news, reminders",
        category="proactive",
        enabled=True,
    ),
    "context_reminders": FeatureDef(
        id="context_reminders",
        name="Context-Aware Reminders",
        description="Remind when in Slack channel, opening repo, etc.",
        category="proactive",
        enabled=True,
    ),
    "proactive_suggestions": FeatureDef(
        id="proactive_suggestions",
        name="Proactive Suggestions",
        description="Suggest before meetings, API expiry, etc.",
        category="proactive",
        enabled=True,
    ),
    "time_of_day": FeatureDef(
        id="time_of_day",
        name="Time-of-Day Awareness",
        description="Different behavior for morning vs evening, work vs weekend",
        category="proactive",
        enabled=True,
    ),
    "notion_integration": FeatureDef(
        id="notion_integration",
        name="Notion Integration",
        description="Sync notes, create pages, query knowledge bases",
        category="integrations",
        enabled=False,
        config={"api_key": ""},
    ),
    "todoist_integration": FeatureDef(
        id="todoist_integration",
        name="Todoist Integration",
        description="Manage tasks from chat",
        category="integrations",
        enabled=False,
        config={"api_key": ""},
    ),
    "linear_integration": FeatureDef(
        id="linear_integration",
        name="Linear Integration",
        description="Manage issues from chat",
        category="integrations",
        enabled=False,
        config={"api_key": ""},
    ),
    "spotify_integration": FeatureDef(
        id="spotify_integration",
        name="Spotify Integration",
        description="Control playback, suggest playlists",
        category="integrations",
        enabled=False,
        config={"client_id": "", "client_secret": ""},
    ),
    "remember_forget": FeatureDef(
        id="remember_forget",
        name="Remember / Forget Commands",
        description="Explicit memory control",
        category="memory",
        enabled=True,
    ),
    "conversation_summarization": FeatureDef(
        id="conversation_summarization",
        name="Conversation Summarization",
        description="Summaries of long threads for context",
        category="memory",
        enabled=True,
    ),
    "learning_from_corrections": FeatureDef(
        id="learning_from_corrections",
        name="Learning from Corrections",
        description="Track user corrections and adjust behavior",
        category="memory",
        enabled=True,
    ),
    "cross_session_context": FeatureDef(
        id="cross_session_context",
        name="Cross-Session Context",
        description="Reuse last-chat context when starting new session",
        category="memory",
        enabled=True,
    ),
    "cost_tracking": FeatureDef(
        id="cost_tracking",
        name="Cost / Usage Dashboard",
        description="API call and token usage vs budget",
        category="dashboard",
        enabled=True,
    ),
    "latency_metrics": FeatureDef(
        id="latency_metrics",
        name="Latency Metrics",
        description="Response times and reliability",
        category="dashboard",
        enabled=True,
    ),
    "custom_widgets": FeatureDef(
        id="custom_widgets",
        name="Custom Widgets",
        description="User-defined dashboard widgets",
        category="dashboard",
        enabled=True,
    ),
    "theme_toggle": FeatureDef(
        id="theme_toggle",
        name="Theme Toggle",
        description="Light / dark and accent colors",
        category="dashboard",
        enabled=True,
    ),
    "data_export": FeatureDef(
        id="data_export",
        name="Data Export",
        description="Export data, audit logs, conversation history",
        category="dashboard",
        enabled=True,
    ),
    "recurring_tasks": FeatureDef(
        id="recurring_tasks",
        name="Recurring Tasks",
        description="e.g. Summarize inbox every Monday",
        category="workflows",
        enabled=True,
    ),
    "custom_workflows": FeatureDef(
        id="custom_workflows",
        name="Custom Workflows",
        description="Chain skills (research → draft → email)",
        category="workflows",
        enabled=True,
    ),
    "webhooks": FeatureDef(
        id="webhooks",
        name="Webhooks",
        description="Inbound events from external systems",
        category="workflows",
        enabled=True,
    ),
    "scheduled_reports": FeatureDef(
        id="scheduled_reports",
        name="Scheduled Reports",
        description="Weekly summaries, status reports",
        category="workflows",
        enabled=True,
    ),
    "pwa": FeatureDef(
        id="pwa",
        name="Progressive Web App",
        description="Installable app, offline support, shortcuts",
        category="mobile",
        enabled=True,
    ),
    "push_notifications": FeatureDef(
        id="push_notifications",
        name="Push Notifications",
        description="Alerts for approvals, reminders, important events",
        category="mobile",
        enabled=True,
    ),
    "keyboard_shortcuts": FeatureDef(
        id="keyboard_shortcuts",
        name="Keyboard Shortcuts",
        description="Power-user navigation and actions",
        category="mobile",
        enabled=True,
    ),
    "voice_first": FeatureDef(
        id="voice_first",
        name="Voice-First Mode",
        description="Hands-free interaction suitable for mobile",
        category="mobile",
        enabled=True,
    ),
    "permission_levels": FeatureDef(
        id="permission_levels",
        name="Permission Levels",
        description="Control who can use which skills or data",
        category="collaboration",
        enabled=True,
    ),
    "delegation": FeatureDef(
        id="delegation",
        name="Multi-Agent Delegation",
        description="Hand off between agents (research → coding, etc.), list running agents",
        category="collaboration",
        enabled=True,
    ),
    "skill_chaining": FeatureDef(
        id="skill_chaining",
        name="Skill Chaining",
        description="Chain skills: research → draft → email in one workflow",
        category="workflows",
        enabled=True,
    ),
    "api_docs": FeatureDef(
        id="api_docs",
        name="Public API Docs",
        description="OpenAPI/Swagger for external integrations",
        category="developer",
        enabled=True,
    ),
    "skill_templates": FeatureDef(
        id="skill_templates",
        name="Skill Templates",
        description="Quick-start templates for new skills",
        category="developer",
        enabled=True,
    ),
    "debug_trace": FeatureDef(
        id="debug_trace",
        name="Debug / Trace Mode",
        description="Inspect LLM calls, tool use, reasoning",
        category="developer",
        enabled=True,
    ),
}


def get_feature(feature_id: str) -> FeatureDef | None:
    """Get feature definition by ID."""
    return FEATURES.get(feature_id)


def is_feature_enabled(feature_id: str) -> bool:
    """Check if a feature is enabled (from config or default)."""
    feat = FEATURES.get(feature_id)
    if not feat:
        return False
    try:
        settings = get_settings()
        overrides = getattr(settings, "feature_overrides", None) or {}
        if isinstance(overrides, dict) and feature_id in overrides:
            return bool(overrides[feature_id])
    except Exception:
        pass
    return feat.enabled
