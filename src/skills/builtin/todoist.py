"""Todoist integration skill — manage tasks from chat."""

import os
from typing import Any

from ..base import BaseSkill, SkillResult
from ...features.registry import is_feature_enabled
from ...utils.logging import get_logger

logger = get_logger(__name__)


class TodoistSkill(BaseSkill):
    """Create and manage Todoist tasks. Requires TODOIST_API_KEY."""

    name = "todoist"
    description = "Create tasks, add to projects, set due dates in Todoist"
    version = "1.0.0"
    enabled = False

    def _get_api_key(self) -> str | None:
        return self.config.get("api_key") or os.environ.get("TODOIST_API_KEY")

    def _register_capabilities(self) -> None:
        self.register_capability(
            name="add_task",
            description="Add a task to Todoist",
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Task content"},
                    "project_id": {"type": "string", "description": "Optional project ID"},
                    "due": {"type": "string", "description": "Due date (e.g. today, tomorrow)"},
                },
                "required": ["content"],
            },
        )
        self.register_capability(
            name="list_tasks",
            description="List tasks (filter by project or label)",
            parameters={
                "type": "object",
                "properties": {"project_id": {"type": "string"}, "label": {"type": "string"}},
            },
        )
        self.register_capability(
            name="complete_task",
            description="Mark a task as complete",
            parameters={"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]},
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        if not is_feature_enabled("todoist_integration"):
            return SkillResult(success=False, error="Todoist integration is disabled")
        if not self._get_api_key():
            return SkillResult(
                success=False,
                error="Todoist API key not configured. Add TODOIST_API_KEY or configure in settings.",
            )
        return SkillResult(
            success=True,
            output="Todoist integration is configured. Full implementation coming soon.",
        )
