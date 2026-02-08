"""Linear integration skill — manage issues from chat."""

import os
from typing import Any

from ..base import BaseSkill, SkillResult
from ...features.registry import is_feature_enabled
from ...utils.logging import get_logger

logger = get_logger(__name__)


class LinearSkill(BaseSkill):
    """Create and manage Linear issues. Requires LINEAR_API_KEY."""

    name = "linear"
    description = "Create issues, assign, update status in Linear"
    version = "1.0.0"
    enabled = False

    def _get_api_key(self) -> str | None:
        return self.config.get("api_key") or os.environ.get("LINEAR_API_KEY")

    def _register_capabilities(self) -> None:
        self.register_capability(
            name="create_issue",
            description="Create a new Linear issue",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "team_id": {"type": "string"},
                    "assignee_id": {"type": "string"},
                },
                "required": ["title"],
            },
        )
        self.register_capability(
            name="list_issues",
            description="List issues (filter by team, status)",
            parameters={
                "type": "object",
                "properties": {"team_id": {"type": "string"}, "status": {"type": "string"}},
            },
        )
        self.register_capability(
            name="update_issue",
            description="Update issue status or assignee",
            parameters={
                "type": "object",
                "properties": {"issue_id": {"type": "string"}, "status": {"type": "string"}},
                "required": ["issue_id"],
            },
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        if not is_feature_enabled("linear_integration"):
            return SkillResult(success=False, error="Linear integration is disabled")
        if not self._get_api_key():
            return SkillResult(
                success=False,
                error="Linear API key not configured. Add LINEAR_API_KEY or configure in settings.",
            )
        return SkillResult(
            success=True,
            output="Linear integration is configured. Full implementation coming soon.",
        )
