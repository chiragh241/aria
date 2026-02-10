"""Workflow skill — chain multiple skills in sequence."""

from datetime import datetime, timezone
from typing import Any

from ..base import BaseSkill, SkillResult
from ...utils.logging import get_logger

logger = get_logger(__name__)


class WorkflowSkill(BaseSkill):
    """Skill that chains multiple skills: research → draft → email, etc."""

    name = "workflow"
    description = "Chain skills together: research → draft → email, etc."
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._engine = None

    def set_engine(self, engine: Any) -> None:
        self._engine = engine

    def _register_capabilities(self) -> None:
        self.register_capability(
            name="run_chain",
            description="Execute a chain of skills. Each step runs after the previous; use {{step_0.output}} in args to pass prior output. Example: research → summarize → email.",
            parameters={
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "description": "List of {skill, capability, args}",
                        "items": {
                            "type": "object",
                            "properties": {
                                "skill": {"type": "string"},
                                "capability": {"type": "string"},
                                "args": {"type": "object"},
                            },
                        },
                    },
                    "channel": {"type": "string"},
                    "user_id": {"type": "string"},
                },
                "required": ["steps"],
            },
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        start = datetime.now(timezone.utc)
        if capability != "run_chain":
            return self._error_result(f"Unknown capability: {capability}", start)

        if not self._engine:
            return self._error_result("Workflow engine not configured", start)

        steps = kwargs.get("steps", [])
        if not steps:
            return self._error_result("steps required", start)

        channel = kwargs.get("channel", "web")
        user_id = kwargs.get("user_id", "default")

        result = await self._engine.run_chain(
            steps=steps,
            channel=channel,
            user_id=user_id,
        )

        if result.get("success"):
            return self._success_result(
                result.get("final_output"),
                start,
                metadata={"step_count": result.get("step_count", 0)},
            )
        return self._error_result(
            result.get("error", "Chain failed"),
            start,
        )
