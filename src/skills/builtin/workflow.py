"""Workflow skill — chain skills in sequence and run named workflows."""

from datetime import datetime, timezone
from typing import Any

from ..base import BaseSkill, SkillResult
from ...utils.logging import get_logger

logger = get_logger(__name__)


class WorkflowSkill(BaseSkill):
    """Skill that chains skills (run_chain) and runs named workflows (list_workflows, run_named_workflow)."""

    name = "workflow"
    description = "Chain skills together or run a named workflow (e.g. feature-dev, bug-fix)."
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._engine = None
        self._named_workflow_runner = None

    def set_engine(self, engine: Any) -> None:
        self._engine = engine

    def set_named_workflow_runner(self, runner: Any) -> None:
        """Set the runner for named workflows."""
        self._named_workflow_runner = runner

    def _register_capabilities(self) -> None:
        self.register_capability(
            name="run_chain",
            description="Execute a chain of skills. Each step runs after the previous; use {{step_0.output}} in args to pass prior output.",
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
        self.register_capability(
            name="list_workflows",
            description="List available named workflows (e.g. feature-dev, bug-fix). Use before run_named_workflow to see ids and descriptions.",
            parameters={
                "type": "object",
                "properties": {"user_id": {"type": "string"}, "channel": {"type": "string"}},
            },
        )
        self.register_capability(
            name="run_named_workflow",
            description=(
                "Run a named workflow for a task. Workflows run steps in a fixed order (e.g. plan → implement → verify) "
                "with specialist agents. Use list_workflows to see available workflow ids (e.g. feature-dev, bug-fix)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string", "description": "Workflow id from list_workflows (e.g. feature-dev, bug-fix)"},
                    "task": {"type": "string", "description": "The task or request to run the workflow for"},
                    "user_id": {"type": "string"},
                    "channel": {"type": "string"},
                },
                "required": ["workflow_id", "task"],
            },
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        start = datetime.now(timezone.utc)

        if capability == "list_workflows":
            if not self._named_workflow_runner:
                return self._error_result("Named workflows not configured", start)
            try:
                workflows = self._named_workflow_runner.list_workflows()
                return self._success_result(
                    {"workflows": workflows},
                    start,
                    metadata={"count": len(workflows)},
                )
            except Exception as e:
                logger.exception("list_workflows failed")
                return self._error_result(str(e), start)

        if capability == "run_named_workflow":
            if not self._named_workflow_runner:
                return self._error_result("Named workflows not configured", start)
            workflow_id = (kwargs.get("workflow_id") or "").strip()
            task = (kwargs.get("task") or "").strip()
            if not workflow_id or not task:
                return self._error_result("workflow_id and task are required", start)
            user_id = kwargs.get("user_id", "default")
            channel = kwargs.get("channel", "web")
            try:
                result = await self._named_workflow_runner.run(
                    workflow_id=workflow_id,
                    task=task,
                    user_id=user_id,
                    channel=channel,
                )
                if result.get("success"):
                    return self._success_result(
                        result.get("final_output", ""),
                        start,
                        metadata={"step_outputs_count": len(result.get("step_outputs", []))},
                    )
                return self._error_result(result.get("error", "Workflow failed"), start)
            except Exception as e:
                logger.exception("run_named_workflow failed")
                return self._error_result(str(e), start)

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
