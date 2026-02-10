"""Agent skill — routes to autonomous research, coding, and data agents."""

from datetime import datetime, timezone
from typing import Any

from ..base import BaseSkill, SkillResult
from ...utils.logging import get_logger

logger = get_logger(__name__)


class AgentSkill(BaseSkill):
    """Skill that delegates to autonomous agents for research, coding, analysis."""

    name = "agent"
    description = "Autonomous research, coding, and data analysis"
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._coordinator = None

    def set_coordinator(self, coordinator: Any) -> None:
        self._coordinator = coordinator

    def _register_capabilities(self) -> None:
        self.register_capability(
            name="research",
            description="Research a topic and provide a summary. Use for 'research X', 'find information about Y'.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to research"},
                    "user_id": {"type": "string"},
                    "channel": {"type": "string"},
                },
                "required": ["query"],
            },
        )
        self.register_capability(
            name="code",
            description="Write and run code to accomplish a task.",
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "What the code should do"},
                    "user_id": {"type": "string"},
                    "channel": {"type": "string"},
                },
                "required": ["task"],
            },
        )
        self.register_capability(
            name="analyze",
            description="Analyze data and find patterns.",
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "What to analyze"},
                    "user_id": {"type": "string"},
                    "channel": {"type": "string"},
                },
                "required": ["task"],
            },
        )
        self.register_capability(
            name="automate",
            description="Execute a multi-step task autonomously.",
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The task to perform"},
                    "user_id": {"type": "string"},
                    "channel": {"type": "string"},
                },
                "required": ["task"],
            },
        )
        self.register_capability(
            name="itinerary",
            description="Create multi-city or multi-country itinerary. Use for 'itinerary for Paris, London, Tokyo', 'trip to Japan and Korea', etc. Runs bots in parallel per destination.",
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "e.g. 'Create itinerary for Paris, London, Tokyo'"},
                    "user_id": {"type": "string"},
                    "channel": {"type": "string"},
                },
                "required": ["task"],
            },
        )
        self.register_capability(
            name="handoff",
            description="Hand off from one agent to another. E.g. research then hand off to coding. Use when a task needs multiple specialist agents in sequence.",
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The overall task"},
                    "first_agent": {"type": "string", "description": "research|coding|data"},
                    "then_agent": {"type": "string", "description": "research|coding|data"},
                    "user_id": {"type": "string"},
                    "channel": {"type": "string"},
                },
                "required": ["task", "first_agent", "then_agent"],
            },
        )
        self.register_capability(
            name="list_agents",
            description="List running and recent agent tasks. Use to check status of delegated work.",
            parameters={"type": "object", "properties": {"user_id": {"type": "string"}, "channel": {"type": "string"}}},
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        start = datetime.now(timezone.utc)
        if not self._coordinator:
            return self._error_result("Agent coordinator not configured", start)

        if capability == "list_agents":
            try:
                running = self._coordinator.list_running_agents()
                all_tasks = self._coordinator.list_all_agents(include_completed=True)
                return self._success_result(
                    {"running": running, "recent": all_tasks[-10:]},
                    start,
                )
            except Exception as e:
                return self._error_result(str(e), start)

        task = kwargs.get("query") or kwargs.get("task", "")
        if not task:
            return self._error_result("Task or query required", start)

        user_id = kwargs.get("user_id", "default")
        channel = kwargs.get("channel", "")

        if capability == "handoff":
            first = kwargs.get("first_agent", "research")
            then = kwargs.get("then_agent", "coding")
            map_agent = {"research": "research", "coding": "coding", "code": "coding", "data": "data", "analyze": "data"}
            first_type = map_agent.get(first, "research")
            then_type = map_agent.get(then, "coding")
            try:
                r1 = await self._coordinator.delegate(task=task, agent_type=first_type, user_id=user_id, channel=channel)
                if not r1.success:
                    return self._error_result(r1.error or "First agent failed", start)
                follow_up = f"Based on this research:\n{r1.output[:2000]}\n\nNow: {task}"
                r2 = await self._coordinator.delegate(task=follow_up, agent_type=then_type, user_id=user_id, channel=channel)
                if r2.success:
                    return self._success_result(
                        f"**{first} agent:**\n{r1.output[:1000]}\n\n**{then} agent:**\n{r2.output}",
                        start,
                    )
                return self._error_result(r2.error or "Second agent failed", start)
            except Exception as e:
                logger.error("Agent handoff failed", error=str(e))
                return self._error_result(str(e), start)

        agent_type = {"research": "research", "code": "coding", "analyze": "data", "automate": None, "itinerary": None}.get(capability)

        try:
            # Itinerary / multi-destination: decompose and run parallel bots
            if capability == "itinerary" and hasattr(self._coordinator, "delegate_parallel_subtasks"):
                result = await self._coordinator.delegate_parallel_subtasks(
                    task=task,
                    user_id=user_id,
                    channel=channel,
                )
            # Automate: try parallel subtasks first (multi-city, etc.), fallback to single agent
            elif capability == "automate" and hasattr(self._coordinator, "delegate_parallel_subtasks"):
                result = await self._coordinator.delegate_parallel_subtasks(
                    task=task,
                    user_id=user_id,
                    channel=channel,
                )
            # Research: multi-bot (Reddit + Web + X in parallel)
            elif capability == "research" and hasattr(self._coordinator, "delegate_multi_bot"):
                result = await self._coordinator.delegate_multi_bot(
                    task=task,
                    user_id=user_id,
                    channel=channel,
                )
            else:
                result = await self._coordinator.delegate(
                    task=task,
                    agent_type=agent_type,
                    user_id=user_id,
                    channel=channel,
                )
            if result.success:
                return self._success_result(result.output or "Task completed.", start)
            return self._error_result(result.error or result.output or "Agent failed", start)
        except Exception as e:
            logger.error("Agent execution failed", capability=capability, error=str(e))
            return self._error_result(str(e), start)
