"""Base agent framework for autonomous multi-step tasks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class AgentStep:
    """A single step in an agent's plan."""

    description: str
    skill: str = ""
    capability: str = ""
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of executing one step."""

    success: bool
    output: Any = None
    error: str | None = None


@dataclass
class BotStatus:
    """Status of a single bot in a multi-bot task."""

    id: str = ""
    name: str = ""
    source: str = ""  # reddit, web, x, etc.
    status: str = "pending"  # pending, running, completed, failed
    output: str = ""
    error: str = ""
    started_at: str = ""
    completed_at: str = ""


@dataclass
class AgentTask:
    """An agent task with status, steps, and optional parallel bots."""

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    description: str = ""
    status: str = "pending"  # pending, running, completed, failed
    steps: list[AgentStep] = field(default_factory=list)
    results: list[StepResult] = field(default_factory=list)
    final_result: str = ""
    parent_task_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Multi-bot: parallel workers (Reddit bot, Web bot, X bot, etc.)
    bots: list[BotStatus] = field(default_factory=list)


@dataclass
class AgentResult:
    """Result of running an agent."""

    success: bool
    output: str = ""
    steps_completed: int = 0
    error: str | None = None
    task_id: str = ""


class BaseAgent(ABC):
    """Base class for autonomous agents."""

    max_iterations: int = 10

    def __init__(self, skill_registry: Any, llm_router: Any, shared_context: dict[str, Any] | None = None) -> None:
        self._skill_registry = skill_registry
        self._llm_router = llm_router
        self._shared_context = shared_context or {}

    @abstractmethod
    async def plan(self, task: str) -> list[AgentStep]:
        """Break task into steps."""
        pass

    @abstractmethod
    async def execute_step(self, step: AgentStep) -> StepResult:
        """Execute a single step."""
        pass

    async def run(self, task: str) -> AgentResult:
        """Plan and execute with self-correction loop."""
        steps = await self.plan(task)
        if not steps:
            return AgentResult(success=False, output="Could not create plan", error="No steps")

        results: list[StepResult] = []
        for i, step in enumerate(steps[: self.max_iterations]):
            result = await self.execute_step(step)
            results.append(result)
            if not result.success:
                return AgentResult(
                    success=False,
                    output=result.error or "Step failed",
                    steps_completed=i,
                    error=result.error,
                )

        return AgentResult(
            success=True,
            output=results[-1].output if results else "",
            steps_completed=len(results),
        )
