"""Built-in specialist agents: Research, Coding, Data."""

from .agents import BaseAgent, AgentStep, StepResult, AgentResult
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ResearchAgent(BaseAgent):
    """Agent for web research and summarization."""

    async def plan(self, task: str) -> list[AgentStep]:
        return [
            AgentStep(description=f"Search the web for: {task}", skill="browser", capability="search", args={"query": task}),
            AgentStep(description="Extract and summarize findings", skill="browser", capability="extract_text", args={}),
        ]

    async def execute_step(self, step: AgentStep) -> StepResult:
        if not self._skill_registry:
            return StepResult(success=False, error="No skill registry")
        skill = self._skill_registry.get_skill(step.skill)
        if not skill or not skill.enabled:
            return StepResult(success=False, error=f"Skill {step.skill} not available")
        try:
            result = await skill.execute(step.capability, **step.args)
            return StepResult(success=result.success, output=result.output, error=result.error)
        except Exception as e:
            return StepResult(success=False, error=str(e))


class CodingAgent(BaseAgent):
    """Agent for writing and running code."""

    async def plan(self, task: str) -> list[AgentStep]:
        return [
            AgentStep(description=f"Write code for: {task}", skill="filesystem", capability="write_file", args={}),
            AgentStep(description="Execute the code", skill="shell", capability="execute", args={}),
        ]

    async def execute_step(self, step: AgentStep) -> StepResult:
        if not self._skill_registry:
            return StepResult(success=False, error="No skill registry")
        skill = self._skill_registry.get_skill(step.skill)
        if not skill or not skill.enabled:
            return StepResult(success=False, error=f"Skill {step.skill} not available")
        try:
            result = await skill.execute(step.capability, **step.args)
            return StepResult(success=result.success, output=result.output, error=result.error)
        except Exception as e:
            return StepResult(success=False, error=str(e))


class DataAgent(BaseAgent):
    """Agent for data analysis."""

    async def plan(self, task: str) -> list[AgentStep]:
        return [
            AgentStep(description=f"Analyze data: {task}", skill="documents", capability="extract_text", args={}),
            AgentStep(description="Run analysis", skill="shell", capability="execute", args={}),
        ]

    async def execute_step(self, step: AgentStep) -> StepResult:
        if not self._skill_registry:
            return StepResult(success=False, error="No skill registry")
        skill = self._skill_registry.get_skill(step.skill)
        if not skill or not skill.enabled:
            return StepResult(success=False, error=f"Skill {step.skill} not available")
        try:
            result = await skill.execute(step.capability, **step.args)
            return StepResult(success=result.success, output=result.output, error=result.error)
        except Exception as e:
            return StepResult(success=False, error=str(e))
