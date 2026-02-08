"""Coordinates autonomous agents — delegate, run, track."""

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .agents import AgentResult, AgentTask, BotStatus
from .agents_builtin import ResearchAgent, CodingAgent, DataAgent
from .task_decomposer import decompose, DecompositionResult, Subtask
from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class AgentCoordinator:
    """Manages agent tasks — delegates to specialists, tracks running tasks."""

    def __init__(
        self,
        skill_registry: Any = None,
        llm_router: Any = None,
        event_bus: Any = None,
    ) -> None:
        self.settings = get_settings()
        self._skill_registry = skill_registry
        self._llm_router = llm_router
        self._event_bus = event_bus
        self._tasks: dict[str, AgentTask] = {}
        self._lock = asyncio.Lock()

    def set_skill_registry(self, sr: Any) -> None:
        self._skill_registry = sr

    def set_llm_router(self, lr: Any) -> None:
        self._llm_router = lr

    def set_event_bus(self, eb: Any) -> None:
        self._event_bus = eb

    async def delegate(
        self,
        task: str,
        agent_type: str | None = None,
        user_id: str = "default",
        channel: str = "",
    ) -> AgentResult:
        """Delegate a task to the appropriate agent."""
        if not self.settings.agents.enabled:
            return AgentResult(success=False, output="Agents disabled", error="disabled")

        agent = self._select_agent(task, agent_type)
        if not agent:
            return AgentResult(success=False, output="No suitable agent", error="no_agent")

        task_obj = AgentTask(description=task, status="running")
        async with self._lock:
            self._tasks[task_obj.id] = task_obj

        if self._event_bus:
            await self._event_bus.emit("agent_started", {
                "task_id": task_obj.id,
                "task": task,
                "user_id": user_id,
                "channel": channel,
            }, source="agent_coordinator")

        try:
            result = await agent.run(task)
            task_obj.status = "completed" if result.success else "failed"
            task_obj.final_result = result.output

            if self._event_bus:
                await self._event_bus.emit("agent_completed", {
                    "task_id": task_obj.id,
                    "success": result.success,
                    "user_id": user_id,
                    "channel": channel,
                }, source="agent_coordinator")

            return AgentResult(
                success=result.success,
                output=result.output,
                steps_completed=result.steps_completed,
                error=result.error,
                task_id=task_obj.id,
            )
        except Exception as e:
            task_obj.status = "failed"
            logger.error("Agent failed", task_id=task_obj.id, error=str(e))
            return AgentResult(success=False, output="", error=str(e), task_id=task_obj.id)
        finally:
            async with self._lock:
                self._tasks[task_obj.id] = task_obj

    def _select_agent(self, task: str, agent_type: str | None) -> Any:
        """Select agent based on task or explicit type."""
        task_lower = task.lower()
        if agent_type:
            if agent_type == "research" and self.settings.agents.research_enabled:
                return ResearchAgent(self._skill_registry, self._llm_router)
            if agent_type == "coding" and self.settings.agents.coding_enabled:
                return CodingAgent(self._skill_registry, self._llm_router)
            if agent_type == "data" and self.settings.agents.data_enabled:
                return DataAgent(self._skill_registry, self._llm_router)

        if any(kw in task_lower for kw in ["research", "find", "search", "look up"]) and self.settings.agents.research_enabled:
            return ResearchAgent(self._skill_registry, self._llm_router)
        if any(kw in task_lower for kw in ["code", "script", "write", "program"]) and self.settings.agents.coding_enabled:
            return CodingAgent(self._skill_registry, self._llm_router)
        if any(kw in task_lower for kw in ["analyze", "data", "pattern"]) and self.settings.agents.data_enabled:
            return DataAgent(self._skill_registry, self._llm_router)

        return ResearchAgent(self._skill_registry, self._llm_router)

    def list_running_agents(self) -> list[dict[str, Any]]:
        """List active agent tasks."""
        return [
            {
                "id": t.id,
                "description": t.description,
                "status": t.status,
                "created_at": t.created_at,
            }
            for t in self._tasks.values()
            if t.status in ("pending", "running")
        ]

    def list_all_agents(self, include_completed: bool = True) -> list[dict[str, Any]]:
        """List all agent tasks with bot status for dashboard."""
        result = []
        for t in self._tasks.values():
            if not include_completed and t.status not in ("pending", "running"):
                continue
            task_dict = {
                "id": t.id,
                "description": t.description,
                "status": t.status,
                "created_at": t.created_at,
                "final_result": t.final_result[:200] + "..." if len(t.final_result or "") > 200 else t.final_result,
            }
            if t.bots:
                task_dict["bots"] = [
                    {
                        "id": b.id,
                        "name": b.name,
                        "source": b.source,
                        "status": b.status,
                        "output": (b.output or "")[:100] + "..." if len(b.output or "") > 100 else (b.output or ""),
                        "error": b.error,
                    }
                    for b in t.bots
                ]
            result.append(task_dict)
        return sorted(result, key=lambda x: x["created_at"], reverse=True)[:50]

    def get_task(self, task_id: str) -> AgentTask | None:
        return self._tasks.get(task_id)

    async def delegate_multi_bot(
        self,
        task: str,
        user_id: str = "default",
        channel: str = "",
    ) -> AgentResult:
        """
        Run multi-bot research: Reddit, Web, and X bots in parallel.
        Each bot runs independently; results are merged.
        """
        if not self.settings.agents.enabled:
            return AgentResult(success=False, output="Agents disabled", error="disabled")

        research_skill = self._skill_registry.get_skill("research") if self._skill_registry else None
        if not research_skill or not research_skill.enabled:
            # Fallback to single ResearchAgent
            return await self.delegate(task, agent_type="research", user_id=user_id, channel=channel)

        task_obj = AgentTask(description=task, status="running")
        sources = [
            ("reddit", "Reddit Bot", "search_reddit"),
            ("web", "Web Bot", "search_web"),
            ("x", "X Bot", "search_x"),
        ]
        task_obj.bots = [
            BotStatus(id=f"{task_obj.id}-{s[0]}", name=s[1], source=s[0], status="pending", started_at="")
            for s in sources
        ]

        async with self._lock:
            self._tasks[task_obj.id] = task_obj

        if self._event_bus:
            await self._event_bus.emit("agent_started", {
                "task_id": task_obj.id,
                "task": task,
                "user_id": user_id,
                "channel": channel,
                "multi_bot": True,
                "bots": [{"id": b.id, "name": b.name, "source": b.source} for b in task_obj.bots],
            }, source="agent_coordinator")

        async def run_bot(idx: int, source: str, name: str, capability: str) -> tuple[str, str, str]:
            """Run one bot and return (source, output, error)."""
            now = datetime.now(timezone.utc).isoformat()
            async with self._lock:
                if idx < len(task_obj.bots):
                    task_obj.bots[idx].status = "running"
                    task_obj.bots[idx].started_at = now
            if self._event_bus:
                await self._event_bus.emit("agent_bot_progress", {
                    "task_id": task_obj.id,
                    "bot_id": task_obj.bots[idx].id if idx < len(task_obj.bots) else "",
                    "source": source,
                    "status": "running",
                }, source="agent_coordinator")

            try:
                result = await research_skill.execute(capability, query=task, limit=5)
                output = str(result.output or "") if result.success else ""
                error = result.error or ""
            except Exception as e:
                output = ""
                error = str(e)

            completed_at = datetime.now(timezone.utc).isoformat()
            status = "completed" if not error else "failed"
            async with self._lock:
                if idx < len(task_obj.bots):
                    task_obj.bots[idx].status = status
                    task_obj.bots[idx].output = output
                    task_obj.bots[idx].error = error
                    task_obj.bots[idx].completed_at = completed_at

            if self._event_bus:
                await self._event_bus.emit("agent_bot_progress", {
                    "task_id": task_obj.id,
                    "bot_id": task_obj.bots[idx].id if idx < len(task_obj.bots) else "",
                    "source": source,
                    "status": status,
                    "output_preview": (output or error)[:100],
                }, source="agent_coordinator")
            return source, output, error

        try:
            tasks_coros = [run_bot(i, s[0], s[1], s[2]) for i, s in enumerate(sources)]
            results = await asyncio.gather(*tasks_coros, return_exceptions=True)

            parts = [f"## Research: {task}\n"]
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    source = sources[i][0]
                    parts.append(f"### {source.upper()} Bot\nError: {r}\n")
                else:
                    src, out, err = r
                    parts.append(f"### {src.upper()} Bot\n{out or err or '(no output)'}\n")

            task_obj.status = "completed"
            task_obj.final_result = "\n".join(parts)

            if self._event_bus:
                await self._event_bus.emit("agent_completed", {
                    "task_id": task_obj.id,
                    "success": True,
                    "user_id": user_id,
                    "channel": channel,
                }, source="agent_coordinator")

            return AgentResult(
                success=True,
                output=task_obj.final_result,
                steps_completed=len(sources),
                task_id=task_obj.id,
            )
        except Exception as e:
            task_obj.status = "failed"
            logger.error("Multi-bot research failed", task_id=task_obj.id, error=str(e))
            return AgentResult(success=False, output="", error=str(e), task_id=task_obj.id)
        finally:
            async with self._lock:
                self._tasks[task_obj.id] = task_obj

    async def delegate_parallel_subtasks(
        self,
        task: str,
        user_id: str = "default",
        channel: str = "",
    ) -> AgentResult:
        """
        Decompose large tasks (e.g. multi-city itinerary) into subtasks and run
        them in parallel with multiple bots.
        """
        if not self.settings.agents.enabled:
            return AgentResult(success=False, output="Agents disabled", error="disabled")

        decomp = decompose(task, self._llm_router)
        if not decomp.can_parallelize:
            return await self.delegate(task, agent_type="research", user_id=user_id, channel=channel)

        research_skill = self._skill_registry.get_skill("research") if self._skill_registry else None
        if not research_skill or not research_skill.enabled:
            agent = ResearchAgent(self._skill_registry, self._llm_router)
            return await self._run_subtasks_via_agent(task, decomp, agent, user_id, channel)

        return await self._run_parallel_subtasks(task, decomp, research_skill, user_id, channel)

    async def _run_parallel_subtasks(
        self,
        task: str,
        decomp: DecompositionResult,
        research_skill: Any,
        user_id: str,
        channel: str,
    ) -> AgentResult:
        """Run decomposed subtasks in parallel using research skill."""
        task_obj = AgentTask(description=task, status="running")
        task_obj.bots = [
            BotStatus(
                id=f"{task_obj.id}-{i}",
                name=f"{st.entity or f'Bot {i}'}",
                source=st.entity or str(i),
                status="pending",
                started_at="",
            )
            for i, st in enumerate(decomp.subtasks)
        ]

        async with self._lock:
            self._tasks[task_obj.id] = task_obj

        if self._event_bus:
            await self._event_bus.emit("agent_started", {
                "task_id": task_obj.id,
                "task": task,
                "user_id": user_id,
                "channel": channel,
                "multi_bot": True,
                "decomposed": True,
                "bots": [{"id": b.id, "name": b.name, "source": b.source} for b in task_obj.bots],
            }, source="agent_coordinator")

        async def run_bot(idx: int, subtask: Subtask) -> tuple[str, str, str]:
            now = datetime.now(timezone.utc).isoformat()
            async with self._lock:
                if idx < len(task_obj.bots):
                    task_obj.bots[idx].status = "running"
                    task_obj.bots[idx].started_at = now
            if self._event_bus:
                await self._event_bus.emit("agent_bot_progress", {
                    "task_id": task_obj.id,
                    "bot_id": task_obj.bots[idx].id if idx < len(task_obj.bots) else "",
                    "source": subtask.entity,
                    "status": "running",
                }, source="agent_coordinator")

            try:
                result = await research_skill.execute(
                    "search_topic",
                    query=subtask.description,
                    sources=["reddit", "web", "x"],
                    limit=5,
                )
                output = str(result.output or "") if result.success else ""
                error = result.error or ""
            except Exception as e:
                output = ""
                error = str(e)

            completed_at = datetime.now(timezone.utc).isoformat()
            status = "completed" if not error else "failed"
            async with self._lock:
                if idx < len(task_obj.bots):
                    task_obj.bots[idx].status = status
                    task_obj.bots[idx].output = output
                    task_obj.bots[idx].error = error
                    task_obj.bots[idx].completed_at = completed_at

            if self._event_bus:
                await self._event_bus.emit("agent_bot_progress", {
                    "task_id": task_obj.id,
                    "bot_id": task_obj.bots[idx].id if idx < len(task_obj.bots) else "",
                    "source": subtask.entity,
                    "status": status,
                    "output_preview": (output or error)[:100],
                }, source="agent_coordinator")
            return subtask.entity, output, error

        try:
            tasks_coros = [run_bot(i, st) for i, st in enumerate(decomp.subtasks)]
            results = await asyncio.gather(*tasks_coros, return_exceptions=True)

            parts = [f"## {task}\n\n"]
            for i, r in enumerate(results):
                st = decomp.subtasks[i]
                if isinstance(r, Exception):
                    parts.append(f"### {st.entity}\nError: {r}\n")
                else:
                    entity, out, err = r
                    parts.append(f"### {entity}\n{out or err or '(no output)'}\n")

            task_obj.status = "completed"
            task_obj.final_result = "\n".join(parts)

            if self._event_bus:
                await self._event_bus.emit("agent_completed", {
                    "task_id": task_obj.id,
                    "success": True,
                    "user_id": user_id,
                    "channel": channel,
                }, source="agent_coordinator")

            return AgentResult(
                success=True,
                output=task_obj.final_result,
                steps_completed=len(decomp.subtasks),
                task_id=task_obj.id,
            )
        except Exception as e:
            task_obj.status = "failed"
            logger.error("Parallel subtasks failed", task_id=task_obj.id, error=str(e))
            return AgentResult(success=False, output="", error=str(e), task_id=task_obj.id)
        finally:
            async with self._lock:
                self._tasks[task_obj.id] = task_obj

    async def _run_subtasks_via_agent(
        self,
        task: str,
        decomp: DecompositionResult,
        agent: Any,
        user_id: str,
        channel: str,
    ) -> AgentResult:
        """Fallback: run subtasks via ResearchAgent when research skill unavailable."""
        task_obj = AgentTask(description=task, status="running")
        task_obj.bots = [
            BotStatus(
                id=f"{task_obj.id}-{i}",
                name=f"{st.entity or f'Bot {i}'}",
                source=st.entity or str(i),
                status="pending",
                started_at="",
            )
            for i, st in enumerate(decomp.subtasks)
        ]

        async with self._lock:
            self._tasks[task_obj.id] = task_obj

        if self._event_bus:
            await self._event_bus.emit("agent_started", {
                "task_id": task_obj.id,
                "task": task,
                "user_id": user_id,
                "channel": channel,
                "multi_bot": True,
                "decomposed": True,
                "bots": [{"id": b.id, "name": b.name, "source": b.source} for b in task_obj.bots],
            }, source="agent_coordinator")

        async def run_bot(idx: int, subtask: Subtask) -> tuple[str, str, str]:
            now = datetime.now(timezone.utc).isoformat()
            async with self._lock:
                if idx < len(task_obj.bots):
                    task_obj.bots[idx].status = "running"
                    task_obj.bots[idx].started_at = now

            try:
                result = await agent.run(subtask.description)
                output = result.output or ""
                error = result.error or ""
            except Exception as e:
                output = ""
                error = str(e)

            completed_at = datetime.now(timezone.utc).isoformat()
            status = "completed" if not error else "failed"
            async with self._lock:
                if idx < len(task_obj.bots):
                    task_obj.bots[idx].status = status
                    task_obj.bots[idx].output = output
                    task_obj.bots[idx].error = error
                    task_obj.bots[idx].completed_at = completed_at

            if self._event_bus:
                await self._event_bus.emit("agent_bot_progress", {
                    "task_id": task_obj.id,
                    "bot_id": task_obj.bots[idx].id if idx < len(task_obj.bots) else "",
                    "source": subtask.entity,
                    "status": status,
                }, source="agent_coordinator")
            return subtask.entity, output, error

        try:
            tasks_coros = [run_bot(i, st) for i, st in enumerate(decomp.subtasks)]
            results = await asyncio.gather(*tasks_coros, return_exceptions=True)

            parts = [f"## {task}\n\n"]
            for i, r in enumerate(results):
                st = decomp.subtasks[i]
                if isinstance(r, Exception):
                    parts.append(f"### {st.entity}\nError: {r}\n")
                else:
                    entity, out, err = r
                    parts.append(f"### {entity}\n{out or err or '(no output)'}\n")

            task_obj.status = "completed"
            task_obj.final_result = "\n".join(parts)

            if self._event_bus:
                await self._event_bus.emit("agent_completed", {
                    "task_id": task_obj.id,
                    "success": True,
                    "user_id": user_id,
                    "channel": channel,
                }, source="agent_coordinator")

            return AgentResult(
                success=True,
                output=task_obj.final_result,
                steps_completed=len(decomp.subtasks),
                task_id=task_obj.id,
            )
        except Exception as e:
            task_obj.status = "failed"
            logger.error("Parallel subtasks failed", task_id=task_obj.id, error=str(e))
            return AgentResult(success=False, output="", error=str(e), task_id=task_obj.id)
        finally:
            async with self._lock:
                self._tasks[task_obj.id] = task_obj
