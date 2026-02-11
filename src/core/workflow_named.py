"""
Named workflows: YAML-defined multi-step pipelines with specialist agents.

Each workflow has a fixed sequence of steps. Each step runs a specialist agent
(research, coding, data); step output is passed to the next via {{step_N.output}}
and {{task}}. Failed steps can retry before failing the run.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WorkflowStep:
    id: str
    agent_type: str
    input_template: str


@dataclass
class NamedWorkflow:
    id: str
    name: str
    description: str
    steps: list[WorkflowStep] = field(default_factory=list)


def _render_template(template: str, variables: dict[str, str]) -> str:
    out = template
    for key, value in variables.items():
        placeholder = "{{" + key + "}}"
        if placeholder in out:
            out = out.replace(placeholder, value or "")
    out = re.sub(r"\{\{[^}]+\}\}", "", out)
    return out


def load_workflows(directory: str | Path) -> dict[str, NamedWorkflow]:
    directory = Path(directory).expanduser().resolve()
    fallback = Path("config/workflows").resolve()
    if not directory.exists() and fallback.exists():
        directory = fallback
    result: dict[str, NamedWorkflow] = {}
    if not directory.exists():
        return result
    for path in directory.glob("*.yaml"):
        if path.name.startswith("."):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            wf_id = data.get("id") or path.stem
            steps = []
            for s in data.get("steps") or []:
                step_id = s.get("id") or str(len(steps))
                agent_type = (s.get("agent_type") or s.get("agent") or "research").lower()
                if agent_type not in ("research", "coding", "data"):
                    agent_type = "research"
                inp = s.get("input") or s.get("input_template") or ""
                if isinstance(inp, list):
                    inp = "\n".join(inp)
                steps.append(WorkflowStep(id=step_id, agent_type=agent_type, input_template=inp))
            result[wf_id] = NamedWorkflow(
                id=wf_id,
                name=data.get("name") or wf_id,
                description=data.get("description") or "",
                steps=steps,
            )
        except Exception as e:
            logger.warning("Failed to load workflow", path=str(path), error=str(e))
    return result


class NamedWorkflowRunner:
    """Runs named workflows: deterministic steps, specialist agents, optional retry per step."""

    def __init__(self, agent_coordinator: Any, workflows_dir: str | Path | None = None) -> None:
        self._coordinator = agent_coordinator
        settings = get_settings()
        cfg = getattr(settings.skills.builtin, "workflow", None) if hasattr(settings, "skills") else None
        if isinstance(cfg, dict):
            self._workflows_dir = Path((workflows_dir or cfg.get("workflows_dir", "./data/workflows")).strip()).expanduser()
            self._retry_per_step = int(cfg.get("retry_per_step", 1))
        else:
            self._workflows_dir = Path((workflows_dir or "./data/workflows").strip()).expanduser()
            self._retry_per_step = 1
        self._cache: dict[str, NamedWorkflow] = {}

    def list_workflows(self) -> list[dict[str, Any]]:
        self._cache = load_workflows(self._workflows_dir)
        return [{"id": wf.id, "name": wf.name, "description": wf.description, "step_count": len(wf.steps)} for wf in self._cache.values()]

    def get_workflow(self, workflow_id: str) -> NamedWorkflow | None:
        if not self._cache:
            self._cache = load_workflows(self._workflows_dir)
        return self._cache.get(workflow_id)

    async def run(self, workflow_id: str, task: str, user_id: str = "default", channel: str = "web") -> dict[str, Any]:
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return {"success": False, "error": f"Workflow '{workflow_id}' not found.", "step_outputs": []}
        if not self._coordinator:
            return {"success": False, "error": "Agent coordinator not available", "step_outputs": []}
        step_outputs: list[str] = []
        variables: dict[str, str] = {"task": task}
        for i, step in enumerate(workflow.steps):
            variables[f"step_{i}.output"] = ""
            inp = _render_template(step.input_template, variables)
            if not inp.strip():
                inp = task
            last_error = ""
            for attempt in range(self._retry_per_step + 1):
                try:
                    result = await self._coordinator.delegate(task=inp, agent_type=step.agent_type, user_id=user_id, channel=channel)
                    out_text = (result.output or "") if result.success else (result.error or "No output")
                    step_outputs.append(out_text)
                    variables[f"step_{i}.output"] = out_text
                    if not result.success:
                        last_error = result.error or result.output or "Step failed"
                        if attempt < self._retry_per_step:
                            logger.info("Workflow step failed, retrying", workflow=workflow_id, step=step.id, attempt=attempt + 1)
                            continue
                        return {"success": False, "error": f"Step '{step.id}': {last_error}", "step_outputs": step_outputs, "final_output": out_text}
                    break
                except Exception as e:
                    last_error = str(e)
                    if attempt < self._retry_per_step:
                        continue
                    logger.exception("Workflow step failed")
                    return {"success": False, "error": f"Step '{step.id}' failed: {last_error}", "step_outputs": step_outputs}
        final_parts = [f"**Workflow: {workflow.name}**\n"]
        for i, step in enumerate(workflow.steps):
            if i < len(step_outputs):
                final_parts.append(f"### {step.id} ({step.agent_type})\n{step_outputs[i][:2000]}")
        return {"success": True, "final_output": "\n\n".join(final_parts), "step_outputs": step_outputs}
