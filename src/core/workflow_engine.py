"""Skill chaining — run multiple skills in sequence, passing output between steps."""

from datetime import datetime, timezone
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


class WorkflowEngine:
    """
    Chains skills together: step N output becomes input for step N+1.

    Each step: { "skill": "research", "capability": "search_topic", "args": {...} }
    Use {{step_0.output}} in args to reference previous step output.
    """

    def __init__(self, skill_registry: Any = None) -> None:
        self._skill_registry = skill_registry or None

    def set_skill_registry(self, sr: Any) -> None:
        self._skill_registry = sr

    async def run_chain(
        self,
        steps: list[dict[str, Any]],
        channel: str = "web",
        user_id: str = "default",
    ) -> dict[str, Any]:
        """
        Execute a chain of skills. Each step gets previous outputs as {{step_N.output}}.

        Args:
            steps: List of {"skill": str, "capability": str, "args": dict}
            channel: Channel for skill execution
            user_id: User ID

        Returns:
            {"success": bool, "outputs": [...], "final_output": Any, "error": str|None}
        """
        if not self._skill_registry:
            return {"success": False, "error": "Skill registry not available", "outputs": []}

        outputs: list[Any] = []
        step_outputs: dict[str, Any] = {}

        for i, step in enumerate(steps):
            skill_name = step.get("skill") or step.get("skill_name")
            capability = step.get("capability")
            args = dict(step.get("args") or step.get("arguments") or {})

            if not skill_name or not capability:
                return {
                    "success": False,
                    "error": f"Step {i}: skill and capability required",
                    "outputs": outputs,
                }

            # Inject channel/user_id
            args.setdefault("channel", channel)
            args.setdefault("user_id", user_id)

            # Replace {{step_N.output}} placeholders
            for k, v in list(args.items()):
                if isinstance(v, str) and "{{" in v:
                    for j in range(i):
                        placeholder = f"{{{{step_{j}.output}}}}"
                        if placeholder in v and j < len(outputs):
                            out_val = outputs[j]
                            out_str = str(out_val.output if hasattr(out_val, "output") else out_val)
                            args[k] = v.replace(placeholder, out_str)
                            v = args[k]

            skill = self._skill_registry.get_skill(skill_name)
            if not skill or not skill.enabled:
                return {
                    "success": False,
                    "error": f"Step {i}: skill '{skill_name}' not found or disabled",
                    "outputs": outputs,
                }

            try:
                result = await skill.execute(capability, **args)
                step_outputs[f"step_{i}"] = result
                out_val = result.output if result.success else result.error
                outputs.append(result)

                if not result.success:
                    return {
                        "success": False,
                        "error": result.error or "Step failed",
                        "outputs": outputs,
                        "final_output": out_val,
                    }
            except Exception as e:
                logger.error("Workflow step failed", step=i, skill=skill_name, error=str(e))
                return {
                    "success": False,
                    "error": str(e),
                    "outputs": outputs,
                }

        final = outputs[-1] if outputs else None
        final_out = final.output if hasattr(final, "output") else final
        return {
            "success": True,
            "outputs": [o.output if hasattr(o, "output") else o for o in outputs],
            "final_output": final_out,
            "step_count": len(outputs),
        }
