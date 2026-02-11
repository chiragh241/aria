"""Dynamic skill generator for creating new skills on demand."""

import ast
import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..utils.config import get_settings
from ..utils.logging import get_logger
from .base import BaseSkill, SkillResult

logger = get_logger(__name__)


@dataclass
class GeneratedSkill:
    """Metadata for a generated skill."""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    code: str = ""
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tested: bool = False
    approved: bool = False
    error: str | None = None


class SkillGenerator:
    """
    Generator for creating new skills dynamically.

    Uses LLM to generate skill code based on descriptions,
    validates the generated code, and saves to the learned
    skills directory.
    """

    def __init__(self, llm_router: Any = None) -> None:
        self.settings = get_settings()
        self._llm_router = llm_router
        self._learned_dir = Path(self.settings.skills.learned.directory).expanduser()
        self._learned_dir.mkdir(parents=True, exist_ok=True)
        self._pending_dir = Path(self.settings.aria.data_dir).expanduser() / "pending_skills"
        self._pending_dir.mkdir(parents=True, exist_ok=True)

    def set_llm_router(self, router: Any) -> None:
        """Set the LLM router for code generation."""
        self._llm_router = router

    async def generate(
        self,
        name: str,
        description: str,
        capabilities: list[dict[str, str]],
        examples: list[dict[str, Any]] | None = None,
    ) -> GeneratedSkill:
        """
        Generate a new skill.

        Args:
            name: Skill name
            description: What the skill does
            capabilities: List of capabilities with name and description
            examples: Optional usage examples

        Returns:
            GeneratedSkill with generated code
        """
        if not self._llm_router:
            return GeneratedSkill(
                name=name,
                description=description,
                error="LLM router not configured",
            )

        # Build the prompt
        prompt = self._build_generation_prompt(name, description, capabilities, examples)

        try:
            from ..core.llm_router import LLMMessage

            messages = [
                LLMMessage(role="system", content=self._get_system_prompt()),
                LLMMessage(role="user", content=prompt),
            ]

            response = await self._llm_router.generate(
                messages=messages,
                task_type="skill_creation",
            )

            # Extract code from response
            code = self._extract_code(response.content)

            if not code:
                return GeneratedSkill(
                    name=name,
                    description=description,
                    error="Failed to generate valid code",
                )

            # Validate the code
            validation_error = self._validate_code(code)
            if validation_error:
                return GeneratedSkill(
                    name=name,
                    description=description,
                    code=code,
                    error=validation_error,
                )

            return GeneratedSkill(
                name=name,
                description=description,
                code=code,
            )

        except Exception as e:
            logger.error("Skill generation failed", error=str(e))
            return GeneratedSkill(
                name=name,
                description=description,
                error=str(e),
            )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for skill generation."""
        return """You are a Python skill generator for an AI assistant called Aria.

Generate skills that follow this pattern:

```python
from datetime import datetime, timezone
from typing import Any
from ..base import BaseSkill, SkillResult

class {SkillName}Skill(BaseSkill):
    name = "{skill_name}"
    description = "{description}"
    version = "1.0.0"

    def _register_capabilities(self) -> None:
        self.register_capability(
            name="capability_name",
            description="What this capability does",
            parameters={
                "type": "object",
                "properties": {
                    "param_name": {"type": "string", "description": "Parameter description"},
                },
                "required": ["param_name"],
            },
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        start_time = datetime.now(timezone.utc)

        if capability == "capability_name":
            try:
                result = await self._capability_name(**kwargs)
                return self._success_result(result, start_time)
            except Exception as e:
                return self._error_result(str(e), start_time)

        return self._error_result(f"Unknown capability: {capability}", start_time)

    async def _capability_name(self, param_name: str) -> dict[str, Any]:
        # Implementation
        return {"result": "value"}
```

Rules:
1. Always inherit from BaseSkill
2. Use async methods
3. Return SkillResult using _success_result or _error_result
4. Include proper type hints
5. Handle errors gracefully
6. You may use: standard library, requests, aiohttp, subprocess, asyncio. For CLI tools use subprocess; for HTTP APIs use requests or aiohttp.
7. To gain access to API keys or config: accept parameters (e.g. api_key, token) or read from os.environ.get("SERVICE_API_KEY") so the user can set env vars. Document required env vars in the capability description if needed.
8. Keep code simple and focused. If the task needs an external CLI, use subprocess to run it.
"""

    def _build_generation_prompt(
        self,
        name: str,
        description: str,
        capabilities: list[dict[str, str]],
        examples: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build the generation prompt."""
        caps_text = "\n".join(
            f"- {c['name']}: {c.get('description', '')}"
            for c in capabilities
        )

        prompt = f"""Generate a Python skill with the following specifications:

Name: {name}
Description: {description}

Capabilities:
{caps_text}

If this skill needs external access (APIs, CLI tools): accept api_key/token as parameters or read from os.environ. Use requests/aiohttp for HTTP, subprocess for CLI. Make the skill self-contained and robust to missing config (return clear error asking for the key or env var).

Prefer free options: use free/public APIs when possible; for bookings or actions on websites, consider browser automation or scraping public pages rather than paid APIs.
"""

        if examples:
            examples_text = "\n".join(
                f"- {e.get('input', '')} -> {e.get('output', '')}"
                for e in examples
            )
            prompt += f"\nExamples:\n{examples_text}"

        prompt += "\n\nGenerate the complete skill class code:"

        return prompt

    def _extract_code(self, response: str) -> str | None:
        """Extract Python code from LLM response."""
        # Look for code blocks
        code_pattern = r"```python\n(.*?)```"
        matches = re.findall(code_pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Try to find class definition directly
        class_pattern = r"(class \w+Skill\(BaseSkill\):.*)"
        matches = re.findall(class_pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        return None

    def _validate_code(self, code: str) -> str | None:
        """
        Validate generated Python code.

        Returns error message if invalid, None if valid.
        """
        try:
            # Parse the code
            tree = ast.parse(code)

            # Check for class definition
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            if not classes:
                return "No class definition found"

            skill_class = classes[0]

            # Check inheritance
            if not any(
                isinstance(base, ast.Name) and base.id == "BaseSkill"
                for base in skill_class.bases
            ):
                return "Class must inherit from BaseSkill"

            # Check for required methods
            methods = {
                node.name
                for node in skill_class.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }

            if "_register_capabilities" not in methods:
                return "Missing _register_capabilities method"

            if "execute" not in methods:
                return "Missing execute method"

            return None

        except SyntaxError as e:
            return f"Syntax error: {str(e)}"
        except Exception as e:
            return f"Validation error: {str(e)}"

    async def test(self, skill: GeneratedSkill) -> tuple[bool, str]:
        """
        Test a generated skill.

        Args:
            skill: The generated skill to test

        Returns:
            Tuple of (success, message)
        """
        if not skill.code:
            return False, "No code to test"

        try:
            # Create a temporary module
            import importlib.util
            import sys
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
            ) as f:
                # Add imports
                f.write("from datetime import datetime, timezone\n")
                f.write("from typing import Any\n")
                f.write("from aria.skills.base import BaseSkill, SkillResult\n\n")
                f.write(skill.code)
                temp_path = f.name

            try:
                spec = importlib.util.spec_from_file_location(skill.name, temp_path)
                if not spec or not spec.loader:
                    return False, "Failed to create module spec"

                module = importlib.util.module_from_spec(spec)
                sys.modules[skill.name] = module
                spec.loader.exec_module(module)

                # Find skill class
                skill_class = None
                for name in dir(module):
                    obj = getattr(module, name)
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, BaseSkill)
                        and obj != BaseSkill
                    ):
                        skill_class = obj
                        break

                if not skill_class:
                    return False, "No skill class found"

                # Instantiate and test
                instance = skill_class()
                if not instance.get_capabilities():
                    return False, "No capabilities registered"

                skill.tested = True
                return True, "Skill tested successfully"

            finally:
                Path(temp_path).unlink(missing_ok=True)
                sys.modules.pop(skill.name, None)

        except Exception as e:
            return False, f"Test failed: {str(e)}"

    async def save(self, skill: GeneratedSkill) -> str | None:
        """
        Save a generated skill to disk.

        Args:
            skill: The skill to save

        Returns:
            File path if successful, None otherwise
        """
        if not skill.code:
            return None

        if not skill.tested and self.settings.skills.learned.auto_test:
            success, message = await self.test(skill)
            if not success:
                logger.warning("Skill test failed", skill=skill.name, message=message)
                if self.settings.skills.learned.require_approval:
                    return None

        if self.settings.skills.learned.require_approval:
            from uuid import uuid4
            pending_id = str(uuid4())[:8]
            pending_path = self._pending_dir / f"{pending_id}.json"
            payload = {
                "id": pending_id,
                "name": skill.name,
                "description": skill.description,
                "code": skill.code,
                "created_at": skill.created_at.isoformat(),
            }
            pending_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.info("Skill saved for approval", skill=skill.name, pending_id=pending_id)
            return f"pending:{pending_id}"

        # Create file
        file_path = self._learned_dir / f"{skill.name}.py"

        # Add header and imports (absolute import for standalone learned skills)
        full_code = f'''"""
Generated skill: {skill.name}
Description: {skill.description}
Generated at: {skill.created_at.isoformat()}
"""

from datetime import datetime, timezone
from typing import Any

from src.skills.base import BaseSkill, SkillResult


{skill.code}
'''

        file_path.write_text(full_code)
        logger.info("Saved generated skill", skill=skill.name, path=str(file_path))

        return str(file_path)

    def list_learned_skills(self) -> list[dict[str, Any]]:
        """List all learned skills."""
        skills = []
        for file_path in self._learned_dir.glob("*.py"):
            if file_path.name.startswith("_"):
                continue

            # Read first few lines to get metadata
            try:
                content = file_path.read_text()
                # Extract docstring
                match = re.search(r'"""(.*?)"""', content, re.DOTALL)
                description = match.group(1).strip() if match else ""

                skills.append({
                    "name": file_path.stem,
                    "path": str(file_path),
                    "description": description,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                })
            except Exception:
                continue

        return skills

    def delete_learned_skill(self, name: str) -> bool:
        """Delete a learned skill."""
        file_path = self._learned_dir / f"{name}.py"
        if file_path.exists():
            file_path.unlink()
            logger.info("Deleted learned skill", skill=name)
            return True
        return False

    def list_pending_skills(self) -> list[dict[str, Any]]:
        """List skills waiting for approval."""
        result = []
        for path in self._pending_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                result.append({
                    "id": data.get("id", path.stem),
                    "name": data.get("name", ""),
                    "description": data.get("description", ""),
                    "created_at": data.get("created_at", ""),
                })
            except Exception:
                continue
        return result

    def approve_pending_skill(self, pending_id: str) -> str | None:
        """Move a pending skill to learned dir and return the path. Returns None on failure."""
        path = self._pending_dir / f"{pending_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            name = data.get("name", "").strip()
            code = data.get("code", "")
            if not name or not code:
                return None
            full_code = f'''"""
Generated skill: {name}
Description: {data.get('description', '')}
Approved from pending: {pending_id}
"""

from datetime import datetime, timezone
from typing import Any

from src.skills.base import BaseSkill, SkillResult


{code}
'''
            file_path = self._learned_dir / f"{name}.py"
            file_path.write_text(full_code)
            path.unlink(missing_ok=True)
            logger.info("Approved pending skill", skill=name, pending_id=pending_id)
            return str(file_path)
        except Exception as e:
            logger.warning("Failed to approve pending skill", pending_id=pending_id, error=str(e))
            return None
