"""Shell command execution skill."""

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

from ...security.sandbox import ExecutionResult, SandboxManager
from ..base import BaseSkill, SkillResult


class ShellSkill(BaseSkill):
    """
    Skill for executing shell commands.

    Supports both sandboxed (Docker) and direct execution
    based on security configuration.
    """

    name = "shell"
    description = "Execute shell commands"
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.timeout = config.get("timeout", 60)
        self._sandbox: SandboxManager | None = None

    def _register_capabilities(self) -> None:
        """Register shell capabilities."""
        self.register_capability(
            name="execute",
            description="Execute a shell command",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"},
                    "working_dir": {"type": "string", "description": "Working directory"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds"},
                    "sandbox": {"type": "string", "enum": ["docker", "direct"], "description": "Sandbox type"},
                },
                "required": ["command"],
            },
            security_action="shell_commands",
        )

        self.register_capability(
            name="run_script",
            description="Execute a script file or inline script",
            parameters={
                "type": "object",
                "properties": {
                    "script": {"type": "string", "description": "Script content or file path"},
                    "language": {"type": "string", "enum": ["bash", "python", "javascript"], "default": "bash"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds"},
                },
                "required": ["script"],
            },
            security_action="shell_commands",
        )

    async def initialize(self) -> None:
        """Initialize the sandbox manager."""
        self._sandbox = SandboxManager()
        await self._sandbox.initialize()
        self._initialized = True

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        """Execute a shell capability."""
        start_time = datetime.now(timezone.utc)

        if capability == "execute":
            return await self._execute_command(start_time, **kwargs)
        elif capability == "run_script":
            return await self._run_script(start_time, **kwargs)
        else:
            return self._error_result(f"Unknown capability: {capability}", start_time)

    async def _execute_command(
        self,
        start_time: datetime,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
        sandbox: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SkillResult:
        """Execute a shell command."""
        if not self._sandbox:
            return self._error_result("Sandbox not initialized", start_time)

        try:
            result = await self._sandbox.execute(
                command=command,
                sandbox_type=sandbox,
                timeout=timeout or self.timeout,
                working_dir=working_dir,
                env=env,
            )

            return SkillResult(
                success=result.success,
                output={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                },
                error=result.error,
                execution_time_ms=result.duration_ms,
                metadata={
                    "sandbox_type": result.sandbox_type,
                    "command": command[:100] + "..." if len(command) > 100 else command,
                },
            )

        except Exception as e:
            return self._error_result(str(e), start_time)

    async def _run_script(
        self,
        start_time: datetime,
        script: str,
        language: str = "bash",
        timeout: int | None = None,
    ) -> SkillResult:
        """Run a script."""
        if not self._sandbox:
            return self._error_result("Sandbox not initialized", start_time)

        try:
            result = await self._sandbox.execute_script(
                script=script,
                language=language,
                timeout=timeout or self.timeout,
            )

            return SkillResult(
                success=result.success,
                output={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                },
                error=result.error,
                execution_time_ms=result.duration_ms,
                metadata={
                    "sandbox_type": result.sandbox_type,
                    "language": language,
                },
            )

        except Exception as e:
            return self._error_result(str(e), start_time)

    async def shutdown(self) -> None:
        """Shutdown the skill."""
        self._initialized = False
