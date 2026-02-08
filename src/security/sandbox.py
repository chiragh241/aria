"""Sandbox manager for isolated code execution."""

import asyncio
import os
import platform
import shutil
import subprocess as sp
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of a sandboxed execution."""

    id: str = field(default_factory=lambda: str(uuid4()))
    success: bool = False
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0
    timed_out: bool = False
    error: str | None = None
    sandbox_type: str = ""


class BaseSandbox(ABC):
    """Abstract base class for sandbox implementations."""

    @abstractmethod
    async def execute(
        self,
        command: str,
        timeout: int = 60,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute a command in the sandbox."""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this sandbox type is available."""
        pass


class DirectSandbox(BaseSandbox):
    """
    Direct execution without isolation.

    Only use for trusted operations on trusted paths.
    """

    def __init__(self, trusted_paths: list[str] | None = None) -> None:
        self.settings = get_settings()
        self.trusted_paths = [
            str(Path(p).expanduser().resolve())
            for p in (trusted_paths or self.settings.sandbox.trusted_paths)
        ]

    def _is_path_trusted(self, path: str) -> bool:
        """Check if a path is within trusted paths."""
        resolved = str(Path(path).expanduser().resolve())
        return any(resolved.startswith(trusted) for trusted in self.trusted_paths)

    async def execute(
        self,
        command: str,
        timeout: int = 60,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute command directly (no isolation)."""
        result = ExecutionResult(sandbox_type="direct")
        start_time = datetime.now(timezone.utc)

        # Validate working directory
        if working_dir and not self._is_path_trusted(working_dir):
            result.error = f"Working directory not in trusted paths: {working_dir}"
            return result

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env={**os.environ, **(env or {})},
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
                result.stdout = stdout.decode("utf-8", errors="replace")
                result.stderr = stderr.decode("utf-8", errors="replace")
                result.exit_code = process.returncode or 0
                result.success = result.exit_code == 0
            except asyncio.TimeoutError:
                process.kill()
                result.timed_out = True
                result.error = f"Command timed out after {timeout}s"

        except Exception as e:
            result.error = str(e)

        result.duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return result

    async def is_available(self) -> bool:
        """Direct execution is always available."""
        return True


class DockerSandbox(BaseSandbox):
    """
    Docker-based sandbox for isolated execution.

    Features:
    - Resource limits (memory, CPU)
    - Network isolation
    - Temporary filesystem
    - Volume mounting for specific paths
    """

    def __init__(
        self,
        image: str | None = None,
        memory_limit: str | None = None,
        cpu_limit: float | None = None,
        network_mode: str | None = None,
    ) -> None:
        self.settings = get_settings()
        docker_config = self.settings.sandbox.docker

        self.image = image or docker_config.image
        self.memory_limit = memory_limit or docker_config.memory_limit
        self.cpu_limit = cpu_limit or docker_config.cpu_limit
        self.network_mode = network_mode or docker_config.network_mode

        self._docker_available: bool | None = None

    async def is_available(self) -> bool:
        """Check if Docker is available."""
        if self._docker_available is not None:
            return self._docker_available

        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                "version",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await process.wait()
            self._docker_available = process.returncode == 0
        except Exception:
            self._docker_available = False

        return self._docker_available

    async def execute(
        self,
        command: str,
        timeout: int = 60,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        volumes: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute command in a Docker container."""
        result = ExecutionResult(sandbox_type="docker")
        start_time = datetime.now(timezone.utc)

        if not await self.is_available():
            result.error = "Docker is not available"
            return result

        # Build docker run command
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            f"--memory={self.memory_limit}",
            f"--cpus={self.cpu_limit}",
        ]

        if self.network_mode == "none":
            docker_cmd.append("--network=none")

        # Add environment variables
        for key, value in (env or {}).items():
            docker_cmd.extend(["-e", f"{key}={value}"])

        # Add volume mounts
        for host_path, container_path in (volumes or {}).items():
            docker_cmd.extend(["-v", f"{host_path}:{container_path}"])

        # Set working directory
        if working_dir:
            docker_cmd.extend(["-w", working_dir])

        # Add image and command
        docker_cmd.extend([self.image, "/bin/sh", "-c", command])

        try:
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout + 5,  # Extra time for container overhead
                )
                result.stdout = stdout.decode("utf-8", errors="replace")
                result.stderr = stderr.decode("utf-8", errors="replace")
                result.exit_code = process.returncode or 0
                result.success = result.exit_code == 0
            except asyncio.TimeoutError:
                # Kill the container
                process.kill()
                result.timed_out = True
                result.error = f"Command timed out after {timeout}s"

        except Exception as e:
            result.error = str(e)

        result.duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return result

    async def image_exists(self) -> bool:
        """Check if the sandbox Docker image exists locally."""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "image", "inspect", self.image,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await process.wait()
            return process.returncode == 0
        except Exception:
            return False

    async def build_sandbox_image(self, dockerfile_path: str | None = None) -> bool:
        """Build the sandbox Docker image."""
        if not await self.is_available():
            return False

        # Find project root (where docker/ directory lives)
        project_root = Path(__file__).resolve().parent.parent.parent

        # Use Dockerfile.sandbox from docker/ if not specified
        if dockerfile_path is None:
            candidate = project_root / "docker" / "Dockerfile.sandbox"
            if candidate.exists():
                dockerfile_path = str(candidate)
            else:
                # Fallback: generate a minimal Dockerfile
                dockerfile_content = """\
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git jq && rm -rf /var/lib/apt/lists/*
RUN useradd -m -s /bin/bash sandbox
USER sandbox
WORKDIR /home/sandbox
CMD ["/bin/bash"]
"""
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".dockerfile", delete=False
                ) as f:
                    f.write(dockerfile_content)
                    dockerfile_path = f.name

        logger.info(
            "Building sandbox Docker image",
            image=self.image,
            dockerfile=dockerfile_path,
        )

        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "build",
                "-t", self.image,
                "-f", dockerfile_path,
                str(project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            # No timeout — docker build can take a long time
            # (installs Node.js, ffmpeg, numpy, pandas, etc.)
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(
                    "Failed to build sandbox image",
                    stderr=stderr.decode()[:500],
                )
                return False

            logger.info("Sandbox image built successfully", image=self.image)
            return True

        except Exception as e:
            logger.error("Failed to build sandbox image", error=str(e))
            return False


class SandboxManager:
    """
    Manages different sandbox implementations.

    Selects the appropriate sandbox based on configuration
    and operation requirements.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

        self._direct = DirectSandbox()
        self._docker = DockerSandbox()

        self._default_sandbox = self.settings.sandbox.default

    async def initialize(self) -> None:
        """Initialize and check sandbox availability.

        When sandbox.default is 'docker', verifies Docker is available
        and the sandbox image exists.  Automatically builds the image
        from docker/Dockerfile.sandbox if it is missing.
        """
        docker_available = await self._docker.is_available()

        if self._default_sandbox == "docker":
            if not docker_available:
                logger.warning(
                    "Docker not available, falling back to direct execution. "
                    "Some operations may be less secure."
                )
                self._default_sandbox = "direct"
            else:
                # Check if the sandbox image exists; build it if not
                image_ok = await self._docker.image_exists()
                if not image_ok:
                    logger.info(
                        "Sandbox Docker image not found — building automatically",
                        image=self._docker.image,
                    )
                    built = await self._docker.build_sandbox_image()
                    if not built:
                        logger.warning(
                            "Failed to build sandbox image, "
                            "falling back to direct execution."
                        )
                        self._default_sandbox = "direct"
                    else:
                        logger.info("Sandbox Docker image ready")
                else:
                    logger.info(
                        "Sandbox Docker image found",
                        image=self._docker.image,
                    )

        logger.info(
            "Sandbox manager initialized",
            default=self._default_sandbox,
            docker_available=docker_available,
        )

    async def execute(
        self,
        command: str,
        sandbox_type: str | None = None,
        timeout: int | None = None,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        volumes: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """
        Execute a command in the appropriate sandbox.

        Args:
            command: The command to execute
            sandbox_type: Override sandbox type (docker, direct)
            timeout: Execution timeout in seconds
            working_dir: Working directory
            env: Environment variables
            volumes: Volume mounts (Docker only)

        Returns:
            ExecutionResult with output and status
        """
        sandbox_type = sandbox_type or self._default_sandbox
        timeout = timeout or self.settings.sandbox.docker.timeout

        logger.debug(
            "Executing command",
            command=command[:100] + "..." if len(command) > 100 else command,
            sandbox_type=sandbox_type,
        )

        if sandbox_type == "docker":
            if not await self._docker.is_available():
                logger.warning("Docker not available, using direct execution")
                sandbox_type = "direct"
            else:
                return await self._docker.execute(
                    command=command,
                    timeout=timeout,
                    working_dir=working_dir,
                    env=env,
                    volumes=volumes,
                )

        return await self._direct.execute(
            command=command,
            timeout=timeout,
            working_dir=working_dir,
            env=env,
        )

    async def execute_script(
        self,
        script: str,
        language: str = "python",
        sandbox_type: str | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """
        Execute a script in the sandbox.

        Args:
            script: The script content
            language: Script language (python, bash, etc.)
            sandbox_type: Override sandbox type
            timeout: Execution timeout

        Returns:
            ExecutionResult with output and status
        """
        # Create temp file for script
        suffix = {
            "python": ".py",
            "bash": ".sh",
            "javascript": ".js",
            "ruby": ".rb",
        }.get(language, ".txt")

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=suffix,
            delete=False,
        ) as f:
            f.write(script)
            script_path = f.name

        try:
            # Build command based on language
            if language == "python":
                command = f"python3 {script_path}"
            elif language == "bash":
                command = f"bash {script_path}"
            elif language == "javascript":
                command = f"node {script_path}"
            elif language == "ruby":
                command = f"ruby {script_path}"
            else:
                command = script_path

            # For Docker, mount the script file
            volumes = None
            if (sandbox_type or self._default_sandbox) == "docker":
                volumes = {script_path: script_path}

            return await self.execute(
                command=command,
                sandbox_type=sandbox_type,
                timeout=timeout,
                volumes=volumes,
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(script_path)
            except Exception:
                pass

    def get_status(self) -> dict[str, Any]:
        """Get sandbox manager status."""
        return {
            "default_sandbox": self._default_sandbox,
            "docker_available": self._docker._docker_available,
            "platform": platform.system(),
        }

    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        logger.info("Sandbox manager cleaned up")
