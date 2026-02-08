"""Background process manager for Aria.

Spawns, monitors, and manages long-running processes so the AI can
kick off builds, downloads, scripts, etc. without blocking the conversation.

Usage:
    pm = ProcessManager(event_bus)
    pid = await pm.spawn("npm run build", cwd="/project")
    info = pm.poll(pid)           # check status
    log = pm.get_log(pid)         # get stdout/stderr
    await pm.kill(pid)            # terminate
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .events import EventBus
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ManagedProcess:
    """A tracked background process."""

    id: str
    command: str
    cwd: str | None
    process: asyncio.subprocess.Process | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    returncode: int | None = None
    stdout_lines: list[str] = field(default_factory=list)
    stderr_lines: list[str] = field(default_factory=list)
    max_log_lines: int = 500
    _monitor_task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def running(self) -> bool:
        return self.process is not None and self.returncode is None

    @property
    def duration(self) -> float:
        end = self.finished_at or time.time()
        return end - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "command": self.command,
            "cwd": self.cwd,
            "running": self.running,
            "returncode": self.returncode,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": round(self.duration, 1),
            "stdout_lines": len(self.stdout_lines),
            "stderr_lines": len(self.stderr_lines),
        }


class ProcessManager:
    """Manages background processes with monitoring and log capture."""

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._processes: dict[str, ManagedProcess] = {}
        self._event_bus = event_bus

    async def spawn(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        shell: bool = True,
    ) -> str:
        """Spawn a background process and return its ID."""
        pid = str(uuid4())[:8]

        import os
        merged_env = {**os.environ, **(env or {})}

        if shell:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=merged_env,
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                *command.split(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=merged_env,
            )

        mp = ManagedProcess(id=pid, command=command, cwd=cwd, process=proc)
        self._processes[pid] = mp

        # Start monitoring in background
        mp._monitor_task = asyncio.create_task(self._monitor(mp))

        logger.info("Spawned background process", pid=pid, command=command[:80])

        if self._event_bus:
            await self._event_bus.emit("process_started", {
                "pid": pid,
                "command": command,
            }, source="process_manager")

        return pid

    async def _monitor(self, mp: ManagedProcess) -> None:
        """Monitor a process, capturing output until it exits."""
        proc = mp.process
        if not proc:
            return

        async def _read_stream(stream: asyncio.StreamReader | None, target: list[str]) -> None:
            if not stream:
                return
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip("\n")
                target.append(decoded)
                if len(target) > mp.max_log_lines:
                    target.pop(0)

        await asyncio.gather(
            _read_stream(proc.stdout, mp.stdout_lines),
            _read_stream(proc.stderr, mp.stderr_lines),
        )

        await proc.wait()
        mp.returncode = proc.returncode
        mp.finished_at = time.time()

        logger.info(
            "Background process completed",
            pid=mp.id,
            returncode=mp.returncode,
            duration=f"{mp.duration:.1f}s",
        )

        if self._event_bus:
            await self._event_bus.emit("process_completed", {
                "pid": mp.id,
                "command": mp.command,
                "returncode": mp.returncode,
                "duration": mp.duration,
                "stdout_tail": mp.stdout_lines[-5:] if mp.stdout_lines else [],
            }, source="process_manager")

    def poll(self, pid: str) -> dict[str, Any] | None:
        """Check the status of a process."""
        mp = self._processes.get(pid)
        if not mp:
            return None
        return mp.to_dict()

    def get_log(self, pid: str, stream: str = "stdout", offset: int = 0, limit: int = 100) -> list[str] | None:
        """Get log lines from a process."""
        mp = self._processes.get(pid)
        if not mp:
            return None
        lines = mp.stdout_lines if stream == "stdout" else mp.stderr_lines
        return lines[offset:offset + limit]

    async def kill(self, pid: str) -> bool:
        """Kill a running process."""
        mp = self._processes.get(pid)
        if not mp or not mp.process or not mp.running:
            return False
        try:
            mp.process.terminate()
            await asyncio.wait_for(mp.process.wait(), timeout=5)
        except asyncio.TimeoutError:
            mp.process.kill()
        mp.returncode = mp.process.returncode
        mp.finished_at = time.time()
        logger.info("Killed background process", pid=pid)
        return True

    async def write_stdin(self, pid: str, data: str) -> bool:
        """Write data to a process's stdin."""
        mp = self._processes.get(pid)
        if not mp or not mp.process or not mp.process.stdin or not mp.running:
            return False
        mp.process.stdin.write(data.encode())
        await mp.process.stdin.drain()
        return True

    def list_processes(self, running_only: bool = False) -> list[dict[str, Any]]:
        """List all tracked processes."""
        procs = self._processes.values()
        if running_only:
            procs = [p for p in procs if p.running]
        return [p.to_dict() for p in procs]

    async def cleanup(self) -> None:
        """Kill all running processes."""
        for mp in list(self._processes.values()):
            if mp.running:
                await self.kill(mp.id)
