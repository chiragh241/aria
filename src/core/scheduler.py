"""Cron / Scheduler for Aria — schedule reminders, recurring tasks, and timed events.

Supports:
- One-shot reminders ("remind me at 5pm")
- Recurring jobs with cron expressions ("every weekday at 9am")
- Interval-based repeats ("every 30 minutes")

Jobs persist to disk as JSON so they survive restarts.
The heartbeat service ticks the scheduler each cycle.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .events import EventBus
from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScheduledJob:
    """A scheduled task."""

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    # "once" = fire once then delete, "interval" = repeat every N seconds,
    # "cron" = cron expression (basic support)
    schedule_type: str = "once"
    # For "once": ISO timestamp. For "interval": seconds. For "cron": expression.
    schedule_value: str = ""
    # What to do when it fires
    action: str = "reminder"  # "reminder", "system_event", "agent_turn"
    payload: dict[str, Any] = field(default_factory=dict)
    # Delivery target
    channel: str = ""  # "" = last active channel
    user_id: str = ""
    # State
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    next_run_at: float = 0.0
    last_run_at: float | None = None
    last_status: str = ""
    run_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "schedule_type": self.schedule_type,
            "schedule_value": self.schedule_value,
            "action": self.action,
            "payload": self.payload,
            "channel": self.channel,
            "user_id": self.user_id,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "next_run_at": self.next_run_at,
            "last_run_at": self.last_run_at,
            "last_status": self.last_status,
            "run_count": self.run_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduledJob":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class Scheduler:
    """Manages scheduled jobs with persistent storage."""

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._jobs: dict[str, ScheduledJob] = {}
        settings = get_settings()
        self._persist_path = Path(settings.aria.data_dir).expanduser() / "scheduler.json"
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

        # Listen for heartbeat ticks
        self._event_bus.on("heartbeat", self._on_heartbeat)

    # --- Public API ---

    def add_job(
        self,
        name: str,
        schedule_type: str,
        schedule_value: str,
        action: str = "reminder",
        payload: dict[str, Any] | None = None,
        channel: str = "",
        user_id: str = "",
    ) -> ScheduledJob:
        """Add a new scheduled job."""
        job = ScheduledJob(
            name=name,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            action=action,
            payload=payload or {},
            channel=channel,
            user_id=user_id,
        )
        job.next_run_at = self._compute_next_run(job)
        self._jobs[job.id] = job
        self._save()
        logger.info("Scheduled job added", job_id=job.id, name=name, next_run=job.next_run_at)
        return job

    def remove_job(self, job_id: str) -> bool:
        """Remove a job."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            self._save()
            return True
        return False

    def enable_job(self, job_id: str, enabled: bool = True) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        job.enabled = enabled
        if enabled:
            job.next_run_at = self._compute_next_run(job)
        self._save()
        return True

    def get_job(self, job_id: str) -> ScheduledJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        jobs = self._jobs.values()
        if enabled_only:
            jobs = [j for j in jobs if j.enabled]
        return [j.to_dict() for j in sorted(jobs, key=lambda j: j.next_run_at)]

    def get_due_jobs(self) -> list[ScheduledJob]:
        """Get jobs that are due to fire right now."""
        now = time.time()
        return [
            j for j in self._jobs.values()
            if j.enabled and j.next_run_at > 0 and j.next_run_at <= now
        ]

    # --- Heartbeat handler ---

    async def _on_heartbeat(self, event: Any) -> None:
        """Check for due jobs on each heartbeat tick."""
        due_jobs = self.get_due_jobs()
        for job in due_jobs:
            try:
                await self._fire_job(job)
            except Exception as e:
                logger.error("Failed to fire scheduled job", job_id=job.id, error=str(e))
                job.last_status = f"error: {e}"

    async def _fire_job(self, job: ScheduledJob) -> None:
        """Execute a due job."""
        logger.info("Firing scheduled job", job_id=job.id, name=job.name, action=job.action)

        job.last_run_at = time.time()
        job.run_count += 1

        # Emit event so orchestrator / channels can act on it
        await self._event_bus.emit("cron_fired", {
            "job_id": job.id,
            "name": job.name,
            "action": job.action,
            "payload": job.payload,
            "channel": job.channel,
            "user_id": job.user_id,
        }, source="scheduler")

        # If it's a reminder, also emit a specific event
        if job.action == "reminder":
            await self._event_bus.emit("reminder_due", {
                "job_id": job.id,
                "message": job.payload.get("message", job.name),
                "channel": job.channel,
                "user_id": job.user_id,
            }, source="scheduler")

        job.last_status = "ok"

        # Compute next run or delete one-shot jobs
        if job.schedule_type == "once":
            job.enabled = False
            job.next_run_at = 0
        else:
            job.next_run_at = self._compute_next_run(job)

        self._save()

    # --- Schedule computation ---

    def _compute_next_run(self, job: ScheduledJob) -> float:
        """Compute the next run timestamp for a job."""
        now = time.time()

        if job.schedule_type == "once":
            # schedule_value is an ISO timestamp or unix timestamp
            try:
                ts = float(job.schedule_value)
                return ts
            except ValueError:
                pass
            try:
                dt = datetime.fromisoformat(job.schedule_value)
                return dt.timestamp()
            except ValueError:
                logger.warning("Invalid once schedule", value=job.schedule_value)
                return 0.0

        elif job.schedule_type == "interval":
            # schedule_value is seconds
            try:
                interval = float(job.schedule_value)
                base = job.last_run_at or now
                return base + interval
            except ValueError:
                return 0.0

        elif job.schedule_type == "cron":
            return self._next_cron_time(job.schedule_value, now)

        return 0.0

    def _next_cron_time(self, expression: str, after: float) -> float:
        """Basic cron expression support (minute hour day month weekday).

        Supports: *, specific numbers, and simple intervals (*/N).
        For full cron support, install croniter.
        """
        try:
            from croniter import croniter
            base = datetime.fromtimestamp(after, tz=timezone.utc)
            cron = croniter(expression, base)
            next_dt = cron.get_next(datetime)
            return next_dt.timestamp()
        except ImportError:
            pass

        # Fallback: treat as interval in minutes if it's just a number
        try:
            minutes = int(expression)
            return after + (minutes * 60)
        except ValueError:
            logger.warning("croniter not installed and can't parse cron expression", expr=expression)
            return 0.0

    # --- Persistence ---

    def _save(self) -> None:
        try:
            data = {jid: j.to_dict() for jid, j in self._jobs.items()}
            self._persist_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save scheduler state", error=str(e))

    def _load(self) -> None:
        if not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for jid, jdata in data.items():
                self._jobs[jid] = ScheduledJob.from_dict(jdata)
            logger.info("Loaded scheduled jobs", count=len(self._jobs))
        except Exception as e:
            logger.warning("Failed to load scheduler state", error=str(e))
