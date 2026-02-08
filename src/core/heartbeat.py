"""Heartbeat service — the clock that keeps Aria alive between messages.

Fires a periodic heartbeat event so the scheduler can check for due jobs,
background processes can report status, and the AI can do proactive work.

The heartbeat message is sent to the orchestrator as a hidden system poll.
If the AI has nothing to say, it replies HEARTBEAT_OK (silently discarded).
If something needs attention, it sends a real message to the user.
"""

import asyncio
from typing import Any, Callable, Coroutine

from .events import EventBus
from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

HEARTBEAT_OK = "HEARTBEAT_OK"


class HeartbeatService:
    """Periodic heartbeat that drives proactive behavior."""

    def __init__(
        self,
        event_bus: EventBus,
        interval_seconds: int = 60,
        on_heartbeat: Callable[[], Coroutine[Any, Any, str | None]] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._interval = interval_seconds
        self._on_heartbeat = on_heartbeat  # callback that returns agent response or None
        self._task: asyncio.Task | None = None
        self._running = False
        self._tick_count = 0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def tick_count(self) -> int:
        return self._tick_count

    async def start(self) -> None:
        """Start the heartbeat loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Heartbeat started", interval=f"{self._interval}s")

    async def stop(self) -> None:
        """Stop the heartbeat loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Heartbeat stopped", ticks=self._tick_count)

    async def _loop(self) -> None:
        """Main heartbeat loop."""
        # Wait a bit before first tick so the system fully initializes
        await asyncio.sleep(min(self._interval, 15))

        while self._running:
            try:
                self._tick_count += 1
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat tick error", error=str(e), exc_info=True)

            await asyncio.sleep(self._interval)

    async def _tick(self) -> None:
        """Single heartbeat tick."""
        await self._event_bus.emit("heartbeat", {
            "tick": self._tick_count,
        }, source="heartbeat")

        # If a callback is registered, call it (orchestrator checks for due work)
        if self._on_heartbeat:
            try:
                response = await self._on_heartbeat()
                # If the AI has something to say, emit it; otherwise discard
                if response and response.strip() != HEARTBEAT_OK:
                    await self._event_bus.emit("heartbeat_alert", {
                        "tick": self._tick_count,
                        "message": response,
                    }, source="heartbeat")
            except Exception as e:
                logger.error("Heartbeat callback error", error=str(e), exc_info=True)

    def get_stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "interval_seconds": self._interval,
            "tick_count": self._tick_count,
        }
