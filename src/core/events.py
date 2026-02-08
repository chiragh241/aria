"""Event system for Aria — the nervous system that connects all components.

Provides a publish/subscribe event bus. Any component can emit events and
any component can listen. Supports both sync and async handlers.

Usage:
    bus = EventBus()
    bus.on("message_received", my_handler)
    await bus.emit("message_received", {"channel": "slack", "content": "hi"})
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Type alias for event handlers
EventHandler = Callable[..., Coroutine[Any, Any, None]] | Callable[..., None]


@dataclass
class Event:
    """An event that flows through the system."""

    name: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""


# Well-known event names (not enforced, just documented)
# message_received   - A message arrived on any channel
# message_sent       - A response was sent to a channel
# tool_executed      - A tool/skill was called
# tool_failed        - A tool/skill call failed
# heartbeat          - Periodic heartbeat tick
# cron_fired         - A scheduled job executed
# process_started    - A background process was spawned
# process_completed  - A background process finished
# config_changed     - Settings were reloaded
# channel_connected  - A channel came online
# channel_disconnected - A channel went offline
# skill_loaded       - A skill was loaded/enabled
# plugin_loaded      - A plugin was loaded
# reminder_due       - A reminder is due for delivery


class EventBus:
    """Central event bus for publish/subscribe messaging."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._once_handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._history: list[Event] = []
        self._max_history = 200

    def on(self, event_name: str, handler: EventHandler) -> Callable[[], None]:
        """Subscribe to an event. Returns an unsubscribe function."""
        self._handlers[event_name].append(handler)

        def unsubscribe() -> None:
            try:
                self._handlers[event_name].remove(handler)
            except ValueError:
                pass

        return unsubscribe

    def once(self, event_name: str, handler: EventHandler) -> None:
        """Subscribe to an event — handler fires once then auto-removes."""
        self._once_handlers[event_name].append(handler)

    def off(self, event_name: str, handler: EventHandler) -> None:
        """Unsubscribe a handler."""
        try:
            self._handlers[event_name].remove(handler)
        except ValueError:
            pass
        try:
            self._once_handlers[event_name].remove(handler)
        except ValueError:
            pass

    async def emit(self, event_name: str, data: dict[str, Any] | None = None, source: str = "") -> None:
        """Emit an event to all subscribers."""
        event = Event(name=event_name, data=data or {}, source=source)

        # Store in history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Collect all handlers
        handlers = list(self._handlers.get(event_name, []))
        once_handlers = list(self._once_handlers.pop(event_name, []))
        # Also fire wildcard "*" handlers
        handlers += list(self._handlers.get("*", []))

        all_handlers = handlers + once_handlers

        if not all_handlers:
            return

        # Execute all handlers, catching errors so one bad handler doesn't break others
        for handler in all_handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(
                    "Event handler error",
                    event=event_name,
                    handler=getattr(handler, "__name__", str(handler)),
                    error=str(e),
                    exc_info=True,
                )

    def get_history(self, event_name: str | None = None, limit: int = 50) -> list[Event]:
        """Get recent event history, optionally filtered by name."""
        if event_name:
            filtered = [e for e in self._history if e.name == event_name]
        else:
            filtered = list(self._history)
        return filtered[-limit:]

    def handler_count(self, event_name: str | None = None) -> int:
        """Count registered handlers."""
        if event_name:
            return len(self._handlers.get(event_name, [])) + len(self._once_handlers.get(event_name, []))
        return sum(len(h) for h in self._handlers.values()) + sum(len(h) for h in self._once_handlers.values())

    def clear(self) -> None:
        """Remove all handlers and history."""
        self._handlers.clear()
        self._once_handlers.clear()
        self._history.clear()


# Global singleton
_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus singleton."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
