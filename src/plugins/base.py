"""Base plugin class that all Aria plugins must extend."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..core.events import EventBus
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PluginManifest:
    """Metadata describing a plugin."""

    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    requires: list[str] = field(default_factory=list)  # Other plugin names
    provides_skills: list[str] = field(default_factory=list)
    provides_tools: list[str] = field(default_factory=list)
    config_schema: dict[str, Any] = field(default_factory=dict)


class BasePlugin(ABC):
    """Abstract base for all Aria plugins.

    Lifecycle:
        1. __init__()  — called once when discovered
        2. initialize() — called with app context (event bus, config)
        3. start()      — called when Aria is ready
        4. stop()       — called on shutdown
    """

    def __init__(self) -> None:
        self._event_bus: EventBus | None = None
        self._config: dict[str, Any] = {}
        self._enabled = True

    @property
    @abstractmethod
    def manifest(self) -> PluginManifest:
        """Return plugin metadata."""
        ...

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    async def initialize(
        self,
        event_bus: EventBus,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with app context. Override to set up resources."""
        self._event_bus = event_bus
        self._config = config or {}

    async def start(self) -> None:
        """Called when Aria is ready. Override to start background work."""
        pass

    async def stop(self) -> None:
        """Called on shutdown. Override to clean up."""
        pass

    def get_tools(self) -> list[dict[str, Any]]:
        """Return tool definitions this plugin provides (LLM-callable).

        Each tool dict: {"name": str, "description": str, "parameters": dict}
        """
        return []

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name. Override if get_tools() returns tools."""
        raise NotImplementedError(f"Tool {tool_name} not implemented")

    def get_event_handlers(self) -> dict[str, Any]:
        """Return event name → handler mappings to auto-register.

        Example: {"message_received": self.on_message}
        """
        return {}

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a plugin config value."""
        return self._config.get(key, default)
