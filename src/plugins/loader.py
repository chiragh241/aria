"""Plugin discovery, loading, and lifecycle management."""

import importlib
import sys
from pathlib import Path
from typing import Any

from ..core.events import EventBus, get_event_bus
from ..utils.config import get_settings
from ..utils.logging import get_logger
from .base import BasePlugin, PluginManifest

logger = get_logger(__name__)


class PluginLoader:
    """Discovers, loads, and manages plugins.

    Plugins are discovered from:
      1. Built-in plugins directory (src/plugins/builtin/)
      2. User plugins directory (data/plugins/ or configured path)
      3. Installed packages with 'aria_plugin' entry point
    """

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._event_bus = event_bus or get_event_bus()
        self._plugins: dict[str, BasePlugin] = {}
        self._load_order: list[str] = []
        self.settings = get_settings()

    @property
    def plugins(self) -> dict[str, BasePlugin]:
        return dict(self._plugins)

    async def discover_and_load(self) -> None:
        """Discover and load all plugins."""
        # Built-in plugins
        builtin_dir = Path(__file__).parent / "builtin"
        if builtin_dir.exists():
            await self._load_from_directory(builtin_dir, "builtin")

        # User plugins
        user_dir = Path(self.settings.aria.data_dir).expanduser() / "plugins"
        user_dir.mkdir(parents=True, exist_ok=True)
        if user_dir.exists():
            await self._load_from_directory(user_dir, "user")

        # Entry-point plugins (pip-installed)
        await self._load_entry_points()

        logger.info(
            "Plugin discovery complete",
            loaded=len(self._plugins),
            names=list(self._plugins.keys()),
        )

    async def _load_from_directory(self, directory: Path, source: str) -> None:
        """Load plugins from a directory.

        Each plugin is either:
          - A single .py file with a class extending BasePlugin
          - A subdirectory with plugin.py containing a class extending BasePlugin
        """
        for item in sorted(directory.iterdir()):
            if item.name.startswith("_"):
                continue

            try:
                if item.is_file() and item.suffix == ".py":
                    module = self._import_file(item)
                    await self._register_from_module(module, source)
                elif item.is_dir() and (item / "plugin.py").exists():
                    module = self._import_file(item / "plugin.py")
                    await self._register_from_module(module, source)
            except Exception as e:
                logger.error(
                    "Failed to load plugin",
                    path=str(item),
                    source=source,
                    error=str(e),
                )

    def _import_file(self, path: Path) -> Any:
        """Import a Python file as a module."""
        module_name = f"aria_plugin_{path.stem}_{id(path)}"
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    async def _register_from_module(self, module: Any, source: str) -> None:
        """Find and register BasePlugin subclasses from a module."""
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BasePlugin)
                and attr is not BasePlugin
            ):
                try:
                    plugin = attr()
                    await self._register(plugin, source)
                except Exception as e:
                    logger.error(
                        "Failed to instantiate plugin",
                        class_name=attr_name,
                        error=str(e),
                    )

    async def _load_entry_points(self) -> None:
        """Load plugins from installed packages with aria_plugin entry point."""
        try:
            from importlib.metadata import entry_points

            eps = entry_points()
            aria_eps = eps.select(group="aria_plugin") if hasattr(eps, "select") else eps.get("aria_plugin", [])

            for ep in aria_eps:
                try:
                    plugin_cls = ep.load()
                    if isinstance(plugin_cls, type) and issubclass(plugin_cls, BasePlugin):
                        plugin = plugin_cls()
                        await self._register(plugin, "package")
                except Exception as e:
                    logger.error("Failed to load entry point plugin", name=ep.name, error=str(e))
        except Exception:
            pass  # No entry points available

    async def _register(self, plugin: BasePlugin, source: str) -> None:
        """Register a plugin."""
        name = plugin.name
        if name in self._plugins:
            logger.warning("Plugin already loaded, skipping duplicate", name=name)
            return

        # Check dependencies
        for dep in plugin.manifest.requires:
            if dep not in self._plugins:
                logger.warning(
                    "Plugin dependency not met",
                    plugin=name,
                    requires=dep,
                )
                return

        # Get plugin-specific config from settings
        plugin_config = {}
        if hasattr(self.settings, "plugins") and isinstance(self.settings.plugins, dict):
            plugin_config = self.settings.plugins.get(name, {})

        # Initialize
        await plugin.initialize(self._event_bus, plugin_config)

        # Auto-register event handlers
        for event_name, handler in plugin.get_event_handlers().items():
            self._event_bus.on(event_name, handler)

        self._plugins[name] = plugin
        self._load_order.append(name)

        logger.info("Plugin loaded", name=name, source=source, version=plugin.manifest.version)
        await self._event_bus.emit("plugin_loaded", {"name": name, "source": source})

    async def start_all(self) -> None:
        """Start all loaded plugins."""
        for name in self._load_order:
            plugin = self._plugins[name]
            if plugin.enabled:
                try:
                    await plugin.start()
                    logger.debug("Plugin started", name=name)
                except Exception as e:
                    logger.error("Plugin failed to start", name=name, error=str(e))

    async def stop_all(self) -> None:
        """Stop all plugins in reverse load order."""
        for name in reversed(self._load_order):
            plugin = self._plugins.get(name)
            if plugin:
                try:
                    await plugin.stop()
                except Exception as e:
                    logger.error("Plugin failed to stop", name=name, error=str(e))

    def get_plugin(self, name: str) -> BasePlugin | None:
        return self._plugins.get(name)

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Collect tools from all enabled plugins."""
        tools = []
        for plugin in self._plugins.values():
            if plugin.enabled:
                for tool in plugin.get_tools():
                    tool["_plugin"] = plugin.name
                    tools.append(tool)
        return tools

    async def execute_plugin_tool(self, plugin_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool on a specific plugin."""
        plugin = self._plugins.get(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_name}")
        if not plugin.enabled:
            raise ValueError(f"Plugin is disabled: {plugin_name}")
        return await plugin.execute_tool(tool_name, arguments)

    def list_plugins(self) -> list[dict[str, Any]]:
        """List all loaded plugins with metadata."""
        result = []
        for name, plugin in self._plugins.items():
            m = plugin.manifest
            result.append({
                "name": m.name,
                "version": m.version,
                "description": m.description,
                "author": m.author,
                "enabled": plugin.enabled,
                "requires": m.requires,
                "provides_skills": m.provides_skills,
                "provides_tools": m.provides_tools,
                "tool_count": len(plugin.get_tools()),
            })
        return result
