"""Config hot-reload — watches settings.yaml for changes and emits events.

When the config file is modified (by the Settings UI, CLI, or manual edit),
this watcher detects the change and emits a `config_changed` event so all
components can react without a restart.
"""

import asyncio
import time
from pathlib import Path
from typing import Any

from ..utils.config import get_settings, reload_settings
from ..utils.logging import get_logger
from .events import EventBus, get_event_bus

logger = get_logger(__name__)


class ConfigWatcher:
    """Watches settings.yaml for changes and triggers hot-reload.

    Uses polling (every N seconds) rather than inotify/kqueue for
    cross-platform compatibility and Docker volume support.
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        poll_interval: float = 3.0,
    ) -> None:
        self._event_bus = event_bus or get_event_bus()
        self._poll_interval = poll_interval
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._last_mtime: float = 0.0
        self._config_path = Path("config/settings.yaml")
        self._env_path = Path(".env")
        self._last_env_mtime: float = 0.0
        self._reload_callbacks: list[Any] = []

    def on_reload(self, callback: Any) -> None:
        """Register a callback for config reload events."""
        self._reload_callbacks.append(callback)

    async def start(self) -> None:
        """Start watching for config changes."""
        if self._running:
            return

        # Record initial mtimes
        self._last_mtime = self._get_mtime(self._config_path)
        self._last_env_mtime = self._get_mtime(self._env_path)

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "Config watcher started",
            config_path=str(self._config_path),
            poll_interval=self._poll_interval,
        )

    async def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Config watcher stopped")

    async def _poll_loop(self) -> None:
        """Main poll loop — checks file mtimes periodically."""
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)

                changed_files = []

                # Check settings.yaml
                current_mtime = self._get_mtime(self._config_path)
                if current_mtime != self._last_mtime and self._last_mtime > 0:
                    changed_files.append("settings.yaml")
                self._last_mtime = current_mtime  # Always update (catches creation)

                # Check .env
                current_env_mtime = self._get_mtime(self._env_path)
                if current_env_mtime != self._last_env_mtime and self._last_env_mtime > 0:
                    changed_files.append(".env")
                self._last_env_mtime = current_env_mtime

                if changed_files:
                    await self._handle_change(changed_files)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Config watcher error", error=str(e))
                await asyncio.sleep(self._poll_interval * 2)

    async def _handle_change(self, changed_files: list[str]) -> None:
        """Handle a detected config change."""
        logger.info("Config change detected", files=changed_files)

        # Reload .env if changed
        if ".env" in changed_files:
            try:
                from dotenv import load_dotenv
                load_dotenv(dotenv_path=self._env_path, override=True)
                logger.info("Reloaded .env file")
            except Exception as e:
                logger.error("Failed to reload .env", error=str(e))

        # Reload settings (clears lru_cache and re-reads from disk)
        try:
            new_settings = reload_settings()
        except Exception as e:
            logger.error("Failed to parse updated config", error=str(e))
            return

        # Emit event
        await self._event_bus.emit(
            "config_changed",
            {"files": changed_files},
            source="config_watcher",
        )

        # Call registered callbacks
        for cb in self._reload_callbacks:
            try:
                result = cb(new_settings)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Config reload callback error", error=str(e))

    @staticmethod
    def _get_mtime(path: Path) -> float:
        """Get file modification time, or 0 if not found."""
        try:
            return path.stat().st_mtime
        except (FileNotFoundError, OSError):
            return 0.0

    async def force_reload(self) -> None:
        """Manually trigger a config reload."""
        await self._handle_change(["settings.yaml", ".env"])
