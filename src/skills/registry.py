"""Skill registry for discovering and managing skills."""

import importlib
from pathlib import Path
from typing import Any

from ..utils.config import get_settings
from ..utils.logging import get_logger
from .base import BaseSkill, SkillResult

logger = get_logger(__name__)


class SkillRegistry:
    """
    Registry for discovering, loading, and managing skills.

    Features:
    - Auto-discovery of built-in skills
    - Loading of learned/custom skills
    - Enable/disable controls
    - Skill execution routing
    """

    # Metadata for all built-in skills (always visible to the UI)
    BUILTIN_SKILL_META: dict[str, dict[str, Any]] = {
        "filesystem": {
            "description": "Read, write, and manage files and directories",
            "version": "1.0.0",
            "capabilities": [
                {"name": "read_file", "description": "Read contents of a file"},
                {"name": "write_file", "description": "Write content to a file"},
                {"name": "list_directory", "description": "List files in a directory"},
                {"name": "delete_file", "description": "Delete a file or directory"},
                {"name": "search_files", "description": "Search for files by pattern"},
            ],
        },
        "shell": {
            "description": "Execute shell commands securely",
            "version": "1.0.0",
            "capabilities": [
                {"name": "execute", "description": "Run a shell command"},
                {"name": "execute_script", "description": "Run a multi-line script"},
            ],
        },
        "browser": {
            "description": "Web browsing, search, and page automation",
            "version": "1.0.0",
            "capabilities": [
                {"name": "navigate", "description": "Navigate to a URL"},
                {"name": "search", "description": "Search the web"},
                {"name": "screenshot", "description": "Take a page screenshot"},
                {"name": "extract_text", "description": "Extract text from a page"},
            ],
        },
        "calendar": {
            "description": "Google Calendar integration for events and scheduling",
            "version": "1.0.0",
            "capabilities": [
                {"name": "list_events", "description": "List upcoming events"},
                {"name": "create_event", "description": "Create a new event"},
                {"name": "update_event", "description": "Update an existing event"},
                {"name": "delete_event", "description": "Delete an event"},
            ],
        },
        "email": {
            "description": "Send and receive emails via SMTP/IMAP",
            "version": "1.0.0",
            "capabilities": [
                {"name": "send_email", "description": "Send an email"},
                {"name": "read_inbox", "description": "Read recent emails"},
                {"name": "search_email", "description": "Search emails"},
            ],
        },
        "sms": {
            "description": "Send SMS messages via Twilio",
            "version": "1.0.0",
            "capabilities": [
                {"name": "send_sms", "description": "Send an SMS message"},
            ],
        },
        "tts": {
            "description": "Convert text to speech audio",
            "version": "1.0.0",
            "capabilities": [
                {"name": "synthesize", "description": "Convert text to speech"},
                {"name": "list_voices", "description": "List available voices"},
            ],
        },
        "stt": {
            "description": "Convert speech audio to text",
            "version": "1.0.0",
            "capabilities": [
                {"name": "transcribe", "description": "Transcribe audio to text"},
            ],
        },
        "image": {
            "description": "Image processing, conversion, and analysis",
            "version": "1.0.0",
            "capabilities": [
                {"name": "resize", "description": "Resize an image"},
                {"name": "convert", "description": "Convert image format"},
                {"name": "analyze", "description": "Analyze image contents"},
            ],
        },
        "video": {
            "description": "Video processing and conversion with FFmpeg",
            "version": "1.0.0",
            "capabilities": [
                {"name": "convert", "description": "Convert video format"},
                {"name": "trim", "description": "Trim a video clip"},
                {"name": "extract_audio", "description": "Extract audio from video"},
                {"name": "thumbnail", "description": "Generate video thumbnail"},
            ],
        },
        "documents": {
            "description": "Process PDFs, Word docs, spreadsheets, and more",
            "version": "1.0.0",
            "capabilities": [
                {"name": "extract_text", "description": "Extract text from a document"},
                {"name": "convert", "description": "Convert document format"},
                {"name": "summarize", "description": "Summarize document contents"},
            ],
        },
        "memory": {
            "description": "Remember facts, recall memories, and manage what the assistant knows about you",
            "version": "1.0.0",
            "capabilities": [
                {"name": "remember", "description": "Store a fact about the user"},
                {"name": "recall", "description": "Search memory for facts"},
                {"name": "forget", "description": "Remove a stored fact"},
                {"name": "list_memories", "description": "List all stored facts"},
            ],
        },
        "weather": {
            "description": "Get current weather, forecasts, and weather alerts",
            "version": "1.0.0",
            "capabilities": [
                {"name": "current", "description": "Get current weather for a location"},
                {"name": "forecast", "description": "Get 5-day forecast"},
                {"name": "alerts", "description": "Get severe weather alerts"},
            ],
        },
        "news": {
            "description": "Get top news headlines, search news, and summarize articles",
            "version": "1.0.0",
            "capabilities": [
                {"name": "headlines", "description": "Get top headlines"},
                {"name": "search", "description": "Search news by keyword"},
                {"name": "summarize", "description": "Summarize an article URL"},
            ],
        },
        "finance": {
            "description": "Check stock prices, crypto prices, and market overview",
            "version": "1.0.0",
            "capabilities": [
                {"name": "stock_price", "description": "Get stock price"},
                {"name": "crypto_price", "description": "Get crypto price"},
                {"name": "market_summary", "description": "Get market overview"},
            ],
        },
        "contacts": {
            "description": "Store and search contacts (name, phone, email)",
            "version": "1.0.0",
            "capabilities": [
                {"name": "add_contact", "description": "Add a contact"},
                {"name": "find_contact", "description": "Search contacts"},
                {"name": "list_contacts", "description": "List all contacts"},
            ],
        },
        "tracking": {
            "description": "Track packages by tracking number",
            "version": "1.0.0",
            "capabilities": [
                {"name": "track", "description": "Track package by number"},
                {"name": "list_packages", "description": "List tracked packages"},
            ],
        },
        "home": {
            "description": "Control smart home devices via Home Assistant",
            "version": "1.0.0",
            "capabilities": [
                {"name": "list_devices", "description": "List HA entities"},
                {"name": "turn_on", "description": "Turn on entity"},
                {"name": "turn_off", "description": "Turn off entity"},
                {"name": "set_climate", "description": "Set thermostat"},
            ],
        },
        "webhook": {
            "description": "Send HTTP requests to webhooks",
            "version": "1.0.0",
            "capabilities": [
                {"name": "fire", "description": "Send HTTP request"},
                {"name": "list_webhooks", "description": "List saved webhooks"},
            ],
        },
        "agent": {
            "description": "Autonomous research, coding, and data analysis agents",
            "version": "1.0.0",
            "capabilities": [
                {"name": "research", "description": "Research a topic"},
                {"name": "code", "description": "Write and run code"},
                {"name": "analyze", "description": "Analyze data"},
            ],
        },
        "research": {
            "description": "Search Reddit, X (Twitter), and the web for any topic",
            "version": "1.0.0",
            "capabilities": [
                {"name": "search_topic", "description": "Research across Reddit, web, and X"},
                {"name": "search_reddit", "description": "Search Reddit"},
                {"name": "search_web", "description": "Search the web"},
                {"name": "search_x", "description": "Search X (Twitter)"},
            ],
        },
    }

    def __init__(
        self,
        sandbox_manager: Any = None,
        security_guardian: Any = None,
        audit_logger: Any = None,
    ) -> None:
        self.settings = get_settings()
        self._skills: dict[str, BaseSkill] = {}
        self._skill_enabled: dict[str, bool] = {}  # Tracks enable state for ALL skills
        self._initialized = False
        self._sandbox_manager = sandbox_manager
        self._security_guardian = security_guardian
        self._audit_logger = audit_logger

    async def initialize(self) -> None:
        """Initialize the registry and load skills."""
        if self._initialized:
            return

        # Load built-in skills
        await self._load_builtin_skills()

        # Load learned skills
        await self._load_learned_skills()

        self._initialized = True
        logger.info("Skill registry initialized", skill_count=len(self._skills))

    async def _load_builtin_skills(self) -> None:
        """Load all built-in skills."""
        builtin_config = self.settings.skills.builtin

        skill_classes = [
            ("filesystem", "FilesystemSkill", builtin_config.filesystem),
            ("shell", "ShellSkill", builtin_config.shell),
            ("browser", "BrowserSkill", builtin_config.browser),
            ("calendar", "CalendarSkill", builtin_config.calendar),
            ("email", "EmailSkill", builtin_config.email),
            ("sms", "SMSSkill", builtin_config.sms),
            ("tts", "TTSSkill", builtin_config.tts),
            ("stt", "STTSkill", builtin_config.stt),
            ("image", "ImageSkill", builtin_config.image),
            ("video", "VideoSkill", builtin_config.video),
            ("documents", "DocumentsSkill", builtin_config.documents),
            ("memory", "MemorySkill", builtin_config.memory),
            ("weather", "WeatherSkill", builtin_config.weather),
            ("news", "NewsSkill", builtin_config.news),
            ("finance", "FinanceSkill", builtin_config.finance),
            ("contacts", "ContactsSkill", builtin_config.contacts),
            ("tracking", "TrackingSkill", builtin_config.tracking),
            ("home", "HomeSkill", builtin_config.home),
            ("webhook", "WebhookSkill", builtin_config.webhook),
            ("agent", "AgentSkill", builtin_config.agent),
            ("research", "ResearchSkill", builtin_config.research),
            ("notion", "NotionSkill", builtin_config.notion),
            ("todoist", "TodoistSkill", builtin_config.todoist),
            ("linear", "LinearSkill", builtin_config.linear),
            ("spotify", "SpotifySkill", builtin_config.spotify),
        ]

        for skill_name, class_name, config in skill_classes:
            is_enabled = config.get("enabled", True)
            self._skill_enabled[skill_name] = is_enabled

            if not is_enabled:
                logger.debug("Skill disabled", skill=skill_name)
                continue

            try:
                module = importlib.import_module(f"..skills.builtin.{skill_name}", __package__)
                skill_class = getattr(module, class_name)
                skill = skill_class(config)
                await skill.initialize()
                self._skills[skill_name] = skill
                logger.debug("Loaded skill", skill=skill_name)
            except ImportError as e:
                logger.warning(
                    "Failed to import skill module",
                    skill=skill_name,
                    error=str(e),
                )
            except Exception as e:
                logger.error(
                    "Failed to load skill",
                    skill=skill_name,
                    error=str(e),
                )

    async def _load_learned_skills(self) -> None:
        """Load dynamically created skills."""
        if not self.settings.skills.learned.enabled:
            return

        learned_dir = Path(self.settings.skills.learned.directory).expanduser()
        if not learned_dir.exists():
            learned_dir.mkdir(parents=True, exist_ok=True)
            return

        for skill_file in learned_dir.glob("*.py"):
            if skill_file.name.startswith("_"):
                continue

            try:
                await self._load_skill_from_file(skill_file)
            except Exception as e:
                logger.error(
                    "Failed to load learned skill",
                    file=str(skill_file),
                    error=str(e),
                )

    async def _load_skill_from_file(self, skill_file: Path) -> None:
        """Load a skill from a Python file."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            skill_file.stem,
            skill_file,
        )
        if not spec or not spec.loader:
            return

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find skill class in module
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseSkill)
                and obj != BaseSkill
            ):
                skill = obj()
                await skill.initialize()
                self._skills[skill.name] = skill
                logger.info("Loaded learned skill", skill=skill.name)
                break

    async def reload_learned_skill(self, skill_name: str) -> bool:
        """
        Load or reload a single learned skill by name.

        Args:
            skill_name: Name of the skill (filename without .py)

        Returns:
            True if loaded successfully
        """
        if not self.settings.skills.learned.enabled:
            return False
        learned_dir = Path(self.settings.skills.learned.directory).expanduser()
        skill_file = learned_dir / f"{skill_name}.py"
        if not skill_file.exists():
            return False
        try:
            await self._load_skill_from_file(skill_file)
            return True
        except Exception as e:
            logger.error(
                "Failed to reload learned skill",
                skill=skill_name,
                error=str(e),
            )
            return False

    def register(self, skill: BaseSkill) -> None:
        """
        Register a skill manually.

        Args:
            skill: The skill instance to register
        """
        self._skills[skill.name] = skill
        logger.info("Registered skill", skill=skill.name)

    def unregister(self, skill_name: str) -> bool:
        """
        Unregister a skill.

        Args:
            skill_name: Name of the skill to remove

        Returns:
            True if skill was removed
        """
        if skill_name in self._skills:
            del self._skills[skill_name]
            logger.info("Unregistered skill", skill=skill_name)
            return True
        return False

    def get_skill(self, name: str) -> BaseSkill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> list[dict[str, Any]]:
        """List all known skills, including disabled ones."""
        result = []
        # Include all built-in skills (loaded or not)
        for name, meta in self.BUILTIN_SKILL_META.items():
            if name in self._skills:
                result.append(self._skills[name].to_dict())
            else:
                # Provide metadata for skills that aren't loaded
                result.append({
                    "name": name,
                    "description": meta["description"],
                    "version": meta["version"],
                    "enabled": self._skill_enabled.get(name, False),
                    "capabilities": meta["capabilities"],
                    "initialized": False,
                })
        # Include any loaded non-builtin (learned) skills
        for name, skill in self._skills.items():
            if name not in self.BUILTIN_SKILL_META:
                result.append(skill.to_dict())
        return result

    def list_capabilities(self) -> list[dict[str, Any]]:
        """List all capabilities across all skills."""
        capabilities = []
        for skill in self._skills.values():
            if not skill.enabled:
                continue
            for cap in skill.get_capabilities():
                capabilities.append({
                    "skill": skill.name,
                    "capability": cap["name"],
                    "description": cap["description"],
                    "parameters": cap["parameters"],
                })
        return capabilities

    async def execute(
        self,
        skill_name: str,
        capability: str,
        **kwargs: Any,
    ) -> SkillResult:
        """
        Execute a skill capability.

        Args:
            skill_name: Name of the skill
            capability: Name of the capability
            **kwargs: Capability parameters

        Returns:
            SkillResult with execution outcome
        """
        skill = self._skills.get(skill_name)
        if not skill:
            return SkillResult(
                success=False,
                error=f"Skill not found: {skill_name}",
            )

        if not skill.enabled:
            return SkillResult(
                success=False,
                error=f"Skill is disabled: {skill_name}",
            )

        try:
            return await skill.execute(capability, **kwargs)
        except Exception as e:
            logger.error(
                "Skill execution failed",
                skill=skill_name,
                capability=capability,
                error=str(e),
            )
            return SkillResult(
                success=False,
                error=str(e),
            )

    def enable_skill(self, skill_name: str) -> bool:
        """Enable a skill and try to hot-load it if not already loaded."""
        self._skill_enabled[skill_name] = True
        skill = self._skills.get(skill_name)
        if skill:
            skill.enabled = True
            return True
        # Skill not loaded yet — try to hot-load it
        if skill_name in self.BUILTIN_SKILL_META:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._hot_load_skill(skill_name))
                else:
                    loop.run_until_complete(self._hot_load_skill(skill_name))
            except Exception as e:
                logger.warning("Could not hot-load skill", skill=skill_name, error=str(e))
            return True
        return False

    def disable_skill(self, skill_name: str) -> bool:
        """Disable a skill."""
        self._skill_enabled[skill_name] = False
        skill = self._skills.get(skill_name)
        if skill:
            skill.enabled = False
        return skill_name in self.BUILTIN_SKILL_META or skill is not None

    async def _hot_load_skill(self, skill_name: str) -> bool:
        """Dynamically load a built-in skill at runtime."""
        class_map = {
            "filesystem": "FilesystemSkill",
            "shell": "ShellSkill",
            "browser": "BrowserSkill",
            "calendar": "CalendarSkill",
            "email": "EmailSkill",
            "sms": "SMSSkill",
            "tts": "TTSSkill",
            "stt": "STTSkill",
            "image": "ImageSkill",
            "video": "VideoSkill",
            "documents": "DocumentsSkill",
            "memory": "MemorySkill",
            "weather": "WeatherSkill",
            "news": "NewsSkill",
            "finance": "FinanceSkill",
            "contacts": "ContactsSkill",
            "tracking": "TrackingSkill",
            "home": "HomeSkill",
            "webhook": "WebhookSkill",
            "agent": "AgentSkill",
            "research": "ResearchSkill",
        }
        class_name = class_map.get(skill_name)
        if not class_name:
            return False
        try:
            builtin_config = self.settings.skills.builtin
            config = getattr(builtin_config, skill_name, {})
            module = importlib.import_module(f"..skills.builtin.{skill_name}", __package__)
            skill_class = getattr(module, class_name)
            skill = skill_class(config)
            await skill.initialize()
            skill.enabled = True
            self._skills[skill_name] = skill
            logger.info("Hot-loaded skill", skill=skill_name)
            return True
        except Exception as e:
            logger.error("Failed to hot-load skill", skill=skill_name, error=str(e))
            return False

    async def shutdown(self) -> None:
        """Shutdown all skills."""
        for skill in self._skills.values():
            try:
                await skill.shutdown()
            except Exception as e:
                logger.error("Skill shutdown failed", skill=skill.name, error=str(e))

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        enabled = sum(1 for s in self._skills.values() if s.enabled)
        total_capabilities = sum(
            len(s.get_capabilities()) for s in self._skills.values()
        )

        return {
            "total_skills": len(self._skills),
            "enabled_skills": enabled,
            "total_capabilities": total_capabilities,
            "skill_names": list(self._skills.keys()),
        }
