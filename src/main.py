"""
Aria - Personal AI Assistant
Main Entry Point

This module initializes and starts all components of the Aria system.
"""

import argparse
import asyncio
import signal
import socket
import sys
import webbrowser
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env BEFORE any other imports that read os.environ or settings
load_dotenv(dotenv_path=Path("." ) / ".env", override=True)

import structlog
import uvicorn
from uvicorn import Config, Server

from src.utils.config import get_settings
from src.utils.logging import setup_logging
from src.core.orchestrator import Orchestrator
from src.core.llm_router import LLMRouter
from src.core.context_manager import ContextManager
from src.core.message_router import MessageRouter
from src.security.guardian import SecurityGuardian
from src.security.audit import AuditLogger
from src.security.sandbox import SandboxManager
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.memory.episodic import EpisodicMemory
from src.memory.rag import RAGPipeline
from src.memory.cognee_graph import CogneeGraphMemory
from src.skills.registry import SkillRegistry
from src.channels.slack import SlackChannel
from src.channels.whatsapp import WhatsAppChannel
from src.channels.websocket import WebSocketChannel
from src.core.events import get_event_bus
from src.core.process_manager import ProcessManager
from src.core.heartbeat import HeartbeatService
from src.core.scheduler import Scheduler
from src.core.config_watcher import ConfigWatcher
from src.core.device_pairing import DevicePairingManager
from src.memory.vector_store import VectorMemory
from src.plugins.loader import PluginLoader
from src.web.api import create_app

logger = structlog.get_logger()


def is_first_run() -> bool:
    """Check if Aria has been configured before."""
    return not Path("data/.aria_configured").exists()


class AriaApplication:
    """Main Aria application that manages all components."""

    def __init__(self):
        self.settings = get_settings()
        self.running = False
        self.shutdown_event = asyncio.Event()

        # Core components
        self.llm_router: Optional[LLMRouter] = None
        self.context_manager: Optional[ContextManager] = None
        self.message_router: Optional[MessageRouter] = None
        self.orchestrator: Optional[Orchestrator] = None

        # Security components
        self.security_guardian: Optional[SecurityGuardian] = None
        self.audit_logger: Optional[AuditLogger] = None
        self.sandbox_manager: Optional[SandboxManager] = None

        # Memory components
        self.short_term_memory: Optional[ShortTermMemory] = None
        self.long_term_memory: Optional[LongTermMemory] = None
        self.episodic_memory: Optional[EpisodicMemory] = None
        self.cognee_memory: Optional[CogneeGraphMemory] = None
        self.rag_pipeline: Optional[RAGPipeline] = None

        # Skills
        self.skill_registry: Optional[SkillRegistry] = None

        # Channels
        self.channels = []
        self.websocket_channel: Optional[WebSocketChannel] = None

        # New systems
        self.event_bus = get_event_bus()
        self.process_manager: Optional[ProcessManager] = None
        self.heartbeat: Optional[HeartbeatService] = None
        self.scheduler: Optional[Scheduler] = None
        self.config_watcher: Optional[ConfigWatcher] = None
        self.device_manager: Optional[DevicePairingManager] = None
        self.vector_memory: Optional[VectorMemory] = None
        self.plugin_loader: Optional[PluginLoader] = None

        # Web server
        self.web_server: Optional[Server] = None

    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing Aria...")

        # Initialize security components first
        logger.info("Initializing security components...")
        self.audit_logger = AuditLogger()
        await self.audit_logger.initialize()

        # Bridge file-based audit logger to database logger
        from src.utils.logging import AuditLogger as FileAuditLogger
        FileAuditLogger.register_db_logger(self.audit_logger)

        self.security_guardian = SecurityGuardian()
        await self.security_guardian.initialize()

        self.sandbox_manager = SandboxManager()
        await self.sandbox_manager.initialize()

        # Initialize memory components
        logger.info("Initializing memory components...")
        self.short_term_memory = ShortTermMemory()
        await self.short_term_memory.initialize()

        self.long_term_memory = LongTermMemory()
        await self.long_term_memory.initialize()

        self.episodic_memory = EpisodicMemory()
        await self.episodic_memory.initialize()

        self.cognee_memory = CogneeGraphMemory(
            data_dir=self.settings.aria.data_dir
        )
        await self.cognee_memory.initialize()

        self.rag_pipeline = RAGPipeline(
            memory=self.long_term_memory,
            cognee_memory=self.cognee_memory,
        )
        await self.rag_pipeline.initialize()

        # Initialize core components
        logger.info("Initializing core components...")
        self.llm_router = LLMRouter()
        await self.llm_router.initialize()

        self.context_manager = ContextManager()

        # Phase 1: User profile, entity extraction, summarizer
        from src.memory.user_profile import UserProfileManager
        from src.processing.entity_extractor import EntityExtractor
        from src.processing.summarizer import ConversationSummarizer
        from src.processing.sentiment import SentimentAnalyzer
        from src.processing.personality import PersonalityAdapter

        self.user_profile_manager = UserProfileManager()
        self.entity_extractor = EntityExtractor()
        self.conversation_summarizer = ConversationSummarizer(llm_router=self.llm_router)
        self.conversation_summarizer.set_llm_router(self.llm_router)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.personality_adapter = PersonalityAdapter()

        self.message_router = MessageRouter()
        await self.message_router.initialize()

        # Initialize skills
        logger.info("Initializing skill registry...")
        self.skill_registry = SkillRegistry(
            sandbox_manager=self.sandbox_manager,
            security_guardian=self.security_guardian,
            audit_logger=self.audit_logger
        )
        await self.skill_registry.initialize()

        # Initialize orchestrator
        logger.info("Initializing orchestrator...")
        self.orchestrator = Orchestrator(
            llm_router=self.llm_router,
            context_manager=self.context_manager,
            skill_registry=self.skill_registry,
            security_guardian=self.security_guardian,
            rag_pipeline=self.rag_pipeline,
            audit_logger=self.audit_logger
        )
        await self.orchestrator.initialize()

        # Wire skill generator for auto-learning
        from src.skills.generator import SkillGenerator
        skill_generator = SkillGenerator(llm_router=self.llm_router)
        self.orchestrator.set_skill_generator(skill_generator)

        # Vector memory will be wired after initialization below

        # Initialize new systems
        logger.info("Initializing new subsystems...")

        # Event bus is already a singleton — just get it
        self.event_bus = get_event_bus()

        # Background process manager
        self.process_manager = ProcessManager(event_bus=self.event_bus)

        # Heartbeat service
        self.heartbeat = HeartbeatService(event_bus=self.event_bus)

        # Scheduler (cron / reminders) — auto-starts by listening to heartbeat events
        self.scheduler = Scheduler(event_bus=self.event_bus)

        # Config hot-reload watcher
        self.config_watcher = ConfigWatcher(event_bus=self.event_bus)

        # Device pairing
        self.device_manager = DevicePairingManager()
        await self.device_manager.initialize()

        # Vector memory (ChromaDB)
        self.vector_memory = VectorMemory()
        await self.vector_memory.initialize()

        # Plugin loader
        self.plugin_loader = PluginLoader(event_bus=self.event_bus)
        await self.plugin_loader.discover_and_load()

        # Wire vector memory to orchestrator
        if self.vector_memory and self.vector_memory.available:
            self.orchestrator.set_vector_memory(self.vector_memory)

        # Wire context manager summarizer and vector memory for auto-summarize on trim
        self.context_manager.set_summarizer(self.conversation_summarizer)
        self.context_manager.set_vector_memory(self.vector_memory)

        # Wire Phase 1 & 5 components to orchestrator
        self.orchestrator.set_user_profile_manager(self.user_profile_manager)
        self.orchestrator.set_entity_extractor(self.entity_extractor)
        self.orchestrator.set_summarizer(self.conversation_summarizer)
        self.orchestrator.set_sentiment_analyzer(self.sentiment_analyzer)
        self.orchestrator.set_personality_adapter(self.personality_adapter)

        # Wire memory skill with profile manager and vector memory
        memory_skill = self.skill_registry.get_skill("memory")
        if memory_skill:
            memory_skill.set_profile_manager(self.user_profile_manager)
            memory_skill.set_vector_memory(self.vector_memory)

        # Phase 4: Proactive engine
        from src.core.proactive import ProactiveEngine
        self.proactive_engine = ProactiveEngine(
            event_bus=self.event_bus,
            orchestrator=self.orchestrator,
            skill_registry=self.skill_registry,
            scheduler=self.scheduler,
        )
        await self.proactive_engine.start()

        # Add default morning briefing job if enabled and not exists
        if self.settings.proactive.morning_briefing:
            jobs = self.scheduler.list_jobs()
            if not any(j.get("payload", {}).get("type") == "morning_briefing" for j in jobs):
                self.scheduler.add_job(
                    name="Morning Briefing",
                    schedule_type="cron",
                    schedule_value="0 8 * * *",
                    action="agent_turn",
                    payload={"type": "morning_briefing"},
                )

        # Phase 6: Agent coordinator
        from src.core.agent_coordinator import AgentCoordinator
        self.agent_coordinator = AgentCoordinator(
            skill_registry=self.skill_registry,
            llm_router=self.llm_router,
            event_bus=self.event_bus,
        )
        agent_skill = self.skill_registry.get_skill("agent")
        if agent_skill:
            agent_skill.set_coordinator(self.agent_coordinator)

        # Initialize channels
        logger.info("Initializing channels...")
        await self._initialize_channels()

        # Ensure web UI frontend is built
        self._ensure_frontend_built()

        # Create web application
        logger.info("Creating web application...")
        self.app = create_app(
            orchestrator=self.orchestrator,
            skill_registry=self.skill_registry,
            security_guardian=self.security_guardian,
            audit_logger=self.audit_logger
        )

        # Attach new subsystems to app.state so API endpoints can access them
        self.app.state.process_manager = self.process_manager
        self.app.state.scheduler = self.scheduler
        self.app.state.vector_memory = self.vector_memory
        self.app.state.plugin_loader = self.plugin_loader
        self.app.state.device_manager = self.device_manager
        self.app.state.proactive_engine = self.proactive_engine
        self.app.state.agent_coordinator = self.agent_coordinator

        logger.info("Aria initialization complete")

    async def _initialize_channels(self) -> None:
        """Initialize messaging channels based on configuration."""
        channels_config = self.settings.channels

        # Slack channel
        if channels_config.slack.enabled:
            try:
                import os
                bot_token = self.settings.slack_bot_token or channels_config.slack.bot_token
                app_token = self.settings.slack_app_token or channels_config.slack.app_token
                if not bot_token:
                    logger.error("Slack enabled but SLACK_BOT_TOKEN is not set in .env")
                elif not app_token:
                    logger.error("Slack enabled but SLACK_APP_TOKEN is not set in .env")
                else:
                    logger.info(
                        "Creating Slack channel",
                        bot_token_prefix=bot_token[:10] + "..." if bot_token else "EMPTY",
                        app_token_prefix=app_token[:10] + "..." if app_token else "EMPTY",
                    )
                    slack_channel = SlackChannel()
                    self.channels.append(slack_channel)
                    logger.info("Slack channel created")
            except Exception as e:
                logger.error("Failed to create Slack channel", error=str(e), exc_info=True)

        # WhatsApp channel
        if channels_config.whatsapp.enabled:
            try:
                logger.info(
                    "Creating WhatsApp channel",
                    bridge_host=channels_config.whatsapp.bridge_host,
                    bridge_port=channels_config.whatsapp.bridge_port,
                )
                whatsapp_channel = WhatsAppChannel()
                self.channels.append(whatsapp_channel)
                logger.info("WhatsApp channel created")
            except Exception as e:
                logger.error("Failed to create WhatsApp channel", error=str(e), exc_info=True)

        # WebSocket channel (always enabled for web UI)
        if channels_config.web.enabled:
            self.websocket_channel = WebSocketChannel()
            self.channels.append(self.websocket_channel)
            logger.info("WebSocket channel created")

        # Register channels with security guardian for approvals
        # AND with orchestrator for bidirectional messaging
        for channel in self.channels:
            self.security_guardian.register_channel(channel.name, channel)
            self.orchestrator.register_channel(channel.name, channel)
            channel.on_message(self._make_channel_handler(channel))
            logger.info("Channel wired to orchestrator", channel=channel.name)

    def _make_channel_handler(self, channel):
        """Create a message handler that routes channel messages through orchestrator."""
        from src.channels.base import Message

        async def handler(message: Message) -> None:
            try:
                logger.info(
                    "Routing channel message to orchestrator",
                    channel=message.channel,
                    user_id=message.user_id,
                    content_preview=message.content[:80] if message.content else "",
                )

                # Acknowledge receipt with a reaction (eyes emoji)
                ack_target = message.metadata.get("channel_id") or message.metadata.get("chat_id") or message.user_id
                try:
                    if message.channel == "slack":
                        await channel.send_reaction(
                            message_id=message.id,
                            reaction="eyes",
                            channel_id=message.metadata.get("channel_id", ""),
                        )
                    else:
                        await channel.send_reaction(
                            message_id=message.id,
                            reaction="👀",
                        )
                except Exception:
                    pass  # Non-critical

                # Show typing indicator while processing
                try:
                    await channel.send_typing_indicator(ack_target)
                except Exception:
                    pass  # Non-critical

                # Use orchestrator.chat() for synchronous response (same as web UI)
                response = await self.orchestrator.chat(
                    channel=message.channel,
                    user_id=message.user_id,
                    content=message.content,
                    metadata=message.metadata,
                )
                # Send response back via the channel
                if response:
                    # Determine where to send the reply:
                    # - Slack: use channel_id from metadata (channel/DM where msg came from)
                    # - WhatsApp: use chat_id from metadata
                    # - Fallback: user_id
                    if message.channel == "slack":
                        reply_to_id = message.metadata.get("channel_id") or message.user_id
                    elif message.channel == "whatsapp":
                        reply_to_id = message.metadata.get("chat_id") or message.user_id
                    else:
                        reply_to_id = message.user_id

                    logger.info(
                        "Sending response back to channel",
                        channel=message.channel,
                        reply_to_id=reply_to_id,
                        response_length=len(response),
                    )
                    # For Slack: don't thread replies in DMs — send flat messages.
                    # Only thread if the original message was already in a thread.
                    slack_reply_to = None
                    slack_thread_id = None
                    if message.channel == "slack":
                        # Only thread if the incoming message was in a thread
                        if message.thread_id:
                            slack_thread_id = message.thread_id
                    else:
                        slack_reply_to = message.id
                        slack_thread_id = message.thread_id

                    send_result = await channel.send_message(
                        user_id=reply_to_id,
                        content=response,
                        reply_to=slack_reply_to,
                        thread_id=slack_thread_id,
                    )
                    if send_result:
                        logger.info(
                            "Response sent successfully",
                            channel=message.channel,
                            message_id=send_result,
                        )
                    else:
                        logger.error(
                            "send_message returned None — response may not have been delivered",
                            channel=message.channel,
                            reply_to_id=reply_to_id,
                        )
                else:
                    logger.warning(
                        "Orchestrator returned empty response",
                        channel=message.channel,
                        user_id=message.user_id,
                    )
            except Exception as e:
                logger.error(
                    "Failed to handle channel message",
                    channel=message.channel,
                    user_id=message.user_id,
                    error=str(e),
                    exc_info=True,
                )
                try:
                    reply_to_id = message.metadata.get("chat_id") or message.metadata.get("channel_id") or message.user_id
                    await channel.send_message(
                        user_id=reply_to_id,
                        content=f"Sorry, I encountered an error: {str(e)}",
                        reply_to=message.id,
                    )
                except Exception as send_err:
                    logger.error("Failed to send error message back to channel", error=str(send_err))

        return handler

    async def _handle_channel_message(
        self,
        channel_name: str,
        user_id: str,
        content: str,
        metadata: Optional[dict] = None
    ) -> str:
        """Handle incoming messages from any channel."""
        try:
            response = await self.orchestrator.process_message(
                channel=channel_name,
                user_id=user_id,
                content=content,
                metadata=metadata or {}
            )
            return response
        except Exception as e:
            logger.error(
                "Error processing message",
                channel=channel_name,
                user_id=user_id,
                error=str(e)
            )
            return f"I apologize, but I encountered an error: {str(e)}"

    async def start(self) -> None:
        """Start all services."""
        self.running = True
        logger.info("Starting Aria services...")

        # Start channels (await each to catch errors immediately)
        channel_tasks = []
        for channel in self.channels:
            try:
                logger.info(f"Starting channel: {channel.name}...")
                await asyncio.wait_for(channel.start(), timeout=30)
                logger.info(
                    f"Channel {channel.name} started successfully (connected={channel.is_connected})"
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"Channel {channel.name} timed out after 30s during startup. "
                    f"Check that the service is reachable."
                )
            except Exception as e:
                logger.error(
                    f"Channel {channel.name} FAILED to start: {e}",
                    exc_info=True,
                )
        # Keep empty task list for the gather at shutdown
        channel_tasks = []

        # Summary of channel status
        connected = [c.name for c in self.channels if c.is_connected]
        failed = [c.name for c in self.channels if not c.is_connected]
        if connected:
            logger.info(f"Connected channels: {', '.join(connected)}")
        if failed:
            logger.warning(f"FAILED channels (will not receive/send messages): {', '.join(failed)}")

        # Start new subsystems
        if self.heartbeat:
            await self.heartbeat.start()
        # Scheduler auto-starts via heartbeat events — no explicit start needed
        if self.config_watcher:
            await self.config_watcher.start()
        if self.plugin_loader:
            await self.plugin_loader.start_all()

        # Write PID file for daemon mode
        from src.core.daemon import write_pid
        write_pid()

        # Start message router
        router_task = asyncio.create_task(self.message_router.start())

        # Start web server
        web_config = self.settings.channels.web
        port = web_config.port
        host = web_config.host

        # Find an available port if the configured one is in use
        port = self._find_available_port(host, port)

        config = Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        self.web_server = Server(config)
        web_task = asyncio.create_task(self.web_server.serve())

        dashboard_url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
        logger.info(f"Aria is now running on {dashboard_url}")
        logger.info("Press Ctrl+C to stop")

        # Open dashboard in browser
        try:
            webbrowser.open(dashboard_url)
            logger.info(f"Opened dashboard in browser: {dashboard_url}")
        except Exception:
            logger.info(f"Open the dashboard at: {dashboard_url}")

        # Wait for shutdown
        await self.shutdown_event.wait()

        # Stop all services
        logger.info("Stopping Aria services...")

        # Stop new subsystems
        if self.plugin_loader:
            await self.plugin_loader.stop_all()
        if self.config_watcher:
            await self.config_watcher.stop()
        # Scheduler has no stop — it's event-driven via heartbeat
        if self.heartbeat:
            await self.heartbeat.stop()
        # Process manager — kill remaining background processes
        if self.process_manager:
            for proc in self.process_manager.list_processes(running_only=True):
                try:
                    await self.process_manager.kill(proc["id"])
                except Exception:
                    pass

        # Remove PID file
        from src.core.daemon import remove_pid
        remove_pid()

        # Stop web server
        if self.web_server:
            self.web_server.should_exit = True

        # Stop channels
        for channel in self.channels:
            await channel.stop()

        # Stop message router
        await self.message_router.stop()

        # Wait for tasks to complete
        await asyncio.gather(*channel_tasks, router_task, web_task, return_exceptions=True)

        logger.info("Aria stopped")

    @staticmethod
    def _ensure_frontend_built() -> None:
        """Build the frontend if dist/ is missing or stale."""
        import subprocess as sp

        frontend_dir = Path(__file__).parent / "web" / "frontend"
        dist_index = frontend_dir / "dist" / "index.html"

        if not frontend_dir.exists():
            return

        # Check if build is needed
        needs_build = not dist_index.exists()
        if not needs_build:
            src_dir = frontend_dir / "src"
            if src_dir.exists():
                newest_src = max(
                    (f.stat().st_mtime for f in src_dir.rglob("*") if f.is_file()),
                    default=0,
                )
                needs_build = newest_src > dist_index.stat().st_mtime

        if not needs_build:
            return

        logger.info("Web UI needs building, running npm install && npm run build...")
        try:
            sp.run(["npm", "--version"], capture_output=True, check=True, timeout=10)
        except (sp.CalledProcessError, FileNotFoundError, sp.TimeoutExpired):
            logger.warning("npm not found — cannot build frontend. Install Node.js to enable the web UI.")
            return

        try:
            result = sp.run(
                ["npm", "install"],
                cwd=str(frontend_dir),
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode != 0:
                logger.warning(f"npm install failed: {result.stderr[:200]}")
                return

            result = sp.run(
                ["npm", "run", "build"],
                cwd=str(frontend_dir),
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode == 0:
                logger.info("Web UI built successfully")
            else:
                logger.warning(f"Frontend build failed: {result.stderr[:200]}")
        except sp.TimeoutExpired:
            logger.warning("Frontend build timed out")
        except Exception as e:
            logger.warning(f"Frontend build error: {e}")

    @staticmethod
    def _find_available_port(host: str, preferred_port: int) -> int:
        """Find an available port, starting with the preferred one."""
        bind_host = "127.0.0.1" if host == "0.0.0.0" else host
        for port in range(preferred_port, preferred_port + 10):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind((bind_host, port))
                    if port != preferred_port:
                        logger.warning(
                            f"Port {preferred_port} is in use, using port {port} instead"
                        )
                    return port
            except OSError:
                continue
        # All ports tried, return the preferred and let uvicorn handle the error
        logger.error(
            f"Ports {preferred_port}-{preferred_port + 9} are all in use"
        )
        return preferred_port

    async def shutdown(self) -> None:
        """Trigger graceful shutdown."""
        logger.info("Shutdown requested")
        self.running = False
        self.shutdown_event.set()

    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up resources...")

        if self.long_term_memory:
            await self.long_term_memory.cleanup()

        if self.audit_logger:
            await self.audit_logger.cleanup()

        if self.sandbox_manager:
            await self.sandbox_manager.cleanup()


async def main():
    """Main entry point."""
    # Set up logging
    setup_logging()

    logger.info("=" * 60)
    logger.info("  Aria - Personal AI Assistant")
    logger.info("=" * 60)

    # Create application
    app = AriaApplication()

    # Set up signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        asyncio.create_task(app.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        # Initialize
        await app.initialize()

        # Start
        await app.start()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("Fatal error", error=str(e), exc_info=True)
        sys.exit(1)
    finally:
        await app.cleanup()


def _find_compose_dir() -> Path:
    """Find the directory containing docker-compose.yaml."""
    project_root = Path(__file__).resolve().parent.parent
    compose_file = project_root / "docker-compose.yaml"
    if not compose_file.exists():
        compose_file = project_root / "docker" / "docker-compose.yaml"
    if not compose_file.exists():
        logger.error("docker-compose.yaml not found")
        sys.exit(1)
    return compose_file.parent


def _docker_compose_cmd(*args: str) -> None:
    """Run a docker compose command and exit."""
    import subprocess as sp

    compose_dir = _find_compose_dir()
    try:
        result = sp.run(
            ["docker", "compose"] + list(args),
            cwd=str(compose_dir),
        )
        sys.exit(result.returncode)
    except FileNotFoundError:
        logger.error("Docker not found. Install Docker first.")
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)


def _start_docker_deployment(force: bool = False) -> None:
    """Start Aria via Docker Compose (for docker deployment mode).

    Uses the root-level docker-compose.yaml in the project directory.
    Falls back to docker/docker-compose.yaml if the root one is absent.

    Args:
        force: Skip the deployment_mode check (used by --docker flag).
    """
    import os
    import subprocess as sp

    # If we're already inside a container, don't try to start Docker again
    if os.environ.get("ARIA_IN_CONTAINER"):
        return

    settings = get_settings()
    if not force:
        if settings.aria.deployment_mode != "docker":
            return

    # Find the project root (where docker-compose.yaml lives)
    project_root = Path(__file__).resolve().parent.parent
    compose_file = project_root / "docker-compose.yaml"
    if not compose_file.exists():
        compose_file = project_root / "docker" / "docker-compose.yaml"
    if not compose_file.exists():
        logger.error(
            "docker-compose.yaml not found in project root or docker/ directory",
            project_root=str(project_root),
        )
        sys.exit(1)

    compose_dir = compose_file.parent

    # Build profile flags based on user config
    # Redis and Ollama are optional profiles — only include if enabled
    profile_args: list[str] = []
    if getattr(settings, "redis", None) and getattr(settings.redis, "enabled", False):
        profile_args += ["--profile", "redis"]
    # Include ollama profile only if the LLM base_url points at the compose service
    llm_url = getattr(settings.llm.local, "base_url", "") or ""
    if "ollama:" in llm_url or "aria-ollama" in llm_url:
        profile_args += ["--profile", "ollama"]

    logger.info(
        "Docker deployment mode — starting containers...",
        compose_file=str(compose_file),
        profiles=profile_args or ["(none)"],
    )

    def _compose_cmd(*args: str) -> list[str]:
        """Build a docker compose command with profile flags."""
        return ["docker", "compose"] + profile_args + list(args)

    try:
        # Check if containers are already running
        result = sp.run(
            _compose_cmd("ps", "--format", "json"),
            capture_output=True, text=True, timeout=10,
            cwd=str(compose_dir),
        )
        if result.returncode == 0 and "aria-main" in result.stdout:
            logger.info("Docker containers already running")
            # Attach to logs so user can see output
            print("\nAria is running in Docker. Attaching to logs (Ctrl+C to detach)...\n")
            try:
                sp.run(
                    _compose_cmd("logs", "-f", "aria"),
                    timeout=None,
                    cwd=str(compose_dir),
                )
            except KeyboardInterrupt:
                print("\nDetached from logs. Containers are still running.")
                print("  Stop with:  docker compose down")
                print("  Logs:       docker compose logs -f aria")
            sys.exit(0)

        # Start containers
        logger.info("Starting Docker containers...")
        result = sp.run(
            _compose_cmd("up", "-d", "--build"),
            capture_output=False, text=True,
            cwd=str(compose_dir),
        )
        if result.returncode != 0:
            logger.error("docker compose up failed")
            sys.exit(1)

        logger.info("Docker containers started successfully")
        print("\nAria is running in Docker. Attaching to logs (Ctrl+C to detach)...\n")
        try:
            sp.run(
                _compose_cmd("logs", "-f", "aria"),
                timeout=None,
                cwd=str(compose_dir),
            )
        except KeyboardInterrupt:
            print("\nDetached from logs. Containers are still running.")
            print("  Stop with:  docker compose down")
            print("  Logs:       docker compose logs -f aria")
        sys.exit(0)

    except FileNotFoundError:
        logger.error("Docker not found. Install Docker or switch to local mode: aria --local")
        sys.exit(1)


def run():
    """Synchronous entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        prog="aria",
        description="Aria - Personal AI Assistant",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run the interactive setup wizard",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip setup wizard (used during restart)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Force local mode (skip Docker even if deployment_mode is docker)",
    )
    parser.add_argument(
        "--docker",
        nargs="?",
        const="start",
        default=None,
        metavar="ACTION",
        help="Docker mode: start (default), stop, restart, logs",
    )
    parser.add_argument(
        "--daemon",
        nargs="?",
        const="status",
        default=None,
        metavar="ACTION",
        help="Daemon mode: install, uninstall, start, stop, status",
    )
    args = parser.parse_args()

    if not args.skip_setup and (args.setup or is_first_run()):
        from src.cli.wizard import run_wizard

        if not run_wizard():
            sys.exit(0)

    # --daemon flag: Daemon/service management
    import os
    if args.daemon:
        from src.core.daemon import handle_daemon_command
        handle_daemon_command(args.daemon)

    # --docker flag: Docker management commands
    if args.docker:
        action = args.docker
        if action == "stop":
            _docker_compose_cmd("down")
        elif action == "restart":
            _docker_compose_cmd("up", "-d", "--build")
        elif action == "logs":
            _docker_compose_cmd("logs", "-f", "aria")
        elif action in ("start", "up"):
            _start_docker_deployment(force=True)
        else:
            print(f"Unknown docker action: {action}")
            print("Usage: python -m src.main --docker [start|stop|restart|logs]")
            sys.exit(1)

    # Check for Docker deployment mode (unless --local flag or inside container)
    if not args.local and not os.environ.get("ARIA_IN_CONTAINER"):
        _start_docker_deployment()

    asyncio.run(main())


if __name__ == "__main__":
    run()
