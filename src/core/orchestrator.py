"""Main orchestrator - the brain of Aria."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator

from ..utils.config import get_settings
from ..utils.logging import get_audit_logger, get_logger
from .context_manager import ContextManager, ConversationContext
from .events import get_event_bus
from .llm_router import LLMMessage, LLMResponse, LLMRouter, Tool
from .message_router import MessageRouter, MessagePriority, QueuedMessage

if TYPE_CHECKING:
    from ..channels.base import BaseChannel
    from ..memory.rag import RAGPipeline
    from ..security.guardian import SecurityGuardian
    from ..skills.registry import SkillRegistry

logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class OrchestratorResponse:
    """Response from the orchestrator."""

    content: str
    tool_results: list[dict[str, Any]] | None = None
    requires_approval: bool = False
    approval_request: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class Orchestrator:
    """
    Main orchestrator that coordinates all Aria components.

    Responsibilities:
    - Process incoming messages from all channels
    - Route to appropriate LLM (local/cloud)
    - Execute skills with security checks
    - Manage conversation context
    - Handle approval workflows
    - Coordinate memory and RAG
    """

    def __init__(
        self,
        llm_router: "LLMRouter | None" = None,
        context_manager: "ContextManager | None" = None,
        skill_registry: "SkillRegistry | None" = None,
        security_guardian: "SecurityGuardian | None" = None,
        rag_pipeline: "RAGPipeline | None" = None,
        audit_logger: Any = None,
    ) -> None:
        self.settings = get_settings()

        # Core components
        self.llm_router = llm_router or LLMRouter()
        self.context_manager = context_manager or ContextManager()
        self.message_router = MessageRouter()

        # Optional components
        self._skill_registry = skill_registry
        self._security_guardian = security_guardian
        self._rag_pipeline = rag_pipeline
        self._audit_logger = audit_logger

        # Channel management
        self._channels: dict[str, "BaseChannel"] = {}

        # Pending approvals
        self._pending_approvals: dict[str, dict[str, Any]] = {}

        # Event bus
        self._event_bus = get_event_bus()

        # Optional: vector memory (set externally)
        self._vector_memory: Any = None

        # User profile, entity extraction, summarizer (Phase 1)
        self._user_profile_manager: Any = None
        self._entity_extractor: Any = None
        self._summarizer: Any = None

        # Sentiment and personality (Phase 5)
        self._sentiment_analyzer: Any = None
        self._personality_adapter: Any = None

        # Running state
        self._running = False

    def set_user_profile_manager(self, pm: Any) -> None:
        self._user_profile_manager = pm

    def set_entity_extractor(self, ee: Any) -> None:
        self._entity_extractor = ee

    async def _extract_and_update_profile(
        self,
        user_id: str,
        user_content: str,
        assistant_content: str = "",
    ) -> None:
        """
        Automatically extract and persist profile updates from every interaction.
        Runs on user message (+ optional assistant response) — no explicit skill call needed.
        Syncs profile to Cognee knowledge graph when enabled.
        """
        if not self.settings.memory.entity_extraction_enabled:
            return
        if not self._entity_extractor or not self._user_profile_manager:
            return
        try:
            # Profile updates (preferred_name, important_people, preferences, etc.)
            updates = self._entity_extractor.extract_profile_updates(user_content)
            if updates:
                self._user_profile_manager.update_profile(user_id, **updates)
                logger.debug("Auto-updated profile from interaction", user_id=user_id, keys=list(updates.keys()))

            # Facts to remember (from conversation context)
            facts = self._entity_extractor.extract_facts_from_conversation(user_content, assistant_content)
            for fact in facts:
                self._user_profile_manager.add_fact(user_id, fact)
                logger.debug("Auto-added fact from interaction", user_id=user_id, fact_preview=fact[:50])

            # Sync profile to Cognee when knowledge graph enabled (Cognee + user profiles work together)
            if updates or facts:
                if (
                    self._rag_pipeline
                    and self.settings.memory.knowledge_graph.enabled
                ):
                    profile = self._user_profile_manager.get_profile(user_id)
                    await self._rag_pipeline.add_user_profile_to_knowledge(user_id, profile)
        except Exception as e:
            logger.debug("Profile extraction failed", user_id=user_id, error=str(e))

    def set_summarizer(self, s: Any) -> None:
        self._summarizer = s

    def set_sentiment_analyzer(self, sa: Any) -> None:
        self._sentiment_analyzer = sa

    def set_personality_adapter(self, pa: Any) -> None:
        self._personality_adapter = pa

    async def initialize(self) -> None:
        """Initialize the orchestrator."""
        logger.info("Orchestrator initialized")

    def set_skill_registry(self, registry: "SkillRegistry") -> None:
        """Set the skill registry."""
        self._skill_registry = registry

    def set_security_guardian(self, guardian: "SecurityGuardian") -> None:
        """Set the security guardian."""
        self._security_guardian = guardian

    def set_rag_pipeline(self, rag: "RAGPipeline") -> None:
        """Set the RAG pipeline."""
        self._rag_pipeline = rag

    def set_vector_memory(self, vm: Any) -> None:
        """Set the vector memory store."""
        self._vector_memory = vm

    def register_channel(self, name: str, channel: "BaseChannel") -> None:
        """Register a messaging channel."""
        self._channels[name] = channel
        logger.info("Registered channel", channel=name)

    async def start(self) -> None:
        """Start the orchestrator and all components."""
        if self._running:
            return

        logger.info("Starting orchestrator")

        # Set up message handler
        self.message_router.set_default_handler(self._handle_message)
        self.message_router.on_message_completed(self._on_message_completed)
        self.message_router.on_message_failed(self._on_message_failed)

        # Start message router
        await self.message_router.start()

        # Check LLM availability
        availability = await self.llm_router.check_availability()
        logger.info("LLM availability", **availability)

        # Start channels
        for name, channel in self._channels.items():
            try:
                await channel.start()
                logger.info("Started channel", channel=name)
            except Exception as e:
                logger.error("Failed to start channel", channel=name, error=str(e))

        self._running = True
        logger.info("Orchestrator started")

    async def stop(self) -> None:
        """Stop the orchestrator and all components."""
        if not self._running:
            return

        logger.info("Stopping orchestrator")

        # Stop channels
        for name, channel in self._channels.items():
            try:
                await channel.stop()
                logger.info("Stopped channel", channel=name)
            except Exception as e:
                logger.error("Failed to stop channel", channel=name, error=str(e))

        # Stop message router
        await self.message_router.stop()

        self._running = False
        logger.info("Orchestrator stopped")

    async def process_message(
        self,
        channel: str,
        user_id: str,
        content: str,
        message_type: str = "text",
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: dict[str, Any] | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Process an incoming message.

        This is the main entry point for messages from channels.

        Args:
            channel: Source channel name
            user_id: User identifier
            content: Message content
            message_type: Type of message
            priority: Processing priority
            metadata: Additional metadata
            attachments: Media attachments

        Returns:
            Message ID for tracking
        """
        message = await self.message_router.enqueue(
            channel=channel,
            user_id=user_id,
            content=content,
            message_type=message_type,
            priority=priority,
            metadata=metadata or {},
            attachments=attachments or [],
        )

        logger.debug(
            "Message queued for processing",
            message_id=message.id,
            channel=channel,
            user_id=user_id,
        )

        return message.id

    async def chat(
        self,
        channel: str,
        user_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Process a chat message synchronously and return the response.

        Unlike process_message which queues, this directly processes
        and waits for the LLM response, including tool execution.

        Args:
            channel: Source channel name
            user_id: User identifier
            content: Message content
            metadata: Additional metadata

        Returns:
            The assistant's response text
        """
        # Log incoming message
        audit_logger.log(
            event="message_received",
            action_type="chat",
            user_id=user_id,
            channel=channel,
            details={"content_preview": content[:100] if content else ""},
        )

        # Get or create conversation context
        context = await self.context_manager.get_context(
            channel=channel,
            user_id=user_id,
        )

        if context is None:
            return "Failed to get conversation context."

        # Add user message to context
        context.add_user_message(content)

        # Handle slash commands (/help, /clear, /status, /skills)
        if content.strip().startswith("/"):
            response_content = await self._handle_slash_command(
                channel=channel,
                user_id=user_id,
                content=content,
                context=context,
            )
            if response_content is not None:
                context.add_assistant_message(response_content)
                await self.context_manager.trim_context(context)
                return response_content

        # Handle "what can you do" type queries with detailed capabilities list
        if self._is_capabilities_query(content):
            response_content = self._get_capabilities_detail()
            context.add_assistant_message(response_content)
            await self._extract_and_update_profile(user_id, content, response_content)
            await self.context_manager.trim_context(context)
            return response_content

        # Emit message_received event
        await self._event_bus.emit("message_received", {
            "channel": channel,
            "user_id": user_id,
            "content_preview": content[:100] if content else "",
        }, source="orchestrator")

        # Auto-extract link content from URLs in the message
        link_context = ""
        try:
            from ..processing.link_extractor import process_message_links
            link_context = await process_message_links(content) or ""
        except Exception as e:
            logger.debug("Link extraction failed", error=str(e))

        # Auto-process media attachments
        media_context = ""
        if metadata and self._skill_registry:
            try:
                from ..processing.media_processor import detect_attachments_in_metadata, process_attachments
                atts = detect_attachments_in_metadata(metadata)
                if atts:
                    media_context = await process_attachments(atts, self._skill_registry) or ""
            except Exception as e:
                logger.debug("Media processing failed", error=str(e))

        # Index message in vector memory for future recall
        if self._vector_memory and self._vector_memory.available:
            try:
                await self._vector_memory.add_message(
                    content=content,
                    role="user",
                    channel=channel,
                    user_id=user_id,
                )
            except Exception:
                pass  # Non-critical

        # Get relevant context from RAG if available
        rag_context = ""
        if self._rag_pipeline:
            try:
                rag_results = await self._rag_pipeline.query(content, top_k=3)
                if rag_results:
                    rag_context = "\n\nRelevant context:\n" + "\n".join(
                        f"- {r['content']}" for r in rag_results
                    )
            except Exception as e:
                logger.warning("RAG query failed", error=str(e))

        # Build messages for LLM — inject any auto-extracted context
        messages = context.get_messages()
        extra_context = rag_context + link_context + media_context
        if extra_context:
            messages[-1] = LLMMessage(
                role="user",
                content=messages[-1].content + extra_context,
            )

        # Inject user profile context (Phase 1)
        if self._user_profile_manager and self.settings.memory.user_profiles_enabled:
            profile_context = self._user_profile_manager.get_context_for_llm(user_id)
            if profile_context and messages and messages[0].role == "system":
                messages[0] = LLMMessage(
                    role="system",
                    content=messages[0].content + profile_context,
                )

        # Sentiment + personality adaptation (Phase 5)
        if self._sentiment_analyzer and self._personality_adapter and messages and messages[0].role == "system":
            sentiment = self._sentiment_analyzer.analyze(content, user_id)
            user_profile = (
                self._user_profile_manager.get_profile(user_id)
                if self._user_profile_manager
                else None
            )
            adapted = self._personality_adapter.adapt_system_prompt(
                messages[0].content, sentiment, channel, user_profile
            )
            messages[0] = LLMMessage(role="system", content=adapted)

        # Search vector memory for relevant past conversations
        if self._vector_memory and self._vector_memory.available:
            try:
                # Fetch extra results to account for filtering out self-match
                memories = await self._vector_memory.search_messages(content, top_k=6)
                if memories:
                    memory_text = "\n\n---\nRelevant past conversations:\n"
                    added = 0
                    for m in memories:
                        # Skip near-exact self-matches (the message we just indexed)
                        if m["score"] > 0.98:
                            continue
                        if m["score"] > 0.6 and added < 3:
                            memory_text += f"- [{m['metadata'].get('role', '?')}] {m['content'][:200]}\n"
                            added += 1
                    if added > 0:
                        messages[-1] = LLMMessage(
                            role="user",
                            content=messages[-1].content + memory_text + "---",
                        )
            except Exception:
                pass

        # Get available tools from skills
        tools = self._get_available_tools()

        # Determine task type for routing
        task_type = self._classify_task(content)

        try:
            # Generate response with tools
            response = await self.llm_router.generate(
                messages=messages,
                tools=tools,
                task_type=task_type,
            )

            # Handle tool calls if the LLM wants to use skills
            max_iterations = 5
            iteration = 0
            while response.tool_calls and iteration < max_iterations:
                iteration += 1
                # Add assistant message with tool calls to context
                context.add_assistant_message(response.content or "", tool_calls=response.tool_calls)

                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("arguments", {})
                    tool_id = tool_call.get("id", "")

                    logger.debug("Executing tool from chat", tool=tool_name, args=tool_args)

                    # Log tool request to audit
                    audit_logger.action_requested(
                        action_type=self._get_action_type_for_tool(tool_name),
                        description=f"Tool: {tool_name}",
                        user_id=user_id,
                        channel=channel,
                    )

                    # Check security if guardian is available
                    if self._security_guardian:
                        action_type = self._get_action_type_for_tool(tool_name)
                        approval_result = await self._security_guardian.check_action(
                            action_type=action_type,
                            details={"tool": tool_name, "arguments": tool_args},
                            user_id=user_id,
                            channel=channel,
                        )
                        if not approval_result.approved:
                            if approval_result.requires_approval:
                                # In interactive chat, the user explicitly asked —
                                # request approval and wait for response
                                logger.info("Requesting approval for tool", tool=tool_name, action=action_type)
                                approval_result = await self._security_guardian.request_approval(
                                    action_type=action_type,
                                    description=approval_result.description or f"Execute {tool_name}",
                                    details={"tool": tool_name, "arguments": tool_args},
                                    user_id=user_id,
                                    channel=channel,
                                )
                            if not approval_result.approved:
                                audit_logger.action_denied(
                                    action_type=action_type,
                                    reason=approval_result.reason,
                                    channel=channel,
                                )
                                context.add_tool_result(
                                    tool_call_id=tool_id,
                                    content=f"Action denied: {approval_result.reason}",
                                )
                                continue

                    # Execute the tool
                    try:
                        result = await self._execute_tool(tool_name, tool_args)
                        result_str = str(result.output if hasattr(result, 'output') else result)
                        context.add_tool_result(tool_call_id=tool_id, content=result_str)
                        audit_logger.action_executed(
                            action_type=self._get_action_type_for_tool(tool_name),
                            result="success",
                            channel=channel,
                        )
                    except Exception as e:
                        logger.error("Tool execution failed in chat", tool=tool_name, error=str(e))
                        context.add_tool_result(tool_call_id=tool_id, content=f"Error: {str(e)}")
                        audit_logger.action_failed(
                            action_type=self._get_action_type_for_tool(tool_name),
                            error=str(e),
                            channel=channel,
                        )

                # Generate follow-up response with tool results
                response = await self.llm_router.generate(
                    messages=context.get_messages(),
                    tools=tools,
                    task_type=task_type,
                )

            # Safety net: if the LLM outputted tool-call JSON as text instead of
            # using structured tool_calls, sanitize the response so users get
            # plain text, not raw JSON.
            final_content = self._sanitize_response(response.content)

            # Add final assistant response to context
            context.add_assistant_message(final_content)

            # Trim context if needed
            await self.context_manager.trim_context(context)

            # Index assistant response in vector memory
            if self._vector_memory and self._vector_memory.available:
                try:
                    await self._vector_memory.add_message(
                        content=final_content,
                        role="assistant",
                        channel=channel,
                        user_id=user_id,
                    )
                except Exception:
                    pass

            # Emit event
            await self._event_bus.emit("message_sent", {
                "channel": channel,
                "user_id": user_id,
                "response_preview": final_content[:100] if final_content else "",
            }, source="orchestrator")

            # Log response
            audit_logger.log(
                event="message_responded",
                action_type="chat",
                user_id=user_id,
                channel=channel,
                details={"response_preview": final_content[:100] if final_content else ""},
            )

            # Auto-extract and update profile from every interaction (no explicit skill needed)
            await self._extract_and_update_profile(user_id, content, final_content)

            return final_content

        except Exception as e:
            logger.error("Chat generation failed", error=str(e))
            audit_logger.action_failed(
                action_type="chat",
                error=str(e),
                channel=channel,
            )
            return f"I encountered an error processing your message: {str(e)}"

    async def _handle_message(self, message: QueuedMessage) -> OrchestratorResponse:
        """
        Handle a queued message.

        This is called by the message router for each message.
        """
        logger.debug(
            "Handling message",
            message_id=message.id,
            channel=message.channel,
        )

        # Get or create conversation context
        context = await self.context_manager.get_context(
            channel=message.channel,
            user_id=message.user_id,
        )

        if context is None:
            raise RuntimeError("Failed to get conversation context")

        # Add user message to context
        context.add_user_message(message.content)

        # Handle special commands
        if message.content.startswith("/"):
            response = await self._handle_command(message, context)
            if response:
                context.add_assistant_message(response.content)
                return response

        # Handle "what can you do" type queries with detailed capabilities list
        if self._is_capabilities_query(message.content):
            response_content = self._get_capabilities_detail()
            context.add_assistant_message(response_content)
            await self._extract_and_update_profile(
                message.user_id,
                message.content,
                response_content,
            )
            await self.context_manager.trim_context(context)
            return OrchestratorResponse(content=response_content)

        # Get relevant context from RAG if available
        rag_context = ""
        if self._rag_pipeline:
            try:
                rag_results = await self._rag_pipeline.query(
                    message.content,
                    top_k=3,
                )
                if rag_results:
                    rag_context = "\n\nRelevant context:\n" + "\n".join(
                        f"- {r['content']}" for r in rag_results
                    )
            except Exception as e:
                logger.warning("RAG query failed", error=str(e))

        # Build messages for LLM
        messages = context.get_messages()
        if rag_context:
            # Inject RAG context into the last user message
            messages[-1] = LLMMessage(
                role="user",
                content=messages[-1].content + rag_context,
            )

        # Get available tools
        tools = self._get_available_tools()

        # Determine task type for routing
        task_type = self._classify_task(message.content)

        # Generate response
        response = await self.llm_router.generate(
            messages=messages,
            tools=tools,
            task_type=task_type,
        )

        # Handle tool calls if any
        if response.tool_calls:
            tool_response = await self._handle_tool_calls(
                response.tool_calls,
                context,
                message,
            )
            if tool_response.requires_approval:
                return tool_response

            # If tools were executed, get a final response
            if tool_response.tool_results:
                # Add tool results to context
                for result in tool_response.tool_results:
                    context.add_tool_result(
                        tool_call_id=result["tool_call_id"],
                        content=str(result["result"]),
                    )

                # Generate final response with tool results
                final_response = await self.llm_router.generate(
                    messages=context.get_messages(),
                    task_type=task_type,
                )
                response = final_response

        # Add assistant response to context
        context.add_assistant_message(response.content)

        # Trim context if needed
        await self.context_manager.trim_context(context)

        # Auto-extract and update profile from every interaction (Slack, WhatsApp, etc.)
        await self._extract_and_update_profile(
            message.user_id,
            message.content,
            response.content,
        )

        return OrchestratorResponse(
            content=response.content,
            metadata={
                "model": response.model,
                "provider": response.provider.value,
                "usage": response.usage,
            },
        )

    async def _handle_slash_command(
        self,
        channel: str,
        user_id: str,
        content: str,
        context: ConversationContext,
    ) -> str | None:
        """Handle slash commands. Returns response text or None if not a recognized command."""
        parts = content.strip().split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command == "/help":
            return self._get_help_text()

        elif command == "/clear":
            await self.context_manager.clear_context(
                channel=channel,
                user_id=user_id,
            )
            return "Conversation cleared. Starting fresh!"

        elif command == "/status":
            return self._get_status()

        elif command == "/skills":
            return self._get_skills_list()

        elif command == "/capabilities":
            return self._get_capabilities_detail()

        # Not a recognized command, let LLM handle it
        return None

    async def _handle_command(
        self,
        message: QueuedMessage,
        context: ConversationContext,
    ) -> OrchestratorResponse | None:
        """Handle slash commands (used by message router for Slack, WhatsApp, etc.)."""
        response = await self._handle_slash_command(
            channel=message.channel,
            user_id=message.user_id,
            content=message.content,
            context=context,
        )
        if response is not None:
            return OrchestratorResponse(content=response)
        return None

    def _is_capabilities_query(self, content: str) -> bool:
        """Check if the user is asking what Aria can do."""
        lower = content.lower().strip()
        patterns = [
            "what can you do",
            "what can you help with",
            "what are your capabilities",
            "what are your skills",
            "what do you do",
            "show me your capabilities",
            "list your capabilities",
            "list your skills",
            "what abilities do you have",
            "what can i ask you",
            "how can you help",
            "what are you capable of",
        ]
        return any(p in lower for p in patterns)

    def _get_capabilities_detail(self) -> str:
        """Build a detailed list of skills and how to trigger them."""
        if not self._skill_registry:
            return "No skills are available right now."

        lines = [
            "**Here's what I can do** — skills and how to trigger them:\n",
        ]

        for skill_info in self._skill_registry.list_skills():
            if not skill_info.get("enabled", True):
                continue

            skill = self._skill_registry.get_skill(skill_info["name"])
            if not skill:
                continue

            caps = skill.get_capabilities()
            if not caps:
                continue

            skill_name = skill.name
            skill_desc = skill_info.get("description", skill.description or "")

            lines.append(f"### {skill_name.replace('_', ' ').title()}")
            lines.append(f"_{skill_desc}_\n")

            for cap in caps:
                cap_name = cap.get("name", "")
                cap_desc = cap.get("description", "")

                # Extract trigger examples from description (e.g. "Use for 'X', 'Y'")
                trigger = ""
                if "use for" in cap_desc.lower():
                    # Extract the part after "Use for"
                    idx = cap_desc.lower().find("use for")
                    trigger = cap_desc[idx:].strip()
                elif "e.g." in cap_desc.lower():
                    idx = cap_desc.lower().find("e.g.")
                    trigger = cap_desc[idx:].strip()
                else:
                    human = cap_name.replace("_", " ").replace("-", " ")
                    trigger = f"Just ask naturally — e.g. \"{human}\" or describe what you need"

                lines.append(f"- **{cap_name}**: {cap_desc}")
                lines.append(f"  → Trigger: {trigger}\n")

            lines.append("")

        lines.append("**Commands:** /help • /clear • /status • /skills")
        lines.append("\nJust ask in plain language — I'll use the right skill automatically.")

        return "\n".join(lines)

    def _get_help_text(self) -> str:
        """Get help text for available commands."""
        return """**Available Commands:**

/help - Show this help message
/clear - Clear conversation history
/status - Show system status
/skills - List available skills
/capabilities - Detailed list of skills and how to trigger them

**Capabilities:**
- File operations (read, write, search)
- Shell command execution
- Web browsing and search
- Document processing (PDF, DOCX)
- Media handling (images, audio, video)
- Calendar management
- Email and messaging

Just ask me what you need help with!"""

    def _get_status(self) -> str:
        """Get system status."""
        llm_status = "Local: ✓" if self.llm_router._local_available else "Local: ✗"
        llm_status += " | Cloud: ✓" if self.llm_router._cloud_available else " | Cloud: ✗"

        channel_status = []
        for name, channel in self._channels.items():
            status = "✓" if channel.is_connected else "✗"
            channel_status.append(f"{name}: {status}")

        context_stats = self.context_manager.get_stats()
        router_stats = self.message_router.get_stats()

        return f"""**System Status**

**LLM:** {llm_status}
**Channels:** {', '.join(channel_status)}
**Active Contexts:** {context_stats['active_contexts']}
**Queue Size:** {router_stats['queue_size']}
**Workers:** {router_stats['workers']}"""

    def _get_skills_list(self) -> str:
        """Get list of available skills."""
        if not self._skill_registry:
            return "No skills registered."

        skills = self._skill_registry.list_skills()
        if not skills:
            return "No skills available."

        lines = ["**Available Skills:**\n"]
        for skill in skills:
            status = "✓" if skill.get("enabled", True) else "✗"
            lines.append(f"- {skill['name']} [{status}]: {skill.get('description', 'No description')}")

        return "\n".join(lines)

    def _get_available_tools(self) -> list[Tool]:
        """Get list of available tools for the LLM."""
        if not self._skill_registry:
            return []

        tools = []
        for skill_info in self._skill_registry.list_skills():
            if not skill_info.get("enabled", True):
                continue

            skill = self._skill_registry.get_skill(skill_info["name"])
            if skill:
                for capability in skill.get_capabilities():
                    tools.append(
                        Tool(
                            name=f"{skill_info['name']}.{capability['name']}",
                            description=capability.get("description", ""),
                            parameters=capability.get("parameters", {}),
                        )
                    )

        return tools

    def _classify_task(self, content: str) -> str | None:
        """Classify the task type for LLM routing."""
        content_lower = content.lower()

        if any(word in content_lower for word in ["code", "implement", "write a function", "debug"]):
            return "code_generation"

        if any(word in content_lower for word in ["analyze", "explain", "compare", "evaluate"]):
            return "complex_reasoning"

        if any(word in content_lower for word in ["plan", "strategy", "design", "architecture"]):
            return "multi_step_planning"

        if any(word in content_lower for word in ["create skill", "new ability", "learn to"]):
            return "skill_creation"

        return None

    async def _handle_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        context: ConversationContext,
        message: QueuedMessage,
    ) -> OrchestratorResponse:
        """Handle tool calls from the LLM."""
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("arguments", {})
            tool_id = tool_call.get("id", "")

            logger.debug(
                "Executing tool",
                tool=tool_name,
                args=tool_args,
            )

            # Check security
            if self._security_guardian:
                action_type = self._get_action_type_for_tool(tool_name)
                approval_result = await self._security_guardian.check_action(
                    action_type=action_type,
                    details={
                        "tool": tool_name,
                        "arguments": tool_args,
                    },
                    user_id=message.user_id,
                    channel=message.channel,
                )

                if approval_result.requires_approval:
                    # Queue for approval
                    self._pending_approvals[tool_id] = {
                        "tool_call": tool_call,
                        "context_id": context.id,
                        "message": message,
                        "approval_request": approval_result,
                    }

                    return OrchestratorResponse(
                        content=f"This action requires approval: {approval_result.description}",
                        requires_approval=True,
                        approval_request={
                            "id": tool_id,
                            "action": tool_name,
                            "description": approval_result.description,
                        },
                    )

                if not approval_result.approved:
                    audit_logger.action_denied(
                        action_type=action_type,
                        reason=approval_result.reason,
                        channel=message.channel,
                    )
                    results.append({
                        "tool_call_id": tool_id,
                        "result": f"Action denied: {approval_result.reason}",
                        "success": False,
                    })
                    continue

            # Execute the tool
            try:
                result = await self._execute_tool(tool_name, tool_args)
                results.append({
                    "tool_call_id": tool_id,
                    "result": result,
                    "success": True,
                })

                audit_logger.action_executed(
                    action_type=self._get_action_type_for_tool(tool_name),
                    result="success",
                    channel=message.channel,
                )

            except Exception as e:
                logger.error("Tool execution failed", tool=tool_name, error=str(e))
                results.append({
                    "tool_call_id": tool_id,
                    "result": f"Error: {str(e)}",
                    "success": False,
                })

                audit_logger.action_failed(
                    action_type=self._get_action_type_for_tool(tool_name),
                    error=str(e),
                    channel=message.channel,
                )

        return OrchestratorResponse(
            content="",
            tool_results=results,
        )

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute a tool by name."""
        if not self._skill_registry:
            raise RuntimeError("Skill registry not available")

        # Parse skill.capability format
        parts = tool_name.split(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid tool name format: {tool_name}")

        skill_name, capability_name = parts

        skill = self._skill_registry.get_skill(skill_name)
        if not skill:
            raise ValueError(f"Skill not found: {skill_name}")

        return await skill.execute(capability_name, **arguments)

    def _sanitize_response(self, content: str) -> str:
        """Sanitize LLM response: strip raw tool-call JSON that leaked into text.

        Some models (especially smaller Ollama models) output tool calls as
        JSON text instead of using the structured tool_calls field.  This
        looks like: {"name": "sms.send", "inputs": {...}} — which is
        meaningless to the end user.  Detect and remove it.
        """
        if not content:
            return content

        stripped = content.strip()
        name = self.settings.aria.name

        def _is_tool_call_json(text: str) -> bool:
            """Check if text is JSON that looks like a tool call."""
            try:
                import json
                parsed = json.loads(text)
                return isinstance(parsed, dict) and "name" in parsed
            except (json.JSONDecodeError, ValueError):
                return False

        fallback = f"Hey — I'm {name}. What can I do for you?"

        # Check if the entire response is a JSON tool call
        if stripped.startswith("{") and stripped.endswith("}") and _is_tool_call_json(stripped):
            logger.warning("LLM returned tool-call JSON as text, discarding", raw=stripped[:120])
            return fallback

        # Check if the response contains multiple JSON blocks (tool call + error)
        import re
        json_blocks = re.findall(r'\{[^{}]*\}', stripped)
        if json_blocks:
            remaining = stripped
            for block in json_blocks:
                remaining = remaining.replace(block, "", 1)
            # If removing all JSON blocks leaves only whitespace, it was all JSON
            if not remaining.strip():
                import json
                try:
                    all_tool = all(
                        isinstance(json.loads(b), dict) and ("name" in json.loads(b) or "code" in json.loads(b))
                        for b in json_blocks
                    )
                    if all_tool:
                        logger.warning("LLM returned multiple JSON blocks as text, discarding", block_count=len(json_blocks))
                        return fallback
                except (json.JSONDecodeError, ValueError):
                    pass
            else:
                # Mixed content: text + JSON. Strip just the JSON parts and keep the text.
                cleaned = stripped
                for block in json_blocks:
                    try:
                        import json
                        parsed = json.loads(block)
                        if isinstance(parsed, dict) and ("name" in parsed or "code" in parsed):
                            cleaned = cleaned.replace(block, "").strip()
                    except (json.JSONDecodeError, ValueError):
                        pass
                if cleaned and cleaned != stripped:
                    logger.warning("Stripped embedded JSON tool calls from response")
                    return cleaned

        return content

    def _get_action_type_for_tool(self, tool_name: str) -> str:
        """Map a tool name to a security action type.

        Tool names are formatted as 'skill.capability', e.g. 'shell.execute'.
        We match on the skill prefix first for accuracy, then fall back to
        substring matching.
        """
        tool_lower = tool_name.lower()
        # Extract skill prefix (e.g. "shell" from "shell.execute")
        skill = tool_lower.split(".")[0] if "." in tool_lower else tool_lower

        # Exact skill-prefix matching (most reliable)
        skill_map = {
            "shell": "shell_commands",
            "filesystem": "read_files",
            "browser": "web_requests",
            "email": "send_emails",
            "sms": "send_messages",
            "calendar": "calendar_read",
            "tts": "web_requests",      # text-to-speech is benign
            "stt": "web_requests",      # speech-to-text is benign
            "image": "read_files",      # image processing is benign
            "video": "read_files",      # video processing is benign
            "documents": "read_files",  # document processing is benign
        }

        if skill in skill_map:
            action = skill_map[skill]
            # Refine: filesystem writes/deletes need higher security
            if skill == "filesystem":
                if "write" in tool_lower or "create" in tool_lower:
                    return "write_files"
                if "delete" in tool_lower:
                    return "delete_files"
            # Refine: calendar writes need higher security
            if skill == "calendar":
                if "create" in tool_lower or "update" in tool_lower or "delete" in tool_lower:
                    return "calendar_write"
            return action

        # Fallback substring matching for unknown tools
        if "shell" in tool_lower or "command" in tool_lower or "execute" in tool_lower:
            return "shell_commands"
        if "email" in tool_lower or "mail" in tool_lower:
            return "send_emails"
        if "sms" in tool_lower:
            return "send_messages"

        # Default to read_files (AUTO) instead of shell_commands (APPROVE)
        # — unknown tools shouldn't block the user with approval requests
        return "read_files"

    async def handle_approval(
        self,
        approval_id: str,
        approved: bool,
        approved_by: str,
        channel: str,
    ) -> OrchestratorResponse | None:
        """
        Handle an approval response.

        Args:
            approval_id: The approval request ID
            approved: Whether the action was approved
            approved_by: Who approved/denied
            channel: Channel where approval came from

        Returns:
            Response to send, or None if approval not found
        """
        if approval_id not in self._pending_approvals:
            return None

        approval_data = self._pending_approvals.pop(approval_id)
        tool_call = approval_data["tool_call"]
        message = approval_data["message"]

        if approved:
            audit_logger.action_approved(
                action_type=self._get_action_type_for_tool(tool_call["name"]),
                approved_by=approved_by,
                channel=channel,
            )

            # Execute the tool
            try:
                result = await self._execute_tool(
                    tool_call["name"],
                    tool_call.get("arguments", {}),
                )

                # Get context and continue conversation
                context = await self.context_manager.get_context(
                    channel=message.channel,
                    user_id=message.user_id,
                )

                if context:
                    context.add_tool_result(
                        tool_call_id=tool_call.get("id", ""),
                        content=str(result),
                    )

                    # Generate follow-up response
                    response = await self.llm_router.generate(
                        messages=context.get_messages(),
                    )

                    context.add_assistant_message(response.content)

                    return OrchestratorResponse(
                        content=response.content,
                        tool_results=[{
                            "tool_call_id": tool_call.get("id", ""),
                            "result": result,
                            "success": True,
                        }],
                    )

            except Exception as e:
                return OrchestratorResponse(
                    content=f"Action failed: {str(e)}",
                )
        else:
            audit_logger.action_denied(
                action_type=self._get_action_type_for_tool(tool_call["name"]),
                denied_by=approved_by,
                channel=channel,
            )

            return OrchestratorResponse(
                content="Action was denied.",
            )

        return None

    async def _on_message_completed(self, message: QueuedMessage) -> None:
        """Callback when a message is completed."""
        if message.result and isinstance(message.result, OrchestratorResponse):
            # Send response to channel
            channel = self._channels.get(message.channel)
            if channel:
                await channel.send_message(
                    user_id=message.user_id,
                    content=message.result.content,
                    reply_to=message.channel_message_id,
                )

    async def _on_message_failed(self, message: QueuedMessage) -> None:
        """Callback when a message fails."""
        channel = self._channels.get(message.channel)
        if channel:
            await channel.send_message(
                user_id=message.user_id,
                content=f"Sorry, I encountered an error: {message.error}",
                reply_to=message.channel_message_id,
            )

    async def stream_response(
        self,
        channel: str,
        user_id: str,
        content: str,
    ) -> AsyncIterator[str]:
        """
        Stream a response for real-time output.

        This bypasses the queue for interactive streaming.
        """
        context = await self.context_manager.get_context(
            channel=channel,
            user_id=user_id,
        )

        if context is None:
            yield "Error: Failed to get conversation context"
            return

        context.add_user_message(content)

        # Handle slash commands
        if content.strip().startswith("/"):
            response_content = await self._handle_slash_command(
                channel=channel,
                user_id=user_id,
                content=content,
                context=context,
            )
            if response_content is not None:
                context.add_assistant_message(response_content)
                await self._extract_and_update_profile(user_id, content, response_content)
                await self.context_manager.trim_context(context)
                yield response_content
                return

        # Handle "what can you do" with full response (no streaming needed)
        if self._is_capabilities_query(content):
            response_content = self._get_capabilities_detail()
            context.add_assistant_message(response_content)
            await self._extract_and_update_profile(user_id, content, response_content)
            await self.context_manager.trim_context(context)
            yield response_content
            return

        messages = context.get_messages()
        task_type = self._classify_task(content)

        full_response = ""
        async for chunk in self.llm_router.generate_stream(
            messages=messages,
            task_type=task_type,
        ):
            full_response += chunk
            yield chunk

        context.add_assistant_message(full_response)

        # Auto-extract and update profile from every interaction
        await self._extract_and_update_profile(user_id, content, full_response)
        await self.context_manager.trim_context(context)
