"""Main orchestrator - the brain of Aria."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator

from ..utils.config import get_settings
from ..utils.logging import get_audit_logger, get_logger
from .context_efficiency import (
    apply_tool_result_cap_to_context_in_place,
    estimate_tokens_messages,
    has_oversized_tool_results,
    is_likely_context_overflow_error,
    proportional_max_tool_result_chars,
    rag_head_tail_trim,
    truncate_tool_result_text,
)
from .context_manager import ContextManager, ConversationContext
from .events import get_event_bus
from .llm_router import LLMMessage, LLMResponse, LLMRouter, Tool
from .message_router import MessageRouter, MessagePriority, QueuedMessage

if TYPE_CHECKING:
    from ..channels.base import BaseChannel
    from ..memory.rag import RAGPipeline
    from ..security.guardian import SecurityGuardian
    from ..skills.generator import SkillGenerator
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

        # Skill generator for auto-learning new capabilities
        self._skill_generator: "SkillGenerator | None" = None

        # Self-healing service (log monitoring, auto-fix)
        self._self_healing_service: Any = None

        # User profile, entity extraction, summarizer (Phase 1)
        self._user_profile_manager: Any = None
        self._entity_extractor: Any = None
        self._summarizer: Any = None

        # Sentiment and personality (Phase 5)
        self._sentiment_analyzer: Any = None
        self._personality_adapter: Any = None

        # Running state
        self._running = False

    def set_skill_generator(self, generator: "SkillGenerator | None") -> None:
        """Set the skill generator for auto-learning new capabilities."""
        self._skill_generator = generator

    def set_self_healing_service(self, service: Any) -> None:
        """Set the self-healing service for log monitoring and auto-fix."""
        self._self_healing_service = service

    def set_user_profile_manager(self, pm: Any) -> None:
        self._user_profile_manager = pm

    def set_entity_extractor(self, ee: Any) -> None:
        self._entity_extractor = ee

    def _get_context_window_tokens(self) -> int:
        """Context window size for proportional truncation and compaction."""
        return self.llm_router.get_context_window() if self.llm_router else 0

    def _get_messages_for_llm(
        self,
        context: ConversationContext,
        prompt_mode: str = "full",
    ) -> list[LLMMessage]:
        """Messages normalized and limited by turns; optionally minimal system prompt."""
        ctx_window = self._get_context_window_tokens()
        count_tokens = self.llm_router.count_tokens if self.llm_router else None
        return self.context_manager.get_messages_for_llm(
            context,
            context_window_tokens=ctx_window,
            count_tokens=count_tokens,
            prompt_mode=prompt_mode,
        )

    def _trim_rag_combined(self, combined: str) -> str:
        """Apply head+tail trim to RAG/injected context using config."""
        max_chars = getattr(self.settings.memory, "rag_max_chars", 2500)
        head_r = getattr(self.settings.memory, "rag_head_ratio", 0.7)
        tail_r = getattr(self.settings.memory, "rag_tail_ratio", 0.2)
        return rag_head_tail_trim(combined, max_chars, head_r, tail_r)

    def _max_tool_result_chars(self) -> int:
        """Proportional or fixed max chars for a single tool result."""
        ctx_window = self._get_context_window_tokens()
        if ctx_window > 0:
            return proportional_max_tool_result_chars(ctx_window)
        return getattr(self.settings.memory.short_term, "max_tool_result_chars", 4000)

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
        Process a message synchronously and return the response content.
        Used by the web API and channel handlers that need a direct reply.
        """
        message = QueuedMessage(
            channel=channel,
            user_id=user_id,
            content=content,
            metadata=metadata or {},
        )
        try:
            response = await self._handle_message(message)
            return response.content if response else ""
        except Exception as e:
            logger.exception("Chat failed", error=str(e))
            audit_logger.action_failed(
                action_type="chat",
                error=str(e),
                channel=channel,
            )
            return f"I encountered an error processing your message: {str(e)}"

    async def chat_edit(
        self,
        channel: str,
        user_id: str,
        message_id: str,
        new_content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Edit a message and re-run from that point. Truncates after the edited message,
        replaces with new content, and regenerates the response.
        """
        context = await self.context_manager.get_context(
            channel=channel, user_id=user_id, create_if_missing=False
        )
        if not context:
            return "Conversation not found."
        idx = context.get_message_index_by_id(message_id)
        if idx is None:
            return "Message not found."
        if idx < len(context.messages) and context.messages[idx].role != "user":
            return "Can only edit user messages."
        context.truncate_after_message_index(idx - 1)
        await self.context_manager.update_context(context)
        return await self.chat(channel=channel, user_id=user_id, content=new_content, metadata=metadata)

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

        # Get relevant context from RAG if available (head+tail trim to avoid token bleed)
        rag_context = ""
        if self._rag_pipeline:
            try:
                rag_results = await self._rag_pipeline.query(content, top_k=3)
                if rag_results:
                    parts = []
                    for r in rag_results:
                        c = (r.get("content") or "")[:1200]
                        if c:
                            parts.append(c)
                    if parts:
                        combined = "\n\n".join(parts)
                        combined = self._trim_rag_combined(combined)
                        rag_context = "\n\nRelevant context:\n" + combined
            except Exception as e:
                logger.warning("RAG query failed", error=str(e))

        # Use minimal system prompt when context is large to save tokens
        prompt_mode = "full"
        if self.llm_router:
            est = estimate_tokens_messages(context.get_messages(), self.llm_router.count_tokens)
            threshold = getattr(self.settings.memory.short_term, "compaction_trigger_tokens", 0) or 100_000
            if threshold and est > threshold:
                prompt_mode = "minimal"
        # Build messages for LLM (normalized, turn-limited) — inject any auto-extracted context
        messages = self._get_messages_for_llm(context, prompt_mode=prompt_mode)
        extra_context = rag_context + link_context + media_context
        if extra_context and messages:
            messages = list(messages)
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

        # Pre-compact if over token trigger
        trigger = getattr(self.settings.memory.short_term, "compaction_trigger_tokens", 0) or 0
        if trigger > 0 and self.llm_router:
            est = estimate_tokens_messages(messages, self.llm_router.count_tokens)
            if est > trigger:
                await self.context_manager.trim_context(context)
                messages = self._get_messages_for_llm(context)
                if extra_context and messages:
                    messages = list(messages)
                    messages[-1] = LLMMessage(role="user", content=messages[-1].content + extra_context)
                # Re-apply profile/sentiment to system if present
                if self._user_profile_manager and self.settings.memory.user_profiles_enabled and messages and messages[0].role == "system":
                    profile_context = self._user_profile_manager.get_context_for_llm(user_id)
                    if profile_context:
                        messages[0] = LLMMessage(role="system", content=messages[0].content + profile_context)
                if self._sentiment_analyzer and self._personality_adapter and messages and messages[0].role == "system":
                    sentiment = self._sentiment_analyzer.analyze(content, user_id)
                    user_profile = (self._user_profile_manager.get_profile(user_id) if self._user_profile_manager else None)
                    adapted = self._personality_adapter.adapt_system_prompt(messages[0].content, sentiment, channel, user_profile)
                    messages[0] = LLMMessage(role="system", content=adapted)

        try:
            # Generate response with tools
            response = await self.llm_router.generate(
                messages=messages,
                tools=tools,
                task_type=task_type,
            )

            # Handle tool calls if the LLM wants to use skills (configurable cap)
            max_iterations = getattr(self.settings.orchestrator, "max_tool_iterations", 15)
            iteration = 0
            ctx_window = self._get_context_window_tokens()
            while response.tool_calls and iteration < max_iterations:
                iteration += 1
                # Add assistant message with tool calls to context
                context.add_assistant_message(response.content or "", tool_calls=response.tool_calls)

                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = dict(tool_call.get("arguments") or {})
                    if "channel" not in tool_args:
                        tool_args["channel"] = channel
                    if "user_id" not in tool_args:
                        tool_args["user_id"] = user_id
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
                        max_chars = self._max_tool_result_chars()
                        result_str = truncate_tool_result_text(result_str, max_chars)
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
                    messages=self._get_messages_for_llm(context),
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
            # Overflow retry: truncate oversized tool results and retry once
            ctx_window = self._get_context_window_tokens()
            if (
                ctx_window > 0
                and is_likely_context_overflow_error(e)
                and has_oversized_tool_results(context.get_messages(), ctx_window)
            ):
                try:
                    apply_tool_result_cap_to_context_in_place(context.messages, ctx_window)
                    messages = self._get_messages_for_llm(context)
                    if extra_context and messages:
                        messages = list(messages)
                        messages[-1] = LLMMessage(role="user", content=messages[-1].content + extra_context)
                    response = await self.llm_router.generate(messages=messages, tools=tools, task_type=task_type)
                    final_content = self._sanitize_response(response.content)
                    context.add_assistant_message(final_content)
                    await self.context_manager.trim_context(context)
                    if self._vector_memory and self._vector_memory.available:
                        try:
                            await self._vector_memory.add_message(content=final_content, role="assistant", channel=channel, user_id=user_id)
                        except Exception:
                            pass
                    await self._event_bus.emit("message_sent", {"channel": channel, "user_id": user_id, "response_preview": (final_content or "")[:100]}, source="orchestrator")
                    audit_logger.log(event="message_responded", action_type="chat", user_id=user_id, channel=channel, details={"response_preview": (final_content or "")[:100]})
                    await self._extract_and_update_profile(user_id, content, final_content)
                    return final_content
                except Exception as retry_e:
                    logger.error("Chat overflow retry failed", error=str(retry_e))
                    e = retry_e
            logger.error("Chat generation failed", error=str(e))
            audit_logger.action_failed(
                action_type="chat",
                error=str(e),
                channel=channel,
            )
            if ctx_window > 0 and is_likely_context_overflow_error(e):
                return (
                    "This conversation is quite long. Try starting a new chat, or ask me to summarize "
                    "and we can continue in a fresh thread."
                )
            return "I ran into a problem processing your message. Please try again or rephrase; if it persists, try a new chat."

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

        # Get relevant context from RAG if available (head+tail trim)
        rag_context = ""
        if self._rag_pipeline:
            try:
                rag_results = await self._rag_pipeline.query(
                    message.content,
                    top_k=3,
                )
                if rag_results:
                    parts: list[str] = []
                    for r in rag_results:
                        c = (r.get("content") or "")[:1200]
                        if c:
                            parts.append(c)
                    if parts:
                        combined = "\n\n".join(parts)
                        combined = self._trim_rag_combined(combined)
                        rag_context = "\n\nRelevant context:\n" + combined
            except Exception as e:
                logger.warning("RAG query failed", error=str(e))

        # Build messages for LLM (normalized, turn-limited)
        messages = self._get_messages_for_llm(context)
        if rag_context and messages:
            messages = list(messages)
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
                max_chars = self._max_tool_result_chars()
                for result in tool_response.tool_results:
                    content = truncate_tool_result_text(str(result["result"]), max_chars)
                    context.add_tool_result(
                        tool_call_id=result["tool_call_id"],
                        content=content,
                    )

                # Generate final response with tool results
                final_response = await self.llm_router.generate(
                    messages=self._get_messages_for_llm(context),
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
        tools: list[Tool] = []

        # Add check_logs_and_heal tool when self-healing is enabled
        if (
            self._self_healing_service
            and self.settings.proactive.self_healing_enabled
        ):
            tools.append(
                Tool(
                    name="check_logs_and_heal",
                    description=(
                        "Check recent logs for errors and automatically fix common issues "
                        "(e.g. missing ChromaDB, Whisper, vite.svg). Use when the user reports "
                        "errors, something is broken, or asks to fix/diagnose issues."
                    ),
                    parameters={"type": "object", "properties": {}},
                )
            )

        # Workspace: let assistant update SOUL/USER/IDENTITY/AGENTS from conversation
        from .workspace import get_workspace_dir
        if get_workspace_dir():
            tools.append(
                Tool(
                    name="update_workspace",
                    description=(
                        "Update a workspace file (SOUL, USER, IDENTITY, AGENTS) with user preferences or facts. "
                        "Use when the user says to remember something for their profile or how the assistant should behave. "
                        "file: soul | identity | user | agents. content: text to set (or append if append=true)."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "description": "Which file: soul, identity, user, or agents"},
                            "content": {"type": "string", "description": "Text to write or append"},
                            "append": {"type": "boolean", "description": "If true, append to existing content; else replace", "default": False},
                        },
                        "required": ["file", "content"],
                    },
                )
            )

        # Add create_skill tool when learning is enabled
        if (
            self._skill_generator
            and self.settings.skills.learned.enabled
        ):
            tools.append(
                Tool(
                    name="create_skill",
                    description=(
                        "Create a new skill when no existing skill fits the user's request. "
                        "Use it when the user asks for something you don't have a direct tool for: "
                        "you can first do the task once via shell/browser/research to learn how, "
                        "then create_skill to make it permanent. The new skill loads immediately; "
                        "you MUST then call it to complete the request (e.g. create_skill → call new_skill.do). "
                        "Never respond with 'I can't' when you have shell and browser — create a skill or do it inline."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Skill name (snake_case, e.g. 'github_issues')",
                            },
                            "description": {
                                "type": "string",
                                "description": "What the skill does and when to use it",
                            },
                            "capabilities": {
                                "type": "array",
                                "description": "List of capabilities this skill provides",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Capability name (snake_case)",
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "What this capability does",
                                        },
                                    },
                                    "required": ["name", "description"],
                                },
                            },
                        },
                        "required": ["name", "description", "capabilities"],
                    },
                )
            )

        if not self._skill_registry:
            return tools

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

        if any(word in content_lower for word in [
            "create skill", "new ability", "learn to", "learn how", "how do i", "can you do",
            "add support for", "integrate", "connect to", "work with", "figure out how",
        ]):
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
            tool_args = dict(tool_call.get("arguments") or {})
            if "channel" not in tool_args:
                tool_args["channel"] = message.channel
            if "user_id" not in tool_args:
                tool_args["user_id"] = message.user_id
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
        # Workspace: update SOUL/USER/IDENTITY/AGENTS from conversation
        if tool_name == "update_workspace":
            return await self._execute_update_workspace(arguments)

        # Handle create_skill (auto-learning) before skill.capability routing
        if tool_name == "create_skill":
            return await self._execute_create_skill(arguments)

        # Handle check_logs_and_heal (self-healing)
        if tool_name == "check_logs_and_heal":
            return await self._execute_check_logs_and_heal()

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

    async def _execute_create_skill(self, arguments: dict[str, Any]) -> Any:
        """Execute the create_skill tool to generate and save a new capability."""
        if not self._skill_generator:
            return {"success": False, "error": "Skill learning is not configured"}

        name = arguments.get("name", "").strip()
        description = arguments.get("description", "").strip()
        capabilities = arguments.get("capabilities", [])

        if not name or not description or not capabilities:
            return {
                "success": False,
                "error": "name, description, and capabilities (each with name and description) are required",
            }

        # Normalize capability format
        caps = []
        for cap in capabilities:
            if isinstance(cap, dict):
                cap_name = cap.get("name", "").strip()
                cap_desc = cap.get("description", "").strip()
                if cap_name and cap_desc:
                    caps.append({"name": cap_name, "description": cap_desc})

        if not caps:
            return {"success": False, "error": "At least one valid capability required"}

        try:
            skill = await self._skill_generator.generate(
                name=name,
                description=description,
                capabilities=caps,
            )

            if skill.error:
                return {"success": False, "error": skill.error}

            path = await self._skill_generator.save(skill)
            if not path:
                return {"success": False, "error": "Failed to save skill (test or approval may have blocked it)"}
            if path.startswith("pending:"):
                pending_id = path.replace("pending:", "", 1)
                return {
                    "success": True,
                    "message": f"Skill '{name}' is pending approval (ID: {pending_id}). Approve it in Settings → Skills → Pending.",
                    "pending_id": pending_id,
                }

            # Reload into registry so it's available immediately
            if self._skill_registry:
                loaded = await self._skill_registry.reload_learned_skill(name)
                if loaded:
                    return {
                        "success": True,
                        "message": f"Created skill '{name}' with {len(caps)} capability(ies). It is now available.",
                        "path": path,
                    }
                else:
                    return {
                        "success": True,
                        "message": f"Created skill '{name}' at {path}. Restart may be needed to load it.",
                        "path": path,
                    }

            return {"success": True, "message": f"Created skill '{name}'", "path": path}

        except Exception as e:
            logger.error("create_skill failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _execute_check_logs_and_heal(self) -> dict[str, Any]:
        """Execute the check_logs_and_heal tool to scan logs and fix issues."""
        if not self._self_healing_service:
            return {"success": False, "error": "Self-healing service not available"}
        try:
            result = await self._self_healing_service.check_and_heal()
            fixed = result.get("fixed", [])
            failed = result.get("failed", [])
            detected = result.get("detected", [])
            if fixed:
                return {
                    "success": True,
                    "message": f"Fixed {len(fixed)} issue(s): " + "; ".join(fixed),
                    "fixed": fixed,
                    "failed": failed,
                    "detected": detected,
                }
            if failed:
                return {
                    "success": False,
                    "message": "Could not fix: " + "; ".join(failed),
                    "failed": failed,
                    "detected": detected,
                }
            if detected:
                return {
                    "success": True,
                    "message": "No new issues to fix (some were already attempted).",
                    "detected": detected,
                }
            return {"success": True, "message": "No errors detected in recent logs."}
        except Exception as e:
            logger.error("check_logs_and_heal failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _execute_update_workspace(self, arguments: dict[str, Any]) -> Any:
        """Update a workspace file (SOUL, USER, IDENTITY, AGENTS)."""
        from .workspace import read_workspace_file, write_workspace_file
        file_key = (arguments.get("file") or "").strip().lower()
        content = (arguments.get("content") or "").strip()
        append = bool(arguments.get("append", False))
        user_id = arguments.get("user_id") or ""
        if not content:
            return {"success": False, "error": "content is required"}
        fmap = {"soul": "SOUL.md", "identity": "IDENTITY.md", "user": "USER.md", "agents": "AGENTS.md"}
        if file_key not in fmap:
            return {"success": False, "error": "file must be one of: soul, identity, user, agents"}
        fname = fmap[file_key]
        if append:
            existing = read_workspace_file(fname, user_id=user_id or None)
            content = (existing or "").rstrip() + "\n\n" + content
        ok = write_workspace_file(fname, content, user_id=user_id or None)
        if not ok:
            return {"success": False, "error": "Failed to write file"}
        return {"success": True, "message": f"Updated {fname}."}

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

        # Special tools
        if tool_lower == "create_skill":
            return "skill_creation"
        if tool_lower == "check_logs_and_heal":
            return "shell_commands"  # runs pip install, file ops

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

            # Execute the tool (inject channel/user_id for skills)
            tool_args = dict(tool_call.get("arguments") or {})
            if "channel" not in tool_args:
                tool_args["channel"] = message.channel
            if "user_id" not in tool_args:
                tool_args["user_id"] = message.user_id
            try:
                result = await self._execute_tool(
                    tool_call["name"],
                    tool_args,
                )

                # Get context and continue conversation
                context = await self.context_manager.get_context(
                    channel=message.channel,
                    user_id=message.user_id,
                )

                if context:
                    content = truncate_tool_result_text(str(result), self._max_tool_result_chars())
                    context.add_tool_result(
                        tool_call_id=tool_call.get("id", ""),
                        content=content,
                    )

                    # Generate follow-up response
                    response = await self.llm_router.generate(
                        messages=self._get_messages_for_llm(context),
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

        messages = self._get_messages_for_llm(context)
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
