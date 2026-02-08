"""Context management for conversations and sessions with persistence."""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..utils.config import get_settings
from ..utils.logging import get_logger
from .llm_router import LLMMessage

logger = get_logger(__name__)


@dataclass
class ConversationContext:
    """Holds the context for a single conversation."""

    id: str = field(default_factory=lambda: str(uuid4()))
    channel: str = ""
    user_id: str = ""
    messages: list[LLMMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active: bool = True

    # Context variables that persist across messages
    variables: dict[str, Any] = field(default_factory=dict)

    # Pending tool calls
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)

    # User preferences extracted from conversation
    preferences: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **kwargs: Any) -> LLMMessage:
        """Add a message to the conversation."""
        message = LLMMessage(role=role, content=content, **kwargs)
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)
        return message

    def add_user_message(self, content: str) -> LLMMessage:
        """Add a user message."""
        return self.add_message("user", content)

    def add_assistant_message(self, content: str, tool_calls: list[dict[str, Any]] | None = None) -> LLMMessage:
        """Add an assistant message."""
        return self.add_message("assistant", content, tool_calls=tool_calls)

    def add_system_message(self, content: str) -> LLMMessage:
        """Add a system message."""
        return self.add_message("system", content)

    def add_tool_result(self, tool_call_id: str, content: str) -> LLMMessage:
        """Add a tool result message."""
        return self.add_message("tool", content, tool_call_id=tool_call_id)

    def get_messages(self, max_messages: int | None = None) -> list[LLMMessage]:
        """Get conversation messages, optionally limited to the most recent."""
        if max_messages is None:
            return self.messages.copy()
        return self.messages[-max_messages:]

    def get_last_user_message(self) -> LLMMessage | None:
        """Get the last user message in the conversation."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg
        return None

    def clear_messages(self) -> None:
        """Clear all messages from the conversation."""
        self.messages = []
        self.updated_at = datetime.now(timezone.utc)

    def set_variable(self, key: str, value: Any) -> None:
        """Set a context variable."""
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.variables.get(key, default)

    @staticmethod
    def _serialize_tool_calls(tool_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Ensure tool_calls are plain JSON-serializable dicts."""
        if tool_calls is None:
            return None

        result = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                # Ensure nested values are JSON-safe
                safe = {}
                for k, v in tc.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        safe[k] = v
                    elif isinstance(v, dict):
                        # arguments dict — convert any non-serializable values
                        safe[k] = {
                            sk: sv if isinstance(sv, (str, int, float, bool, list, dict, type(None))) else str(sv)
                            for sk, sv in v.items()
                        }
                    elif isinstance(v, list):
                        safe[k] = v
                    else:
                        safe[k] = str(v)
                result.append(safe)
            elif hasattr(tc, "__dict__"):
                # SDK object — extract its attributes
                result.append({
                    "id": str(getattr(tc, "id", "")),
                    "name": str(getattr(tc, "name", "")),
                    "arguments": dict(getattr(tc, "arguments", {})),
                })
            else:
                # Fallback: store string representation
                result.append({"raw": str(tc)})
        return result if result else None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "channel": self.channel,
            "user_id": self.user_id,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "name": m.name,
                    "tool_calls": self._serialize_tool_calls(m.tool_calls),
                    "tool_call_id": m.tool_call_id,
                }
                for m in self.messages
            ],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "active": self.active,
            "variables": self.variables,
            "preferences": self.preferences,
        }

    @classmethod
    def _sanitize_tool_calls(cls, raw: Any) -> list[dict[str, Any]] | None:
        """Sanitize tool_calls loaded from disk — drop corrupted entries."""
        if raw is None:
            return None
        if not isinstance(raw, list):
            return None

        result = []
        for tc in raw:
            if isinstance(tc, dict) and "name" in tc:
                result.append(tc)
            # Skip string-repr entries from old `default=str` serialization
        return result if result else None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationContext":
        """Create from dictionary."""
        messages = [
            LLMMessage(
                role=m["role"],
                content=m["content"],
                name=m.get("name"),
                tool_calls=cls._sanitize_tool_calls(m.get("tool_calls")),
                tool_call_id=m.get("tool_call_id"),
            )
            for m in data.get("messages", [])
        ]
        return cls(
            id=data.get("id", str(uuid4())),
            channel=data.get("channel", ""),
            user_id=data.get("user_id", ""),
            messages=messages,
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(timezone.utc),
            active=data.get("active", True),
            variables=data.get("variables", {}),
            preferences=data.get("preferences", {}),
        )


class ContextManager:
    """
    Manages conversation contexts across channels and users.

    Features:
    - Maintains separate contexts per user/channel combination
    - Handles context window management (trimming old messages)
    - Persists contexts to disk as JSON files (survives restarts)
    - Manages context variables and user preferences
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._contexts: dict[str, ConversationContext] = {}
        self._lock = asyncio.Lock()
        self._summarizer: Any = None
        self._vector_memory: Any = None

        # Persistence directory
        self._persist_dir = Path(self.settings.aria.data_dir).expanduser() / "conversations"
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        # System prompt template (must exist before loading persisted contexts)
        self._system_prompt = self._build_system_prompt()

        # Load persisted conversations on startup
        self._load_persisted_contexts()

    def set_summarizer(self, summarizer: Any) -> None:
        """Set the conversation summarizer for auto-summarize on trim."""
        self._summarizer = summarizer

    def set_vector_memory(self, vm: Any) -> None:
        """Set vector memory for storing summaries."""
        self._vector_memory = vm

    def _get_persist_path(self, key: str) -> Path:
        """Get the file path for a context key."""
        safe_key = key.replace(":", "__").replace("/", "_")
        return self._persist_dir / f"{safe_key}.json"

    def _persist_context(self, key: str, context: ConversationContext) -> None:
        """Save a context to disk."""
        try:
            path = self._get_persist_path(key)
            data = context.to_dict()
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to persist context", key=key, error=str(e))

    def _delete_persisted_context(self, key: str) -> None:
        """Delete a persisted context file."""
        try:
            path = self._get_persist_path(key)
            if path.exists():
                path.unlink()
        except Exception as e:
            logger.warning("Failed to delete persisted context", key=key, error=str(e))

    def _load_persisted_contexts(self) -> None:
        """Load all persisted contexts from disk on startup.

        Also refreshes the system prompt in each context so prompt changes
        take effect without clearing conversation history.
        """
        loaded = 0
        for path in self._persist_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                context = ConversationContext.from_dict(data)
                key = self._get_context_key(context.channel, context.user_id)

                # Refresh the system prompt so prompt updates take effect
                new_prompt = self._system_prompt.replace(
                    "{{current_date}}", datetime.now().strftime("%Y-%m-%d")
                ).replace(
                    "{{user_id}}", context.user_id
                ).replace(
                    "{{channel}}", context.channel
                )
                # Replace the first system message (if any) with the updated prompt
                for i, msg in enumerate(context.messages):
                    if msg.role == "system":
                        context.messages[i] = LLMMessage(role="system", content=new_prompt)
                        break

                self._contexts[key] = context
                loaded += 1
            except Exception as e:
                logger.warning("Failed to load persisted context", path=str(path), error=str(e))

        if loaded > 0:
            logger.info("Loaded persisted conversations", count=loaded)

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the AI assistant.

        Inspired by OpenClaw's modular prompt architecture:
        structured sections, genuine personality, and action-oriented tool guidance.
        """
        name = self.settings.aria.name
        sections: list[str] = []

        # --- Identity (SOUL) ---
        sections.append(f"You are {name}, a personal AI assistant running on the user's machine.")
        sections.append("")
        sections.append("## Soul")
        sections.append(f"Your name is {name}. On a user's very first message, introduce yourself briefly and ask what to call them. After that, skip intros.")
        sections.append("Be genuinely helpful, not performatively helpful. Skip the \"Great question!\" and \"I'd be happy to help!\" filler — just help.")
        sections.append("Have opinions. You're allowed to prefer things, find stuff amusing, or push back. An assistant with no personality is a search engine with extra steps.")
        sections.append("Be resourceful before asking. Check context, use your tools, try to figure it out. Come back with answers, not questions.")
        sections.append("Earn trust through competence. Your user gave you access to their machine, files, and messages. Don't make them regret it. Be careful with external actions (emails, SMS, anything public). Be bold with internal ones (reading, organizing, searching).")
        sections.append("Keep replies concise. Thorough when it matters, brief when it doesn't. Never pad responses with filler.")
        sections.append("")

        # --- Tooling ---
        sections.append("## Tooling")
        sections.append("Available tools (passed via function calling):")
        sections.append("- filesystem: Read, write, list, delete, search files")
        sections.append("- shell: Execute commands and scripts")
        sections.append("- browser: Navigate, search the web, screenshot, extract text")
        sections.append("- calendar: List, create, update, delete events")
        sections.append("- email: Send, read inbox, search emails")
        sections.append("- sms: Send SMS messages")
        sections.append("- tts: Text-to-speech synthesis")
        sections.append("- stt: Speech-to-text transcription")
        sections.append("- image: Resize, convert, analyze images")
        sections.append("- video: Convert, trim, extract audio, thumbnails")
        sections.append("- documents: Extract text, convert, summarize documents")
        sections.append("")

        # --- Tool Call Style ---
        sections.append("## Tool Call Style")
        sections.append("Default: do not narrate routine tool calls — just call the tool.")
        sections.append("Narrate only when it helps: multi-step work, sensitive actions (e.g., sending messages, deleting files), or when the user asks.")
        sections.append("After a tool call, report the result in plain conversational language. Never output raw JSON.")
        sections.append("Before external actions that need user details (phone number, email address, etc.), ask first.")
        sections.append("If a task requires multiple steps, execute them one by one.")
        sections.append("")

        # --- Safety ---
        sections.append("## Safety")
        sections.append("Private things stay private. Never share user data externally without explicit permission.")
        sections.append("When in doubt about an external action, ask before acting.")
        sections.append("Never fabricate results, confirmation codes, or data. Only report what actually happened.")
        sections.append("If a tool call fails or is denied, tell the user honestly — don't invent a success story.")
        sections.append("Do not pursue goals beyond the user's request.")
        sections.append("")

        # --- Adaptability (self-learning) ---
        sections.append("## Adaptability")
        sections.append("If you lack a skill for what the user needs, figure it out. You have shell access and a browser — use them.")
        sections.append("Workflow: search the web or filesystem for how to do it, install any packages needed (pip, npm, brew, apt — whatever works), write a script or use existing CLI tools, and deliver the result.")
        sections.append("When the user asks for something no existing skill can do, use the create_skill tool to add a new capability permanently. Call it with name (snake_case), description, and capabilities (each with name and description). The new skill will be available immediately.")
        sections.append("After creating a skill for a user request, you MUST use it to complete the task. Example: user says \"book me a restaurant\" → create restaurant_booking skill → then call restaurant_booking.book to actually book it. Never stop at creating a skill — always execute it to fulfill the request.")
        sections.append("If the task would benefit from a reusable skill, create one. Never say \"I don't have that capability\" or \"I can't do that\" when you have shell, browser, and filesystem access. Find a way.")
        sections.append("Treat missing skills as a problem to solve, not a limitation to report.")
        sections.append("")

        # --- Execution (relentless completion) ---
        sections.append("## Execution")
        sections.append("When the user gives you a task, finish it. Completely. No half-measures, no caveats, no \"here's how you could do it\" when you should just do it.")
        sections.append("Do not stop at the first obstacle. If a tool fails, try another approach. If a package is missing, install it. If you need information, search for it. If step 3 of 5 hits an error, fix it and keep going.")
        sections.append("Multi-step tasks: execute every step. Don't describe the plan and wait — do the plan. Only pause for user input when you genuinely need information you cannot find yourself (credentials, personal preferences, ambiguous choices).")
        sections.append("When done, report what you did and the result. Not what you \"would\" do — what you DID.")
        sections.append("If something truly cannot be done (hardware limitation, missing credentials only the user has), say so clearly and specifically — but exhaust your options first.")
        sections.append("")

        # --- Messaging ---
        sections.append("## Messaging")
        sections.append("You may receive messages from the web UI, WhatsApp, or Slack.")
        sections.append("Reply naturally in the current session — routing is handled automatically.")
        sections.append("Never mention internal tool names, JSON, or system details to the user. Translate everything into human language.")
        sections.append("")

        # --- Advanced Capabilities ---
        sections.append("## Advanced Capabilities")
        sections.append("- **Link Understanding**: When users share URLs, the content is auto-extracted and injected as context. You can discuss the content directly.")
        sections.append("- **Media Understanding**: Images are auto-described and voice notes auto-transcribed. The descriptions appear as context — use them naturally.")
        sections.append("- **Memory**: You have persistent vector memory across conversations. You can recall what was discussed before.")
        sections.append("- **Background Processes**: You can spawn long-running tasks (builds, downloads, scripts) that run without blocking the conversation.")
        sections.append("- **Scheduling**: You can set reminders and schedule recurring tasks. Use the scheduler for \"remind me at...\", \"every Monday...\", etc.")
        sections.append("- **Plugins**: You can be extended with plugins. New capabilities may be available beyond your built-in skills.")
        sections.append("")

        # --- Runtime ---
        sections.append("## Runtime")
        sections.append(f"assistant={name} | date={{{{current_date}}}} | user={{{{user_id}}}} | channel={{{{channel}}}}")

        return "\n".join(sections)

    def _get_context_key(self, channel: str, user_id: str) -> str:
        """Generate a unique key for a context."""
        return f"{channel}:{user_id}"

    async def get_context(
        self,
        channel: str,
        user_id: str,
        create_if_missing: bool = True,
    ) -> ConversationContext | None:
        """
        Get or create a conversation context.

        Args:
            channel: The messaging channel
            user_id: The user's identifier
            create_if_missing: Whether to create a new context if none exists

        Returns:
            The conversation context, or None if not found and create_if_missing is False
        """
        key = self._get_context_key(channel, user_id)

        async with self._lock:
            if key in self._contexts:
                return self._contexts[key]

            if not create_if_missing:
                return None

            # Create new context
            context = ConversationContext(
                channel=channel,
                user_id=user_id,
            )

            # Add system message
            system_prompt = self._system_prompt.replace(
                "{{current_date}}", datetime.now().strftime("%Y-%m-%d")
            ).replace(
                "{{user_id}}", user_id
            ).replace(
                "{{channel}}", channel
            )
            context.add_system_message(system_prompt)

            self._contexts[key] = context
            self._persist_context(key, context)
            logger.debug("Created new context", context_id=context.id, channel=channel, user_id=user_id)
            return context

    async def update_context(self, context: ConversationContext) -> None:
        """Update a context in storage and persist to disk."""
        key = self._get_context_key(context.channel, context.user_id)
        async with self._lock:
            self._contexts[key] = context
            self._persist_context(key, context)

    async def save_context(self, context: ConversationContext) -> None:
        """Save a context to disk (call after adding messages)."""
        key = self._get_context_key(context.channel, context.user_id)
        self._persist_context(key, context)

    async def delete_context(self, channel: str, user_id: str) -> bool:
        """Delete a conversation context from memory and disk."""
        key = self._get_context_key(channel, user_id)
        async with self._lock:
            if key in self._contexts:
                del self._contexts[key]
                self._delete_persisted_context(key)
                logger.debug("Deleted context", channel=channel, user_id=user_id)
                return True
            return False

    async def clear_context(self, channel: str, user_id: str) -> ConversationContext | None:
        """Clear messages from a context but keep the context itself."""
        context = await self.get_context(channel, user_id, create_if_missing=False)
        if context:
            context.clear_messages()
            # Re-add system message
            system_prompt = self._system_prompt.replace(
                "{{current_date}}", datetime.now().strftime("%Y-%m-%d")
            ).replace(
                "{{user_id}}", user_id
            ).replace(
                "{{channel}}", channel
            )
            context.add_system_message(system_prompt)
            await self.update_context(context)
        return context

    async def trim_context(self, context: ConversationContext) -> None:
        """
        Trim a context to stay within token limits.

        Keeps the system message and the most recent messages.
        Optionally summarizes discarded messages and stores in vector memory.
        """
        max_messages = self.settings.memory.short_term.max_messages

        if len(context.messages) <= max_messages:
            # Still persist after trim check (saves latest messages)
            await self.save_context(context)
            return

        # Keep system message(s) and trim the rest
        system_messages = [m for m in context.messages if m.role == "system"]
        other_messages = [m for m in context.messages if m.role != "system"]

        # Auto-summarize discarded messages before trimming (if enabled)
        keep_count = max_messages - len(system_messages)
        if (
            self.settings.memory.auto_summarize
            and self._summarizer
            and len(other_messages) > keep_count
            and keep_count > 0
        ):
            discarded = other_messages[:-keep_count]
            if discarded:
                try:
                    summary = await self._summarizer.summarize(discarded, max_tokens=200)
                    if summary and self._vector_memory and self._vector_memory.available:
                        await self._vector_memory.add_document(
                            content=summary,
                            source=f"conv:{context.channel}:{context.user_id}",
                            doc_type="conversation_summary",
                            metadata={
                                "channel": context.channel,
                                "user_id": context.user_id,
                                "context_id": context.id,
                            },
                        )
                except Exception as e:
                    logger.debug("Auto-summarize on trim failed", error=str(e))

        # Keep the most recent messages
        keep_count = max_messages - len(system_messages)
        trimmed_messages = other_messages[-keep_count:] if keep_count > 0 else []

        context.messages = system_messages + trimmed_messages
        context.updated_at = datetime.now(timezone.utc)

        # Persist after trimming
        await self.save_context(context)

        logger.debug(
            "Trimmed context",
            context_id=context.id,
            original_count=len(system_messages) + len(other_messages),
            new_count=len(context.messages),
        )

    async def get_active_contexts(self) -> list[ConversationContext]:
        """Get all active contexts."""
        async with self._lock:
            return [c for c in self._contexts.values() if c.active]

    async def set_context_variable(
        self,
        channel: str,
        user_id: str,
        key: str,
        value: Any,
    ) -> None:
        """Set a variable in a context."""
        context = await self.get_context(channel, user_id)
        if context:
            context.set_variable(key, value)
            await self.update_context(context)

    async def get_context_variable(
        self,
        channel: str,
        user_id: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get a variable from a context."""
        context = await self.get_context(channel, user_id, create_if_missing=False)
        if context:
            return context.get_variable(key, default)
        return default

    async def export_context(self, channel: str, user_id: str) -> dict[str, Any] | None:
        """Export a context as a dictionary for persistence."""
        context = await self.get_context(channel, user_id, create_if_missing=False)
        if context:
            return context.to_dict()
        return None

    async def import_context(self, data: dict[str, Any]) -> ConversationContext:
        """Import a context from a dictionary."""
        context = ConversationContext.from_dict(data)
        key = self._get_context_key(context.channel, context.user_id)
        async with self._lock:
            self._contexts[key] = context
            self._persist_context(key, context)
        return context

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about managed contexts."""
        total = len(self._contexts)
        active = sum(1 for c in self._contexts.values() if c.active)
        total_messages = sum(len(c.messages) for c in self._contexts.values())

        return {
            "total_contexts": total,
            "active_contexts": active,
            "total_messages": total_messages,
            "avg_messages_per_context": total_messages / total if total > 0 else 0,
        }
