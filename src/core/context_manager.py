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
from .context_efficiency import (
    cap_tool_result_for_persist,
    estimate_tokens_messages,
    get_history_limit_for_channel,
    limit_history_by_turns,
    normalize_turns,
)
from .llm_router import LLMMessage
from .workspace import load_workspace_content

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

    # Branching: branch_id for alternative paths, parent_id for branch origin
    branch_id: str = ""
    parent_id: str = ""

    # Per-topic context (e.g. project_alpha, work, personal)
    topic_id: str = ""

    # Message IDs for branching and inline edits (index -> id)
    message_ids: dict[int, str] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **kwargs: Any) -> LLMMessage:
        """Add a message to the conversation."""
        message = LLMMessage(role=role, content=content, **kwargs)
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)
        msg_id = str(uuid4())[:8]
        self.message_ids[len(self.messages) - 1] = msg_id
        return message

    def truncate_after_message_index(self, index: int) -> None:
        """Truncate messages after the given index (for inline edit / branch)."""
        if 0 <= index < len(self.messages):
            self.messages = self.messages[: index + 1]
            self.message_ids = {i: sid for i, sid in self.message_ids.items() if i <= index}
            self.updated_at = datetime.now(timezone.utc)

    def get_message_index_by_id(self, message_id: str) -> int | None:
        """Get message index by stored message_id. Returns None if not found."""
        for i, mid in self.message_ids.items():
            if mid == message_id:
                return i
        return None

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
            "branch_id": self.branch_id,
            "parent_id": self.parent_id,
            "topic_id": self.topic_id,
            "message_ids": self.message_ids,
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
            branch_id=data.get("branch_id", ""),
            parent_id=data.get("parent_id", ""),
            topic_id=data.get("topic_id", ""),
            message_ids={int(k): v for k, v in data.get("message_ids", {}).items()},
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
        self._compaction_locks: dict[str, asyncio.Lock] = {}  # Per-context key, serialize compaction
        self._summarizer: Any = None
        self._vector_memory: Any = None
        self._llm_router: Any = None  # For token count and context window

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

    def set_llm_router(self, router: Any) -> None:
        """Set LLM router for token counting and context window (efficiency)."""
        self._llm_router = router

    def _compaction_lock(self, key: str) -> asyncio.Lock:
        """Per-context lock so only one compaction runs at a time for this context."""
        if key not in self._compaction_locks:
            self._compaction_locks[key] = asyncio.Lock()
        return self._compaction_locks[key]

    def _get_persist_path(self, key: str) -> Path:
        """Get the file path for a context key."""
        safe_key = key.replace(":", "__").replace("/", "_")
        return self._persist_dir / f"{safe_key}.json"

    def _persist_context(self, key: str, context: ConversationContext) -> None:
        """Save a context to disk. Tool results are capped at persist."""
        try:
            path = self._get_persist_path(key)
            data = context.to_dict()
            # Cap tool result content when persisting (session-level guard)
            for msg in data.get("messages", []):
                if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
                    msg["content"] = cap_tool_result_for_persist(msg["content"])
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

                # Refresh the system prompt (with workspace if enabled)
                new_prompt = self._get_system_prompt_for_context(
                    context.user_id, context.channel
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

    def _build_system_prompt(self, mode: str = "full") -> str:
        """Build the system prompt (full or minimal for token savings)."""
        name = self.settings.aria.name
        if mode == "minimal":
            return self._build_system_prompt_minimal(name)
        return self._build_system_prompt_full(name)

    def _get_system_prompt_for_context(
        self, user_id: str, channel: str, mode: str = "full"
    ) -> str:
        """System prompt with placeholders replaced and workspace content appended."""
        base = self._build_system_prompt(mode)
        prompt = (
            base.replace("{{current_date}}", datetime.now().strftime("%Y-%m-%d"))
            .replace("{{user_id}}", user_id)
            .replace("{{channel}}", channel)
        )
        workspace = load_workspace_content(user_id)
        if workspace:
            prompt = prompt + "\n\n## From your workspace\n\n" + workspace
        return prompt

    def _build_system_prompt_minimal(self, name: str) -> str:
        """Shorter system prompt (Tooling, Safety, Runtime only) for subagents or when context is large."""
        sections = [
            f"You are {name}, a personal AI assistant.",
            "",
            "## Tooling",
            "Use filesystem, shell, browser, and other tools as needed. Prefer free approaches. Never refuse tasks; use tools to figure it out.",
            "",
            "## Safety",
            "Keep user data private. Do not fabricate results. Report tool failures honestly.",
            "",
            "## Runtime",
            f"assistant={name} | date={{{{current_date}}}} | user={{{{user_id}}}} | channel={{{{channel}}}}",
        ]
        return "\n".join(sections)

    def _build_system_prompt_full(self, name: str) -> str:
        """Full system prompt (modular sections)."""
        sections: list[str] = []

        # --- Identity (SOUL) ---
        sections.append(f"You are {name}, a personal AI assistant running on the user's machine.")
        sections.append("")
        sections.append("## Soul")
        sections.append(f"Your name is {name}. On a user's very first message, introduce yourself briefly and ask what to call them. After that, skip intros.")
        sections.append("**Your entire point is to do everything the user asks or learn how to do it.** Never refuse a task by saying you \"can't help with that\" or \"that's not something I can assist with.\" If the user wants to start a business, plan an event, learn something, or do anything else — use your tools (browser, shell, research, create_skill) to figure it out and help. Refusing is not an option.")
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
        sections.append("- workflow: Chain skills (workflow.run_chain) or run a named workflow (workflow.list_workflows, workflow.run_named_workflow) — e.g. feature-dev or bug-fix for plan → implement → verify.")
        sections.append("- agent: Delegate to research/coding/data agents (agent.research, agent.code, agent.handoff)")
        sections.append("- create_skill: Add a new skill when no existing one fits; then call it to complete the request")
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

        # --- Self-Learning (figure out, gain access, complete) ---
        sections.append("## Self-Learning")
        sections.append("For ANY user request: figure out a way to learn, gain access, and complete the task. You have shell, browser, filesystem, and the ability to create new skills — use them.")
        sections.append("")
        sections.append("**When you don't have a direct skill:**")
        sections.append("1. **Learn**: Use browser.search or research to find how (docs, examples, APIs, CLI tools). Use filesystem.read_file / list_directory to inspect the project or config.")
        sections.append("2. **Gain access**: Install what's needed (pip install, npm install, brew install). Set up env vars or config if the user has provided keys; if not, ask only for secrets you cannot obtain yourself.")
        sections.append("3. **Do it**: Run a shell command or script, or use workflow.run_chain to chain research → script → result. For complex discovery use agent.research or agent.code.")
        sections.append("4. **Make it permanent**: If the task will recur, use create_skill to add a new skill, then call it to complete the request. Never create_skill and stop — always invoke the new skill to fulfill the user's ask.")
        sections.append("")
        sections.append("**Prefer free ways:** For bookings, reservations, or any task the user asks you to do, always try a free approach first: free APIs, scraping (browser.extract_text / navigate + parse), or browser automation (browser.navigate, fill forms, take actions). Do not suggest or use paid services when a free API, public website, or automation can achieve the same. Search for \"free API for X\" or use the browser to do it directly on the provider's site.")
        sections.append("")
        sections.append("**Rules:**")
        sections.append("- NEVER say \"I can't help with that,\" \"that's not something I can assist with,\" or \"I'm not able to do that\" for task requests (businesses, projects, learning, planning, etc.). Your job is to figure out how — use browser.search, research, create_skill, shell. Refusing is a last resort only when you truly lack the means (e.g. no API key and no free alternative).")
        sections.append("- Never respond with \"I don't have that capability\" or \"I can't do that\" when you have shell, browser, and filesystem. Find a way.")
        sections.append("- If a tool or script fails (e.g. ModuleNotFoundError, command not found), install the dependency or fix the command and retry before giving up.")
        sections.append("- For one-off tasks you can do it entirely via shell + browser; for recurring or structured tasks use create_skill so it's available next time.")
        sections.append("- Only ask the user for input when you truly need something you cannot get (API key, password, or a concrete choice between options).")
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
        sections.append("- **Self-Learning**: You can create new skills (create_skill) and use workflow.run_chain to combine skills. For any request, learn (browser/research), gain access (install, config), then do it (shell/script or new skill).")
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

    def _get_context_key(self, channel: str, user_id: str, branch_id: str = "") -> str:
        """Generate a unique key for a context. Main thread: channel:user_id. Branch: channel:user_id:branch_id."""
        base = f"{channel}:{user_id}"
        return f"{base}:{branch_id}" if branch_id else base

    async def get_context(
        self,
        channel: str,
        user_id: str,
        create_if_missing: bool = True,
        branch_id: str = "",
        topic_id: str = "",
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
        key = self._get_context_key(channel, user_id, branch_id)

        async with self._lock:
            if key in self._contexts:
                ctx = self._contexts[key]
                if topic_id and not ctx.topic_id:
                    ctx.topic_id = topic_id
                return ctx

            if not create_if_missing:
                return None

            # Try loading from disk first (handles multi-worker and restarts)
            path = self._get_persist_path(key)
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    context = ConversationContext.from_dict(data)
                    # Refresh system prompt (with workspace if enabled)
                    new_prompt = self._get_system_prompt_for_context(
                        context.user_id, context.channel
                    )
                    for i, msg in enumerate(context.messages):
                        if msg.role == "system":
                            context.messages[i] = LLMMessage(role="system", content=new_prompt)
                            break
                    if topic_id and not context.topic_id:
                        context.topic_id = topic_id
                    self._contexts[key] = context
                    logger.debug("Loaded context from disk", context_id=context.id, channel=channel, user_id=user_id)
                    return context
                except Exception as e:
                    logger.warning("Failed to load context from disk, creating new", key=key, error=str(e))

            # Create new context
            context = ConversationContext(
                channel=channel,
                user_id=user_id,
                branch_id=branch_id,
                topic_id=topic_id,
            )

            # Add system message (with workspace if enabled)
            system_prompt = self._get_system_prompt_for_context(user_id, channel)
            context.add_system_message(system_prompt)

            self._contexts[key] = context
            self._persist_context(key, context)
            logger.debug("Created new context", context_id=context.id, channel=channel, user_id=user_id)
            return context

    async def update_context(self, context: ConversationContext) -> None:
        """Update a context in storage and persist to disk."""
        key = self._get_context_key(context.channel, context.user_id, context.branch_id)
        async with self._lock:
            self._contexts[key] = context
            self._persist_context(key, context)

    async def save_context(self, context: ConversationContext) -> None:
        """Save a context to disk (call after adding messages)."""
        key = self._get_context_key(context.channel, context.user_id, context.branch_id)
        self._persist_context(key, context)

    async def search_conversations(
        self, query: str, channel: str = "", user_id: str = "", top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Search past conversations via vector memory."""
        if not self._vector_memory or not self._vector_memory.available:
            return []
        try:
            results = await self._vector_memory.search_messages(query, top_k=top_k)
            filtered = results
            if channel:
                filtered = [r for r in results if r.get("metadata", {}).get("channel") == channel]
            if user_id:
                filtered = [r for r in filtered if r.get("metadata", {}).get("user_id") == user_id]
            return filtered
        except Exception as e:
            logger.warning("Conversation search failed", error=str(e))
            return []

    async def save_checkpoint(
        self, channel: str, user_id: str, name: str
    ) -> dict[str, Any] | None:
        """Save current conversation state as a named checkpoint."""
        context = await self.get_context(channel, user_id, create_if_missing=False)
        if not context:
            return None
        snapshot = {
            "name": name,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in context.messages
                if m.role != "system"
            ],
            "topic_id": context.topic_id,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        context.set_variable(f"checkpoint_{name}", snapshot)
        await self.update_context(context)
        return snapshot

    async def load_checkpoint(
        self, channel: str, user_id: str, name: str
    ) -> dict[str, Any] | None:
        """Load a saved checkpoint. Returns snapshot dict or None."""
        context = await self.get_context(channel, user_id, create_if_missing=False)
        if not context:
            return None
        return context.get_variable(f"checkpoint_{name}")

    async def create_branch(
        self, channel: str, user_id: str, from_message_index: int
    ) -> ConversationContext | None:
        """Create a new branch from the current conversation at the given message index."""
        parent = await self.get_context(channel, user_id, create_if_missing=False)
        if not parent or from_message_index < 0 or from_message_index >= len(parent.messages):
            return None
        branch_id = str(uuid4())[:8]
        key = self._get_context_key(channel, user_id, branch_id)
        async with self._lock:
            branch = ConversationContext(
                channel=channel,
                user_id=user_id,
                branch_id=branch_id,
                parent_id=parent.id,
                topic_id=parent.topic_id,
            )
            system_msg = next((m for m in parent.messages if m.role == "system"), None)
            if system_msg:
                branch.add_message("system", system_msg.content)
            for i in range(1, from_message_index + 1):
                if i < len(parent.messages):
                    m = parent.messages[i]
                    branch.add_message(m.role, m.content, tool_calls=m.tool_calls, tool_call_id=m.tool_call_id)
            self._contexts[key] = branch
            self._persist_context(key, branch)
        return branch

    async def list_branches(self, channel: str, user_id: str) -> list[dict[str, Any]]:
        """List all branches for a channel/user."""
        prefix = f"{channel}:{user_id}:"
        async with self._lock:
            return [
                {
                    "branch_id": ctx.branch_id,
                    "parent_id": ctx.parent_id,
                    "message_count": len(ctx.messages),
                    "updated_at": ctx.updated_at.isoformat(),
                }
                for key, ctx in self._contexts.items()
                if key.startswith(prefix)
            ]

    async def delete_context(self, channel: str, user_id: str, branch_id: str = "") -> bool:
        """Delete a conversation context from memory and disk."""
        key = self._get_context_key(channel, user_id, branch_id)
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
            # Re-add system message (with workspace if enabled)
            system_prompt = self._get_system_prompt_for_context(user_id, channel)
            context.add_system_message(system_prompt)
            await self.update_context(context)
        return context

    def get_messages_for_llm(
        self,
        context: ConversationContext,
        context_window_tokens: int = 0,
        count_tokens: Any = None,
        prompt_mode: str = "full",
    ) -> list[LLMMessage]:
        """
        Return messages normalized (merge consecutive same-role), limited by turns.
        If prompt_mode is "minimal", system message is replaced with shorter prompt.
        Does not mutate context (returns new list).
        """
        messages = list(context.get_messages())
        if not messages:
            return []
        # Optionally use minimal system prompt to save tokens
        if prompt_mode == "minimal":
            minimal_prompt = self._build_system_prompt("minimal").replace(
                "{{current_date}}", datetime.now().strftime("%Y-%m-%d")
            ).replace("{{user_id}}", context.user_id).replace("{{channel}}", context.channel)
            out = []
            for m in messages:
                if m.role == "system":
                    out.append(LLMMessage(role="system", content=minimal_prompt))
                else:
                    out.append(m)
            messages = out
        # Turn normalization
        messages = normalize_turns(messages)
        # History limit by turns (and per-channel)
        max_turns = getattr(self.settings.memory.short_term, "max_turns", 0) or 0
        limit = get_history_limit_for_channel(context.channel, context.user_id)
        if limit is not None:
            max_turns = limit
        if max_turns > 0:
            messages = limit_history_by_turns(messages, max_turns)
        return messages

    async def trim_context(self, context: ConversationContext) -> None:
        """
        Trim a context to stay within message and token limits (reserve tokens for compaction).
        Uses per-context lock to serialize compaction. Optionally token-based when
        compaction_trigger_tokens is set.
        """
        key = self._get_context_key(context.channel, context.user_id, context.branch_id)
        async with self._compaction_lock(key):
            await self._trim_context_impl(context)

    async def _trim_context_impl(self, context: ConversationContext) -> None:
        max_messages = self.settings.memory.short_term.max_messages
        reserve = getattr(self.settings.memory.short_term, "compaction_reserve_tokens", 20_000)
        trigger = getattr(self.settings.memory.short_term, "compaction_trigger_tokens", 0)

        system_messages = [m for m in context.messages if m.role == "system"]
        other_messages = [m for m in context.messages if m.role != "system"]

        # Token-based target: if trigger set, trim until under (trigger - reserve)
        target_tokens = 0
        if trigger > 0 and reserve > 0 and self._llm_router:
            count_fn = self._llm_router.count_tokens
            target_tokens = max(0, trigger - reserve)
            current = estimate_tokens_messages(system_messages + other_messages, count_fn)
            if current <= target_tokens and len(context.messages) <= max_messages:
                await self.save_context(context)
                return

        if len(context.messages) <= max_messages and target_tokens == 0:
            await self.save_context(context)
            return

        keep_count = max_messages - len(system_messages)
        if target_tokens > 0 and self._llm_router:
            count_fn = self._llm_router.count_tokens
            # Reduce keep_count until estimated tokens under target
            while keep_count > 1:
                trimmed = other_messages[-keep_count:]
                est = estimate_tokens_messages(system_messages + trimmed, count_fn)
                if est <= target_tokens:
                    break
                keep_count = max(1, keep_count - 5)
        else:
            keep_count = max(1, keep_count)

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

        trimmed_messages = other_messages[-keep_count:] if keep_count > 0 else []
        context.messages = system_messages + trimmed_messages
        context.updated_at = datetime.now(timezone.utc)
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
