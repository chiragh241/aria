"""Short-term memory for conversation context."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryItem:
    """A single item in short-term memory."""

    content: str
    role: str  # user, assistant, system
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "role": self.role,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryItem":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            role=data["role"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(timezone.utc),
            metadata=data.get("metadata", {}),
            token_count=data.get("token_count", 0),
        )


class ShortTermMemory:
    """
    Short-term memory for maintaining conversation context.

    Features:
    - Fixed-size sliding window
    - Token counting for context management
    - Automatic summarization of old context
    - Working memory for current task
    """

    def __init__(
        self,
        max_messages: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.settings = get_settings()
        self.max_messages = max_messages or self.settings.memory.short_term.max_messages
        self.max_tokens = max_tokens or self.settings.memory.short_term.max_tokens

        self._messages: deque[MemoryItem] = deque(maxlen=self.max_messages * 2)  # Extra buffer
        self._total_tokens = 0

        # Working memory for current task
        self._working_memory: dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize short-term memory."""
        logger.info("Short-term memory initialized", max_messages=self.max_messages, max_tokens=self.max_tokens)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string."""
        # Simple estimation: ~4 characters per token
        return len(text) // 4 + 1

    def add(
        self,
        content: str,
        role: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        """
        Add a message to short-term memory.

        Args:
            content: Message content
            role: Message role (user, assistant, system)
            metadata: Optional metadata

        Returns:
            The created memory item
        """
        token_count = self._estimate_tokens(content)

        item = MemoryItem(
            content=content,
            role=role,
            metadata=metadata or {},
            token_count=token_count,
        )

        self._messages.append(item)
        self._total_tokens += token_count

        # Trim if over token limit
        self._trim_to_limit()

        return item

    def add_user_message(self, content: str, **metadata: Any) -> MemoryItem:
        """Add a user message."""
        return self.add(content, "user", metadata)

    def add_assistant_message(self, content: str, **metadata: Any) -> MemoryItem:
        """Add an assistant message."""
        return self.add(content, "assistant", metadata)

    def add_system_message(self, content: str, **metadata: Any) -> MemoryItem:
        """Add a system message."""
        return self.add(content, "system", metadata)

    def _trim_to_limit(self) -> None:
        """Trim messages to stay within token and message limits."""
        # First, ensure we're within message count
        while len(self._messages) > self.max_messages:
            removed = self._messages.popleft()
            self._total_tokens -= removed.token_count

        # Then, ensure we're within token limit
        while self._total_tokens > self.max_tokens and len(self._messages) > 1:
            # Keep at least system message and last exchange
            if len(self._messages) <= 3:
                break
            removed = self._messages.popleft()
            self._total_tokens -= removed.token_count

    def get_messages(
        self,
        limit: int | None = None,
        include_system: bool = True,
    ) -> list[MemoryItem]:
        """
        Get messages from memory.

        Args:
            limit: Maximum number of messages
            include_system: Whether to include system messages

        Returns:
            List of memory items
        """
        messages = list(self._messages)

        if not include_system:
            messages = [m for m in messages if m.role != "system"]

        if limit:
            # Keep system messages plus last N
            system_msgs = [m for m in messages if m.role == "system"]
            other_msgs = [m for m in messages if m.role != "system"]
            messages = system_msgs + other_msgs[-limit:]

        return messages

    def get_context_string(self, include_system: bool = True) -> str:
        """Get all messages as a formatted string."""
        messages = self.get_messages(include_system=include_system)
        parts = []
        for msg in messages:
            parts.append(f"{msg.role.upper()}: {msg.content}")
        return "\n\n".join(parts)

    def get_last_message(self, role: str | None = None) -> MemoryItem | None:
        """Get the last message, optionally filtered by role."""
        for msg in reversed(self._messages):
            if role is None or msg.role == role:
                return msg
        return None

    def clear(self) -> None:
        """Clear all messages from memory."""
        self._messages.clear()
        self._total_tokens = 0

    def clear_except_system(self) -> None:
        """Clear all messages except system messages."""
        system_msgs = [m for m in self._messages if m.role == "system"]
        self._messages.clear()
        self._total_tokens = 0
        for msg in system_msgs:
            self._messages.append(msg)
            self._total_tokens += msg.token_count

    # Working memory operations
    def set_working(self, key: str, value: Any) -> None:
        """Set a value in working memory."""
        self._working_memory[key] = value

    def get_working(self, key: str, default: Any = None) -> Any:
        """Get a value from working memory."""
        return self._working_memory.get(key, default)

    def clear_working(self) -> None:
        """Clear working memory."""
        self._working_memory.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        role_counts: dict[str, int] = {}
        for msg in self._messages:
            role_counts[msg.role] = role_counts.get(msg.role, 0) + 1

        return {
            "message_count": len(self._messages),
            "token_count": self._total_tokens,
            "max_messages": self.max_messages,
            "max_tokens": self.max_tokens,
            "by_role": role_counts,
            "working_memory_keys": list(self._working_memory.keys()),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "messages": [m.to_dict() for m in self._messages],
            "working_memory": self._working_memory,
            "stats": self.get_stats(),
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore from dictionary."""
        self._messages.clear()
        self._total_tokens = 0

        for msg_data in data.get("messages", []):
            item = MemoryItem.from_dict(msg_data)
            self._messages.append(item)
            self._total_tokens += item.token_count

        self._working_memory = data.get("working_memory", {})

    async def summarize(self, summarizer: Any = None) -> str | None:
        """
        Summarize old messages to compress context.

        Args:
            summarizer: Optional function to generate summary

        Returns:
            Summary string, or None if summarization not needed
        """
        if len(self._messages) < self.max_messages // 2:
            return None

        # Get messages to summarize (all except recent ones)
        all_messages = list(self._messages)
        keep_count = min(10, len(all_messages) // 3)
        to_summarize = all_messages[:-keep_count]
        to_keep = all_messages[-keep_count:]

        if len(to_summarize) < 5:
            return None

        # Build summary text
        summary_input = self.get_context_string()

        if summarizer:
            summary = await summarizer(summary_input)
        else:
            # Simple extractive summary
            user_messages = [m.content for m in to_summarize if m.role == "user"]
            summary = "Previous conversation summary:\n" + "\n".join(
                f"- {msg[:100]}..." if len(msg) > 100 else f"- {msg}"
                for msg in user_messages[-5:]
            )

        # Replace old messages with summary
        self._messages.clear()
        self._total_tokens = 0

        # Add summary as system message
        self.add_system_message(summary)

        # Add back recent messages
        for msg in to_keep:
            self._messages.append(msg)
            self._total_tokens += msg.token_count

        return summary
