"""Base channel interface for messaging platforms."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine
from uuid import uuid4

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Types of messages."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    LOCATION = "location"
    REACTION = "reaction"
    SYSTEM = "system"


@dataclass
class Attachment:
    """Media attachment in a message."""

    type: MessageType
    url: str | None = None
    path: str | None = None
    mime_type: str | None = None
    filename: str | None = None
    size: int | None = None
    data: bytes | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """A message from a channel."""

    id: str = field(default_factory=lambda: str(uuid4()))
    channel: str = ""
    user_id: str = ""
    user_name: str | None = None
    content: str = ""
    message_type: MessageType = MessageType.TEXT
    attachments: list[Attachment] = field(default_factory=list)
    reply_to: str | None = None
    thread_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: Any = None  # Original message from the platform

    def has_attachments(self) -> bool:
        """Check if message has attachments."""
        return len(self.attachments) > 0

    def get_attachment_by_type(self, attachment_type: MessageType) -> Attachment | None:
        """Get first attachment of a specific type."""
        for attachment in self.attachments:
            if attachment.type == attachment_type:
                return attachment
        return None


# Type for message handler callbacks
MessageHandler = Callable[[Message], Coroutine[Any, Any, None]]


class BaseChannel(ABC):
    """
    Abstract base class for messaging channels.

    All channel implementations (Slack, WhatsApp, WebSocket, etc.)
    must inherit from this class and implement the abstract methods.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the channel.

        Args:
            name: Unique name for this channel
        """
        self.name = name
        self._connected = False
        self._message_handlers: list[MessageHandler] = []

    @property
    def is_connected(self) -> bool:
        """Check if the channel is connected."""
        return self._connected

    def on_message(self, handler: MessageHandler) -> None:
        """
        Register a message handler.

        Args:
            handler: Async function to call when a message is received
        """
        self._message_handlers.append(handler)

    async def _dispatch_message(self, message: Message) -> None:
        """
        Dispatch a message to all registered handlers.

        Args:
            message: The received message
        """
        if not self._message_handlers:
            logger.warning(
                "No message handlers registered for channel %s — message from %s dropped",
                self.name,
                message.user_id,
            )
            return

        for handler in self._message_handlers:
            try:
                await handler(message)
            except Exception as e:
                logger.error(
                    "Error in message handler for channel %s: %s",
                    self.name,
                    str(e),
                    exc_info=True,
                )

    @abstractmethod
    async def start(self) -> None:
        """
        Start the channel and begin listening for messages.

        This should establish connections and set up event handlers.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the channel and clean up resources.

        This should close connections and cancel any running tasks.
        """
        pass

    @abstractmethod
    async def send_message(
        self,
        user_id: str,
        content: str,
        reply_to: str | None = None,
        thread_id: str | None = None,
        attachments: list[Attachment] | None = None,
    ) -> str | None:
        """
        Send a message to a user.

        Args:
            user_id: The recipient's user ID
            content: Message content
            reply_to: Message ID to reply to
            thread_id: Thread ID for threaded replies
            attachments: Optional attachments

        Returns:
            The sent message ID, or None if failed
        """
        pass

    @abstractmethod
    async def send_reaction(
        self,
        message_id: str,
        reaction: str,
    ) -> bool:
        """
        Add a reaction to a message.

        Args:
            message_id: The message to react to
            reaction: The reaction (emoji or reaction name)

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def request_approval(
        self,
        user_id: str,
        action_description: str,
        approval_id: str,
        timeout: int = 300,
    ) -> dict[str, Any]:
        """
        Request approval from a user for an action.

        Args:
            user_id: The user to request approval from
            action_description: Description of the action
            approval_id: Unique ID for this approval request
            timeout: Timeout in seconds

        Returns:
            Dict with 'approved' (bool), 'approved_by' (str), and other metadata
        """
        pass

    async def send_typing_indicator(self, user_id: str) -> None:
        """
        Send a typing indicator to show the bot is working.

        Args:
            user_id: The user to show typing to

        Default implementation does nothing. Override if channel supports it.
        """
        pass

    async def get_user_info(self, user_id: str) -> dict[str, Any] | None:
        """
        Get information about a user.

        Args:
            user_id: The user ID

        Returns:
            Dict with user info, or None if not available

        Default implementation returns None. Override if channel supports it.
        """
        return None

    async def download_attachment(self, attachment: Attachment) -> bytes | None:
        """
        Download an attachment's data.

        Args:
            attachment: The attachment to download

        Returns:
            The attachment data, or None if failed

        Default implementation returns the existing data. Override if needed.
        """
        return attachment.data

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} connected={self._connected}>"
