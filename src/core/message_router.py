"""Async message routing and queue management."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine
from uuid import uuid4

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MessagePriority(int, Enum):
    """Message priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class MessageStatus(str, Enum):
    """Message processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueuedMessage:
    """A message in the processing queue."""

    id: str = field(default_factory=lambda: str(uuid4()))
    channel: str = ""
    user_id: str = ""
    content: str = ""
    message_type: str = "text"
    priority: MessagePriority = MessagePriority.NORMAL
    status: MessageStatus = MessageStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processed_at: datetime | None = None
    result: Any = None
    error: str | None = None

    # Original channel message reference
    channel_message_id: str | None = None

    # Media attachments
    attachments: list[dict[str, Any]] = field(default_factory=list)

    def __lt__(self, other: "QueuedMessage") -> bool:
        """Compare by priority (higher priority first) then by creation time."""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


MessageHandler = Callable[[QueuedMessage], Coroutine[Any, Any, Any]]


class MessageRouter:
    """
    Async message router with priority queue support.

    Features:
    - Priority-based message processing
    - Concurrent processing with configurable workers
    - Message deduplication
    - Retry handling
    - Event callbacks for message lifecycle
    """

    def __init__(
        self,
        max_workers: int = 5,
        max_queue_size: int = 1000,
    ) -> None:
        self.settings = get_settings()
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size

        # Priority queue for messages
        self._queue: asyncio.PriorityQueue[QueuedMessage] = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )

        # Message storage for deduplication and status tracking
        self._messages: dict[str, QueuedMessage] = {}
        self._lock = asyncio.Lock()

        # Message handlers by type
        self._handlers: dict[str, MessageHandler] = {}
        self._default_handler: MessageHandler | None = None

        # Event callbacks
        self._on_message_received: list[Callable[[QueuedMessage], Coroutine[Any, Any, None]]] = []
        self._on_message_completed: list[Callable[[QueuedMessage], Coroutine[Any, Any, None]]] = []
        self._on_message_failed: list[Callable[[QueuedMessage], Coroutine[Any, Any, None]]] = []

        # Worker management
        self._workers: list[asyncio.Task[None]] = []
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize the message router."""
        logger.info("Message router initialized", max_workers=self.max_workers, max_queue=self.max_queue_size)

    def register_handler(
        self,
        message_type: str,
        handler: MessageHandler,
    ) -> None:
        """Register a handler for a specific message type."""
        self._handlers[message_type] = handler
        logger.debug("Registered handler", message_type=message_type)

    def set_default_handler(self, handler: MessageHandler) -> None:
        """Set the default handler for untyped messages."""
        self._default_handler = handler

    def on_message_received(
        self,
        callback: Callable[[QueuedMessage], Coroutine[Any, Any, None]],
    ) -> None:
        """Register a callback for when a message is received."""
        self._on_message_received.append(callback)

    def on_message_completed(
        self,
        callback: Callable[[QueuedMessage], Coroutine[Any, Any, None]],
    ) -> None:
        """Register a callback for when a message is completed."""
        self._on_message_completed.append(callback)

    def on_message_failed(
        self,
        callback: Callable[[QueuedMessage], Coroutine[Any, Any, None]],
    ) -> None:
        """Register a callback for when a message fails."""
        self._on_message_failed.append(callback)

    async def enqueue(
        self,
        channel: str,
        user_id: str,
        content: str,
        message_type: str = "text",
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: dict[str, Any] | None = None,
        channel_message_id: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> QueuedMessage:
        """
        Add a message to the processing queue.

        Args:
            channel: Source channel identifier
            user_id: User identifier
            content: Message content
            message_type: Type of message for routing
            priority: Processing priority
            metadata: Additional message metadata
            channel_message_id: Original message ID from the channel
            attachments: Media attachments

        Returns:
            The queued message
        """
        message = QueuedMessage(
            channel=channel,
            user_id=user_id,
            content=content,
            message_type=message_type,
            priority=priority,
            metadata=metadata or {},
            channel_message_id=channel_message_id,
            attachments=attachments or [],
        )

        # Check for duplicate (same channel message ID)
        if channel_message_id:
            async with self._lock:
                for existing in self._messages.values():
                    if (
                        existing.channel_message_id == channel_message_id
                        and existing.status in (MessageStatus.PENDING, MessageStatus.PROCESSING)
                    ):
                        logger.debug(
                            "Duplicate message ignored",
                            message_id=channel_message_id,
                        )
                        return existing

        # Store and queue
        async with self._lock:
            self._messages[message.id] = message

        await self._queue.put(message)

        logger.debug(
            "Message enqueued",
            message_id=message.id,
            channel=channel,
            priority=priority.name,
        )

        # Fire received callbacks
        for callback in self._on_message_received:
            try:
                await callback(message)
            except Exception as e:
                logger.error("Message received callback failed", error=str(e))

        return message

    async def get_message(self, message_id: str) -> QueuedMessage | None:
        """Get a message by ID."""
        async with self._lock:
            return self._messages.get(message_id)

    async def cancel_message(self, message_id: str) -> bool:
        """Cancel a pending message."""
        async with self._lock:
            message = self._messages.get(message_id)
            if message and message.status == MessageStatus.PENDING:
                message.status = MessageStatus.CANCELLED
                return True
            return False

    async def _process_message(self, message: QueuedMessage) -> None:
        """Process a single message."""
        if message.status == MessageStatus.CANCELLED:
            return

        message.status = MessageStatus.PROCESSING
        logger.debug("Processing message", message_id=message.id, type=message.message_type)

        try:
            # Find handler
            handler = self._handlers.get(message.message_type, self._default_handler)

            if handler is None:
                raise ValueError(f"No handler for message type: {message.message_type}")

            # Execute handler
            result = await handler(message)
            message.result = result
            message.status = MessageStatus.COMPLETED
            message.processed_at = datetime.now(timezone.utc)

            logger.debug("Message processed", message_id=message.id)

            # Fire completed callbacks
            for callback in self._on_message_completed:
                try:
                    await callback(message)
                except Exception as e:
                    logger.error("Message completed callback failed", error=str(e))

        except Exception as e:
            message.status = MessageStatus.FAILED
            message.error = str(e)
            message.processed_at = datetime.now(timezone.utc)

            logger.error(
                "Message processing failed",
                message_id=message.id,
                error=str(e),
            )

            # Fire failed callbacks
            for callback in self._on_message_failed:
                try:
                    await callback(message)
                except Exception as cb_error:
                    logger.error("Message failed callback failed", error=str(cb_error))

    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine that processes messages from the queue."""
        logger.debug("Worker started", worker_id=worker_id)

        while self._running:
            try:
                # Wait for a message with timeout
                try:
                    message = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                await self._process_message(message)
                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Worker error", worker_id=worker_id, error=str(e))

        logger.debug("Worker stopped", worker_id=worker_id)

    async def start(self) -> None:
        """Start the message router workers."""
        if self._running:
            return

        self._running = True
        self._shutdown_event.clear()

        # Start workers
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker(i))
            self._workers.append(task)

        logger.info("Message router started", workers=self.max_workers)

    async def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the message router gracefully.

        Args:
            timeout: Maximum time to wait for workers to finish
        """
        if not self._running:
            return

        self._running = False
        self._shutdown_event.set()

        # Wait for queue to be processed
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Queue drain timeout, cancelling workers")

        # Cancel workers
        for task in self._workers:
            task.cancel()

        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

        self._workers = []
        logger.info("Message router stopped")

    async def wait_for_message(
        self,
        message_id: str,
        timeout: float | None = None,
    ) -> QueuedMessage | None:
        """
        Wait for a message to be processed.

        Args:
            message_id: The message ID to wait for
            timeout: Maximum time to wait

        Returns:
            The processed message, or None if timeout
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            message = await self.get_message(message_id)
            if message and message.status in (
                MessageStatus.COMPLETED,
                MessageStatus.FAILED,
                MessageStatus.CANCELLED,
            ):
                return message

            if timeout is not None:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    return None

            await asyncio.sleep(0.1)

    def get_stats(self) -> dict[str, Any]:
        """Get router statistics."""
        status_counts: dict[str, int] = {}
        for message in self._messages.values():
            status = message.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "queue_size": self._queue.qsize(),
            "total_messages": len(self._messages),
            "running": self._running,
            "workers": len(self._workers),
            "status_counts": status_counts,
        }

    async def cleanup_old_messages(self, max_age_seconds: int = 3600) -> int:
        """
        Remove old completed/failed messages from storage.

        Args:
            max_age_seconds: Maximum age of messages to keep

        Returns:
            Number of messages removed
        """
        now = datetime.now(timezone.utc)
        to_remove = []

        async with self._lock:
            for msg_id, message in self._messages.items():
                if message.status in (
                    MessageStatus.COMPLETED,
                    MessageStatus.FAILED,
                    MessageStatus.CANCELLED,
                ):
                    if message.processed_at:
                        age = (now - message.processed_at).total_seconds()
                        if age > max_age_seconds:
                            to_remove.append(msg_id)

            for msg_id in to_remove:
                del self._messages[msg_id]

        if to_remove:
            logger.debug("Cleaned up old messages", count=len(to_remove))

        return len(to_remove)
