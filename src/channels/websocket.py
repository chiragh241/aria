"""WebSocket channel for web dashboard communication."""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ..utils.logging import get_logger
from .base import Attachment, BaseChannel, Message, MessageType

logger = get_logger(__name__)


@dataclass
class WebSocketConnection:
    """Represents a WebSocket connection."""

    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    websocket: Any = None  # fastapi.WebSocket
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class WebSocketChannel(BaseChannel):
    """
    WebSocket channel for real-time web communication.

    Features:
    - Multiple concurrent connections per user
    - Real-time message streaming
    - Connection health monitoring
    - Approval flow via web UI
    """

    def __init__(self) -> None:
        super().__init__("web")

        # Connection management
        self._connections: dict[str, WebSocketConnection] = {}
        self._user_connections: dict[str, list[str]] = {}  # user_id -> connection_ids

        # Pending approvals
        self._pending_approvals: dict[str, asyncio.Future[dict[str, Any]]] = {}

        # Lock for connection management
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the WebSocket channel."""
        self._connected = True
        logger.info("WebSocket channel started")

    async def stop(self) -> None:
        """Stop the WebSocket channel and close all connections."""
        # Close all connections
        async with self._lock:
            for conn in list(self._connections.values()):
                try:
                    if conn.websocket:
                        await conn.websocket.close()
                except Exception:
                    pass

            self._connections.clear()
            self._user_connections.clear()

        self._connected = False
        logger.info("WebSocket channel stopped")

    async def register_connection(
        self,
        websocket: Any,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> WebSocketConnection:
        """
        Register a new WebSocket connection.

        Args:
            websocket: The WebSocket instance
            user_id: The user's ID
            metadata: Optional connection metadata

        Returns:
            The connection object
        """
        connection = WebSocketConnection(
            user_id=user_id,
            websocket=websocket,
            metadata=metadata or {},
        )

        async with self._lock:
            self._connections[connection.id] = connection

            if user_id not in self._user_connections:
                self._user_connections[user_id] = []
            self._user_connections[user_id].append(connection.id)

        logger.debug(
            "WebSocket connection registered",
            connection_id=connection.id,
            user_id=user_id,
        )

        return connection

    async def unregister_connection(self, connection_id: str) -> None:
        """
        Unregister a WebSocket connection.

        Args:
            connection_id: The connection ID to remove
        """
        async with self._lock:
            connection = self._connections.pop(connection_id, None)
            if connection:
                user_id = connection.user_id
                if user_id in self._user_connections:
                    self._user_connections[user_id] = [
                        cid for cid in self._user_connections[user_id]
                        if cid != connection_id
                    ]
                    if not self._user_connections[user_id]:
                        del self._user_connections[user_id]

        logger.debug("WebSocket connection unregistered", connection_id=connection_id)

    async def handle_connection(
        self,
        websocket: Any,
        user_id: str,
    ) -> None:
        """
        Handle a WebSocket connection lifecycle.

        This should be called from the FastAPI WebSocket endpoint.

        Args:
            websocket: The WebSocket instance
            user_id: The authenticated user's ID
        """
        connection = await self.register_connection(websocket, user_id)

        try:
            # Send welcome message
            await self._send_to_connection(
                connection,
                {
                    "type": "connected",
                    "connection_id": connection.id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Listen for messages
            while True:
                try:
                    data = await websocket.receive_json()
                    await self._handle_incoming_message(connection, data)
                except Exception as e:
                    if "disconnect" in str(e).lower():
                        break
                    logger.error("WebSocket receive error", error=str(e))
                    break

        finally:
            await self.unregister_connection(connection.id)

    async def _handle_incoming_message(
        self,
        connection: WebSocketConnection,
        data: dict[str, Any],
    ) -> None:
        """Handle an incoming WebSocket message."""
        message_type = data.get("type", "message")
        connection.last_activity = datetime.now(timezone.utc)

        if message_type == "message":
            # Regular chat message
            message = Message(
                id=data.get("id", str(uuid4())),
                channel=self.name,
                user_id=connection.user_id,
                content=data.get("content", ""),
                message_type=MessageType.TEXT,
                metadata={
                    "connection_id": connection.id,
                    **data.get("metadata", {}),
                },
            )

            logger.debug(
                "Received WebSocket message",
                user_id=connection.user_id,
                message_id=message.id,
            )

            await self._dispatch_message(message)

        elif message_type == "approval_response":
            # Approval response
            approval_id = data.get("approval_id")
            if approval_id and approval_id in self._pending_approvals:
                future = self._pending_approvals[approval_id]
                future.set_result({
                    "approved": data.get("approved", False),
                    "approved_by": connection.user_id,
                })

        elif message_type == "ping":
            # Health check
            await self._send_to_connection(
                connection,
                {"type": "pong", "timestamp": datetime.now(timezone.utc).isoformat()},
            )

    async def _send_to_connection(
        self,
        connection: WebSocketConnection,
        data: dict[str, Any],
    ) -> bool:
        """Send data to a specific connection."""
        try:
            if connection.websocket:
                await connection.websocket.send_json(data)
                return True
        except Exception as e:
            logger.error(
                "Failed to send to WebSocket",
                connection_id=connection.id,
                error=str(e),
            )
        return False

    async def send_message(
        self,
        user_id: str,
        content: str,
        reply_to: str | None = None,
        thread_id: str | None = None,
        attachments: list[Attachment] | None = None,
    ) -> str | None:
        """Send a message to a user via WebSocket."""
        message_id = str(uuid4())

        data: dict[str, Any] = {
            "type": "message",
            "id": message_id,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if reply_to:
            data["reply_to"] = reply_to
        if thread_id:
            data["thread_id"] = thread_id
        if attachments:
            data["attachments"] = [
                {
                    "type": a.type.value,
                    "url": a.url,
                    "filename": a.filename,
                    "mime_type": a.mime_type,
                }
                for a in attachments
            ]

        # Send to all user's connections
        sent = False
        async with self._lock:
            connection_ids = self._user_connections.get(user_id, [])

        for conn_id in connection_ids:
            connection = self._connections.get(conn_id)
            if connection:
                if await self._send_to_connection(connection, data):
                    sent = True

        return message_id if sent else None

    async def send_reaction(
        self,
        message_id: str,
        reaction: str,
    ) -> bool:
        """Send a reaction notification (not really applicable for web)."""
        # Could implement as a notification
        return True

    async def request_approval(
        self,
        user_id: str,
        action_description: str,
        approval_id: str,
        timeout: int = 300,
    ) -> dict[str, Any]:
        """Request approval via the web interface."""
        # Create future for the response
        future: asyncio.Future[dict[str, Any]] = asyncio.Future()
        self._pending_approvals[approval_id] = future

        # Send approval request
        data = {
            "type": "approval_request",
            "approval_id": approval_id,
            "description": action_description,
            "timeout": timeout,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Send to all user's connections
        async with self._lock:
            connection_ids = self._user_connections.get(user_id, [])

        sent = False
        for conn_id in connection_ids:
            connection = self._connections.get(conn_id)
            if connection:
                if await self._send_to_connection(connection, data):
                    sent = True

        if not sent:
            self._pending_approvals.pop(approval_id, None)
            return {"approved": False, "error": "No active connections"}

        # Wait for response
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            return {"approved": False, "timeout": True}
        finally:
            self._pending_approvals.pop(approval_id, None)

    async def send_typing_indicator(self, user_id: str) -> None:
        """Send typing indicator to the web client."""
        data = {
            "type": "typing",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        async with self._lock:
            connection_ids = self._user_connections.get(user_id, [])

        for conn_id in connection_ids:
            connection = self._connections.get(conn_id)
            if connection:
                await self._send_to_connection(connection, data)

    async def broadcast(
        self,
        data: dict[str, Any],
        exclude_users: list[str] | None = None,
    ) -> int:
        """
        Broadcast a message to all connected users.

        Args:
            data: The data to broadcast
            exclude_users: List of user IDs to exclude

        Returns:
            Number of connections that received the message
        """
        exclude_set = set(exclude_users or [])
        sent_count = 0

        async with self._lock:
            connections = list(self._connections.values())

        for connection in connections:
            if connection.user_id not in exclude_set:
                if await self._send_to_connection(connection, data):
                    sent_count += 1

        return sent_count

    async def stream_to_user(
        self,
        user_id: str,
        stream_id: str,
        content_generator: Any,
    ) -> None:
        """
        Stream content to a user in real-time.

        Args:
            user_id: The user to stream to
            stream_id: Unique ID for this stream
            content_generator: Async generator yielding content chunks
        """
        # Send stream start
        await self.send_message(
            user_id,
            "",
            metadata={"type": "stream_start", "stream_id": stream_id},
        )

        try:
            async for chunk in content_generator:
                data = {
                    "type": "stream_chunk",
                    "stream_id": stream_id,
                    "content": chunk,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                async with self._lock:
                    connection_ids = self._user_connections.get(user_id, [])

                for conn_id in connection_ids:
                    connection = self._connections.get(conn_id)
                    if connection:
                        await self._send_to_connection(connection, data)

        finally:
            # Send stream end
            data = {
                "type": "stream_end",
                "stream_id": stream_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            async with self._lock:
                connection_ids = self._user_connections.get(user_id, [])

            for conn_id in connection_ids:
                connection = self._connections.get(conn_id)
                if connection:
                    await self._send_to_connection(connection, data)

    def get_stats(self) -> dict[str, Any]:
        """Get channel statistics."""
        return {
            "total_connections": len(self._connections),
            "unique_users": len(self._user_connections),
            "pending_approvals": len(self._pending_approvals),
        }
