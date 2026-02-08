"""WebSocket manager for real-time communication."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import WebSocket

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Connection:
    """A WebSocket connection."""

    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    websocket: WebSocket | None = None
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class WebSocketManager:
    """
    Manages WebSocket connections for real-time updates.

    Features:
    - User-based connection tracking
    - Broadcasting to all users
    - Targeted messaging
    - Connection health monitoring
    """

    def __init__(self) -> None:
        self._connections: dict[str, Connection] = {}
        self._user_connections: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
    ) -> Connection:
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: The WebSocket instance
            user_id: The authenticated user's ID

        Returns:
            The connection object
        """
        await websocket.accept()

        connection = Connection(
            user_id=user_id,
            websocket=websocket,
        )

        async with self._lock:
            self._connections[connection.id] = connection

            if user_id not in self._user_connections:
                self._user_connections[user_id] = []
            self._user_connections[user_id].append(connection.id)

        logger.debug(
            "WebSocket connected",
            connection_id=connection.id,
            user_id=user_id,
        )

        # Send welcome message
        await self._send(
            connection,
            {
                "type": "connected",
                "connection_id": connection.id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        return connection

    async def disconnect(self, connection_id: str) -> None:
        """
        Remove a WebSocket connection.

        Args:
            connection_id: The connection ID to remove
        """
        async with self._lock:
            connection = self._connections.pop(connection_id, None)
            if connection:
                user_id = connection.user_id
                if user_id in self._user_connections:
                    self._user_connections[user_id] = [
                        cid
                        for cid in self._user_connections[user_id]
                        if cid != connection_id
                    ]
                    if not self._user_connections[user_id]:
                        del self._user_connections[user_id]

        logger.debug("WebSocket disconnected", connection_id=connection_id)

    async def _send(
        self,
        connection: Connection,
        data: dict[str, Any],
    ) -> bool:
        """Send data to a specific connection."""
        try:
            if connection.websocket:
                await connection.websocket.send_json(data)
                return True
        except Exception as e:
            logger.warning(
                "Failed to send WebSocket message",
                connection_id=connection.id,
                error=str(e),
            )
        return False

    async def send_to_user(
        self,
        user_id: str,
        data: dict[str, Any],
    ) -> int:
        """
        Send data to all connections for a user.

        Args:
            user_id: The user ID
            data: Data to send

        Returns:
            Number of successful sends
        """
        async with self._lock:
            connection_ids = self._user_connections.get(user_id, []).copy()

        sent = 0
        for conn_id in connection_ids:
            connection = self._connections.get(conn_id)
            if connection and await self._send(connection, data):
                sent += 1

        return sent

    async def send_to_connection(
        self,
        connection_id: str,
        data: dict[str, Any],
    ) -> bool:
        """Send data to a specific connection."""
        connection = self._connections.get(connection_id)
        if connection:
            return await self._send(connection, data)
        return False

    async def broadcast(
        self,
        data: dict[str, Any],
        exclude_users: list[str] | None = None,
    ) -> int:
        """
        Broadcast data to all connected users.

        Args:
            data: Data to broadcast
            exclude_users: User IDs to exclude

        Returns:
            Number of successful sends
        """
        exclude_set = set(exclude_users or [])
        sent = 0

        async with self._lock:
            connections = list(self._connections.values())

        for connection in connections:
            if connection.user_id not in exclude_set:
                if await self._send(connection, data):
                    sent += 1

        return sent

    async def send_message(
        self,
        user_id: str,
        message_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Send a formatted message to a user.

        Args:
            user_id: The user ID
            message_type: Type of message
            content: Message content
            metadata: Additional metadata

        Returns:
            Number of successful sends
        """
        return await self.send_to_user(
            user_id,
            {
                "type": message_type,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **(metadata or {}),
            },
        )

    async def send_approval_request(
        self,
        user_id: str,
        approval_id: str,
        description: str,
        timeout: int,
    ) -> int:
        """Send an approval request to a user."""
        return await self.send_to_user(
            user_id,
            {
                "type": "approval_request",
                "approval_id": approval_id,
                "description": description,
                "timeout": timeout,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def send_stream_chunk(
        self,
        user_id: str,
        stream_id: str,
        content: str,
    ) -> int:
        """Send a streaming content chunk."""
        return await self.send_to_user(
            user_id,
            {
                "type": "stream_chunk",
                "stream_id": stream_id,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def send_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        level: str = "info",
    ) -> int:
        """Send a notification to a user."""
        return await self.send_to_user(
            user_id,
            {
                "type": "notification",
                "title": title,
                "message": message,
                "level": level,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def get_user_connections(self, user_id: str) -> list[str]:
        """Get all connection IDs for a user."""
        return self._user_connections.get(user_id, []).copy()

    def is_user_connected(self, user_id: str) -> bool:
        """Check if a user has any active connections."""
        return bool(self._user_connections.get(user_id))

    def get_stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        return {
            "total_connections": len(self._connections),
            "unique_users": len(self._user_connections),
            "connections_per_user": {
                user_id: len(conns)
                for user_id, conns in self._user_connections.items()
            },
        }


# Global instance
manager = WebSocketManager()


def get_ws_manager() -> WebSocketManager:
    """Get the global WebSocket manager."""
    return manager
