"""Audit logging system for tracking all actions."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""

    pass


class AuditRecord(Base):
    """SQLAlchemy model for audit records."""

    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(String(36), unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    event = Column(String(100), nullable=False)
    action_type = Column(String(100))
    user_id = Column(String(100))
    channel = Column(String(50))
    status = Column(String(50))
    details = Column(Text)  # JSON string
    session_id = Column(String(36))


@dataclass
class AuditEntry:
    """An audit log entry."""

    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event: str = ""
    action_type: str = ""
    user_id: str | None = None
    channel: str | None = None
    status: str = "info"
    details: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None


class AuditLogger:
    """
    Persistent audit logger that stores all actions to SQLite.

    Features:
    - Async database operations
    - Searchable audit trail
    - Session replay capability
    - Retention policies
    """

    def __init__(self, db_url: str | None = None) -> None:
        self.settings = get_settings()
        self._db_url = db_url or self.settings.database.url

        # Ensure async driver
        if "sqlite:///" in self._db_url and "aiosqlite" not in self._db_url:
            self._db_url = self._db_url.replace("sqlite:///", "sqlite+aiosqlite:///")

        # Ensure directory exists
        if "sqlite" in self._db_url:
            db_path = self._db_url.split("///")[-1]
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_async_engine(self._db_url, echo=False)
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        self._initialized = False
        self._current_session_id: str | None = None

    async def initialize(self) -> None:
        """Initialize the database schema."""
        if self._initialized:
            return

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        self._initialized = True
        self._current_session_id = str(uuid4())
        logger.info("Audit logger initialized", session_id=self._current_session_id)

    async def log(
        self,
        event: str,
        action_type: str = "",
        user_id: str | None = None,
        channel: str | None = None,
        status: str = "info",
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """
        Log an audit entry.

        Args:
            event: Event name
            action_type: Type of action
            user_id: User who triggered the event
            channel: Channel where it occurred
            status: Event status (info, warning, error)
            details: Additional details

        Returns:
            The created audit entry
        """
        if not self._initialized:
            await self.initialize()

        import orjson

        entry = AuditEntry(
            event=event,
            action_type=action_type,
            user_id=user_id,
            channel=channel,
            status=status,
            details=details or {},
            session_id=self._current_session_id,
        )

        record = AuditRecord(
            record_id=entry.id,
            timestamp=entry.timestamp,
            event=entry.event,
            action_type=entry.action_type,
            user_id=entry.user_id,
            channel=entry.channel,
            status=entry.status,
            details=orjson.dumps(entry.details).decode(),
            session_id=entry.session_id,
        )

        async with self._session_factory() as session:
            session.add(record)
            await session.commit()

        return entry

    async def action_requested(
        self,
        action_type: str,
        description: str,
        user_id: str | None = None,
        channel: str | None = None,
    ) -> AuditEntry:
        """Log an action request."""
        return await self.log(
            event="action_requested",
            action_type=action_type,
            user_id=user_id,
            channel=channel,
            details={"description": description},
        )

    async def action_approved(
        self,
        action_type: str,
        approved_by: str,
        channel: str | None = None,
    ) -> AuditEntry:
        """Log an action approval."""
        return await self.log(
            event="action_approved",
            action_type=action_type,
            user_id=approved_by,
            channel=channel,
        )

    async def action_denied(
        self,
        action_type: str,
        denied_by: str | None = None,
        reason: str | None = None,
        channel: str | None = None,
    ) -> AuditEntry:
        """Log an action denial."""
        return await self.log(
            event="action_denied",
            action_type=action_type,
            user_id=denied_by,
            channel=channel,
            status="warning",
            details={"reason": reason},
        )

    async def action_executed(
        self,
        action_type: str,
        result: str,
        execution_time_ms: float | None = None,
        user_id: str | None = None,
        channel: str | None = None,
    ) -> AuditEntry:
        """Log an action execution."""
        return await self.log(
            event="action_executed",
            action_type=action_type,
            user_id=user_id,
            channel=channel,
            details={"result": result, "execution_time_ms": execution_time_ms},
        )

    async def action_failed(
        self,
        action_type: str,
        error: str,
        user_id: str | None = None,
        channel: str | None = None,
    ) -> AuditEntry:
        """Log an action failure."""
        return await self.log(
            event="action_failed",
            action_type=action_type,
            user_id=user_id,
            channel=channel,
            status="error",
            details={"error": error},
        )

    async def security_violation(
        self,
        action_type: str,
        violation_type: str,
        details: dict[str, Any] | None = None,
        user_id: str | None = None,
        channel: str | None = None,
    ) -> AuditEntry:
        """Log a security violation."""
        return await self.log(
            event="security_violation",
            action_type=action_type,
            user_id=user_id,
            channel=channel,
            status="error",
            details={"violation_type": violation_type, **(details or {})},
        )

    async def query(
        self,
        event: str | None = None,
        action_type: str | None = None,
        user_id: str | None = None,
        channel: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """
        Query audit log entries.

        Args:
            event: Filter by event name
            action_type: Filter by action type
            user_id: Filter by user
            channel: Filter by channel
            status: Filter by status
            session_id: Filter by session
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum results
            offset: Result offset

        Returns:
            List of matching audit entries
        """
        if not self._initialized:
            await self.initialize()

        import orjson

        stmt = select(AuditRecord).order_by(AuditRecord.timestamp.desc())

        if event:
            stmt = stmt.where(AuditRecord.event == event)
        if action_type:
            stmt = stmt.where(AuditRecord.action_type == action_type)
        if user_id:
            stmt = stmt.where(AuditRecord.user_id == user_id)
        if channel:
            stmt = stmt.where(AuditRecord.channel == channel)
        if status:
            stmt = stmt.where(AuditRecord.status == status)
        if session_id:
            stmt = stmt.where(AuditRecord.session_id == session_id)
        if start_time:
            stmt = stmt.where(AuditRecord.timestamp >= start_time)
        if end_time:
            stmt = stmt.where(AuditRecord.timestamp <= end_time)

        stmt = stmt.limit(limit).offset(offset)

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            records = result.scalars().all()

        entries = []
        for record in records:
            entries.append(
                AuditEntry(
                    id=record.record_id,
                    timestamp=record.timestamp,
                    event=record.event,
                    action_type=record.action_type or "",
                    user_id=record.user_id,
                    channel=record.channel,
                    status=record.status or "info",
                    details=orjson.loads(record.details) if record.details else {},
                    session_id=record.session_id,
                )
            )

        return entries

    async def get_session_replay(self, session_id: str) -> list[AuditEntry]:
        """Get all entries for a session in chronological order."""
        return await self.query(
            session_id=session_id,
            limit=10000,
        )

    async def get_stats(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Get audit statistics."""
        if not self._initialized:
            await self.initialize()

        from sqlalchemy import func

        async with self._session_factory() as session:
            # Total count
            total_stmt = select(func.count(AuditRecord.id))
            if start_time:
                total_stmt = total_stmt.where(AuditRecord.timestamp >= start_time)
            if end_time:
                total_stmt = total_stmt.where(AuditRecord.timestamp <= end_time)
            total_result = await session.execute(total_stmt)
            total = total_result.scalar() or 0

            # Count by event
            event_stmt = select(
                AuditRecord.event,
                func.count(AuditRecord.id),
            ).group_by(AuditRecord.event)
            if start_time:
                event_stmt = event_stmt.where(AuditRecord.timestamp >= start_time)
            if end_time:
                event_stmt = event_stmt.where(AuditRecord.timestamp <= end_time)
            event_result = await session.execute(event_stmt)
            by_event = {row[0]: row[1] for row in event_result}

            # Count by status
            status_stmt = select(
                AuditRecord.status,
                func.count(AuditRecord.id),
            ).group_by(AuditRecord.status)
            if start_time:
                status_stmt = status_stmt.where(AuditRecord.timestamp >= start_time)
            if end_time:
                status_stmt = status_stmt.where(AuditRecord.timestamp <= end_time)
            status_result = await session.execute(status_stmt)
            by_status = {row[0]: row[1] for row in status_result}

        return {
            "total_entries": total,
            "by_event": by_event,
            "by_status": by_status,
            "current_session": self._current_session_id,
        }

    async def cleanup(self, retention_days: int = 30) -> int:
        """
        Remove audit entries older than retention period.

        Args:
            retention_days: Number of days to retain

        Returns:
            Number of entries removed
        """
        if not self._initialized:
            await self.initialize()

        from datetime import timedelta

        from sqlalchemy import delete

        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

        async with self._session_factory() as session:
            stmt = delete(AuditRecord).where(AuditRecord.timestamp < cutoff)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount or 0
