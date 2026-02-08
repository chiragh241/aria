"""Episodic memory for task history and learning."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""

    pass


class EpisodeRecord(Base):
    """SQLAlchemy model for episodes."""

    __tablename__ = "episodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(String(36), unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    task_type = Column(String(100))
    description = Column(Text)
    input_summary = Column(Text)
    output_summary = Column(Text)
    success = Column(Integer)  # 0 or 1
    duration_ms = Column(Float)
    user_id = Column(String(100))
    channel = Column(String(50))
    feedback = Column(String(50))  # positive, negative, neutral
    skills_used = Column(Text)  # JSON array
    corrections = Column(Text)  # JSON array of corrections
    extra_data = Column(Text)  # JSON metadata


@dataclass
class Episode:
    """Represents a completed task episode."""

    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    task_type: str = ""
    description: str = ""
    input_summary: str = ""
    output_summary: str = ""
    success: bool = True
    duration_ms: float = 0
    user_id: str | None = None
    channel: str | None = None
    feedback: str = "neutral"  # positive, negative, neutral
    skills_used: list[str] = field(default_factory=list)
    corrections: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    """
    Episodic memory for tracking task completion history.

    Features:
    - Records completed tasks with outcomes
    - Tracks user corrections for learning
    - Enables pattern recognition in task handling
    - Provides statistics for performance improvement
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

    async def initialize(self) -> None:
        """Initialize the database schema."""
        if self._initialized:
            return

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        self._initialized = True
        logger.info("Episodic memory initialized")

    async def record(self, episode: Episode) -> Episode:
        """
        Record a completed episode.

        Args:
            episode: The episode to record

        Returns:
            The recorded episode
        """
        if not self._initialized:
            await self.initialize()

        import orjson

        record = EpisodeRecord(
            episode_id=episode.id,
            timestamp=episode.timestamp,
            task_type=episode.task_type,
            description=episode.description,
            input_summary=episode.input_summary,
            output_summary=episode.output_summary,
            success=1 if episode.success else 0,
            duration_ms=episode.duration_ms,
            user_id=episode.user_id,
            channel=episode.channel,
            feedback=episode.feedback,
            skills_used=orjson.dumps(episode.skills_used).decode(),
            corrections=orjson.dumps(episode.corrections).decode(),
            extra_data=orjson.dumps(episode.metadata).decode(),
        )

        async with self._session_factory() as session:
            session.add(record)
            await session.commit()

        logger.debug("Recorded episode", episode_id=episode.id, task_type=episode.task_type)
        return episode

    async def add_correction(
        self,
        episode_id: str,
        correction: dict[str, Any],
    ) -> bool:
        """
        Add a correction to an episode.

        Args:
            episode_id: The episode ID
            correction: The correction details

        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()

        import orjson

        async with self._session_factory() as session:
            stmt = select(EpisodeRecord).where(EpisodeRecord.episode_id == episode_id)
            result = await session.execute(stmt)
            record = result.scalar_one_or_none()

            if not record:
                return False

            corrections = orjson.loads(record.corrections) if record.corrections else []
            corrections.append(correction)
            record.corrections = orjson.dumps(corrections).decode()
            record.feedback = "negative"  # Corrections imply negative feedback

            await session.commit()

        return True

    async def add_feedback(
        self,
        episode_id: str,
        feedback: str,
    ) -> bool:
        """
        Add feedback to an episode.

        Args:
            episode_id: The episode ID
            feedback: positive, negative, or neutral

        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()

        async with self._session_factory() as session:
            stmt = select(EpisodeRecord).where(EpisodeRecord.episode_id == episode_id)
            result = await session.execute(stmt)
            record = result.scalar_one_or_none()

            if not record:
                return False

            record.feedback = feedback
            await session.commit()

        return True

    async def get(self, episode_id: str) -> Episode | None:
        """Get an episode by ID."""
        if not self._initialized:
            await self.initialize()

        import orjson

        async with self._session_factory() as session:
            stmt = select(EpisodeRecord).where(EpisodeRecord.episode_id == episode_id)
            result = await session.execute(stmt)
            record = result.scalar_one_or_none()

            if not record:
                return None

            return Episode(
                id=record.episode_id,
                timestamp=record.timestamp,
                task_type=record.task_type or "",
                description=record.description or "",
                input_summary=record.input_summary or "",
                output_summary=record.output_summary or "",
                success=bool(record.success),
                duration_ms=record.duration_ms or 0,
                user_id=record.user_id,
                channel=record.channel,
                feedback=record.feedback or "neutral",
                skills_used=orjson.loads(record.skills_used) if record.skills_used else [],
                corrections=orjson.loads(record.corrections) if record.corrections else [],
                metadata=orjson.loads(record.extra_data) if record.extra_data else {},
            )

    async def search(
        self,
        task_type: str | None = None,
        user_id: str | None = None,
        channel: str | None = None,
        success: bool | None = None,
        feedback: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Episode]:
        """
        Search for episodes.

        Args:
            task_type: Filter by task type
            user_id: Filter by user
            channel: Filter by channel
            success: Filter by success status
            feedback: Filter by feedback
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum results
            offset: Result offset

        Returns:
            List of matching episodes
        """
        if not self._initialized:
            await self.initialize()

        import orjson

        stmt = select(EpisodeRecord).order_by(EpisodeRecord.timestamp.desc())

        if task_type:
            stmt = stmt.where(EpisodeRecord.task_type == task_type)
        if user_id:
            stmt = stmt.where(EpisodeRecord.user_id == user_id)
        if channel:
            stmt = stmt.where(EpisodeRecord.channel == channel)
        if success is not None:
            stmt = stmt.where(EpisodeRecord.success == (1 if success else 0))
        if feedback:
            stmt = stmt.where(EpisodeRecord.feedback == feedback)
        if start_time:
            stmt = stmt.where(EpisodeRecord.timestamp >= start_time)
        if end_time:
            stmt = stmt.where(EpisodeRecord.timestamp <= end_time)

        stmt = stmt.limit(limit).offset(offset)

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            records = result.scalars().all()

        episodes = []
        for record in records:
            episodes.append(
                Episode(
                    id=record.episode_id,
                    timestamp=record.timestamp,
                    task_type=record.task_type or "",
                    description=record.description or "",
                    input_summary=record.input_summary or "",
                    output_summary=record.output_summary or "",
                    success=bool(record.success),
                    duration_ms=record.duration_ms or 0,
                    user_id=record.user_id,
                    channel=record.channel,
                    feedback=record.feedback or "neutral",
                    skills_used=orjson.loads(record.skills_used) if record.skills_used else [],
                    corrections=orjson.loads(record.corrections) if record.corrections else [],
                    metadata=orjson.loads(record.metadata) if record.metadata else {},
                )
            )

        return episodes

    async def get_similar_episodes(
        self,
        task_type: str,
        limit: int = 5,
    ) -> list[Episode]:
        """
        Get similar past episodes for learning.

        Args:
            task_type: The current task type
            limit: Maximum results

        Returns:
            List of similar episodes, prioritizing successful ones
        """
        # First, get successful episodes of the same type
        successful = await self.search(
            task_type=task_type,
            success=True,
            feedback="positive",
            limit=limit,
        )

        if len(successful) >= limit:
            return successful

        # Fill with other successful episodes
        more_successful = await self.search(
            task_type=task_type,
            success=True,
            limit=limit - len(successful),
        )

        return successful + [e for e in more_successful if e.id not in {s.id for s in successful}]

    async def get_corrections_for_task_type(
        self,
        task_type: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get corrections from past episodes of a task type.

        Useful for learning from past mistakes.

        Args:
            task_type: The task type
            limit: Maximum corrections

        Returns:
            List of correction dictionaries
        """
        episodes = await self.search(
            task_type=task_type,
            feedback="negative",
            limit=limit * 2,  # Get more to filter for corrections
        )

        corrections = []
        for episode in episodes:
            corrections.extend(episode.corrections)
            if len(corrections) >= limit:
                break

        return corrections[:limit]

    async def get_stats(
        self,
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Get episodic memory statistics."""
        if not self._initialized:
            await self.initialize()

        from sqlalchemy import func

        async with self._session_factory() as session:
            # Base query
            base_stmt = select(EpisodeRecord)
            if user_id:
                base_stmt = base_stmt.where(EpisodeRecord.user_id == user_id)
            if start_time:
                base_stmt = base_stmt.where(EpisodeRecord.timestamp >= start_time)
            if end_time:
                base_stmt = base_stmt.where(EpisodeRecord.timestamp <= end_time)

            # Total count
            total_stmt = select(func.count(EpisodeRecord.id))
            total_result = await session.execute(total_stmt)
            total = total_result.scalar() or 0

            # Success rate
            success_stmt = select(func.count(EpisodeRecord.id)).where(
                EpisodeRecord.success == 1
            )
            success_result = await session.execute(success_stmt)
            successful = success_result.scalar() or 0

            # By task type
            type_stmt = select(
                EpisodeRecord.task_type,
                func.count(EpisodeRecord.id),
            ).group_by(EpisodeRecord.task_type)
            type_result = await session.execute(type_stmt)
            by_type = {row[0]: row[1] for row in type_result if row[0]}

            # By feedback
            feedback_stmt = select(
                EpisodeRecord.feedback,
                func.count(EpisodeRecord.id),
            ).group_by(EpisodeRecord.feedback)
            feedback_result = await session.execute(feedback_stmt)
            by_feedback = {row[0]: row[1] for row in feedback_result if row[0]}

            # Average duration
            duration_stmt = select(func.avg(EpisodeRecord.duration_ms))
            duration_result = await session.execute(duration_stmt)
            avg_duration = duration_result.scalar() or 0

        return {
            "total_episodes": total,
            "successful": successful,
            "success_rate": successful / total if total > 0 else 0,
            "by_task_type": by_type,
            "by_feedback": by_feedback,
            "avg_duration_ms": avg_duration,
        }

    async def cleanup(self, retention_count: int | None = None) -> int:
        """
        Clean up old episodes beyond retention limit.

        Args:
            retention_count: Maximum episodes to keep

        Returns:
            Number of episodes removed
        """
        if not self._initialized:
            await self.initialize()

        retention_count = retention_count or self.settings.memory.episodic.max_episodes

        from sqlalchemy import delete, func

        async with self._session_factory() as session:
            # Count total
            count_stmt = select(func.count(EpisodeRecord.id))
            count_result = await session.execute(count_stmt)
            total = count_result.scalar() or 0

            if total <= retention_count:
                return 0

            # Get IDs to keep (most recent)
            keep_stmt = (
                select(EpisodeRecord.id)
                .order_by(EpisodeRecord.timestamp.desc())
                .limit(retention_count)
            )
            keep_result = await session.execute(keep_stmt)
            keep_ids = {row[0] for row in keep_result}

            # Delete the rest
            delete_stmt = delete(EpisodeRecord).where(
                EpisodeRecord.id.notin_(keep_ids)
            )
            result = await session.execute(delete_stmt)
            await session.commit()

            return result.rowcount or 0
