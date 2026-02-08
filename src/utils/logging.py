"""Structured logging setup for Aria."""

import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from structlog.types import Processor

from .config import get_settings


class _TeeLoggerFactory:
    """Logger factory that writes to both stdout and a log file."""

    def __init__(self, file_path: Path) -> None:
        self._file = open(file_path, "a", buffering=1)  # line-buffered

    def __call__(self, *args: Any, **kwargs: Any) -> "_TeeLogger":
        return _TeeLogger(self._file)


class _TeeLogger:
    """Logger that writes each message to stdout and a file."""

    def __init__(self, file: Any) -> None:
        self._file = file

    def msg(self, message: str) -> None:
        print(message, flush=True)
        self._file.write(message + "\n")

    log = debug = info = warn = warning = msg
    fatal = failure = err = error = critical = msg


def setup_logging(level: str | None = None, format_type: str | None = None) -> None:
    """
    Set up structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Output format ('json' or 'text')
    """
    settings = get_settings()
    log_level = level or settings.logging.level
    log_format = format_type or settings.logging.format

    # Ensure log directory exists
    log_file = Path(settings.logging.file).expanduser()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure standard library logging (for third-party libs like httpx, uvicorn)
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=settings.logging.max_size_mb * 1024 * 1024,
                backupCount=settings.logging.backup_count,
            ),
        ],
    )

    # Shared processors for both formats
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        # JSON format for production
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Human-readable format for development (no ANSI colors in file)
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=False),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=_TeeLoggerFactory(log_file),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__ of the calling module)

    Returns:
        Configured structured logger
    """
    return structlog.get_logger(name)


class AuditLogger:
    """
    Separate audit logger for security-relevant events.

    This logs to a separate file and always uses JSON format
    for easy parsing and analysis.
    """

    _db_logger: Any = None  # Database-backed audit logger bridge

    @classmethod
    def register_db_logger(cls, db_logger: Any) -> None:
        """Register a database audit logger to receive copies of all events."""
        cls._db_logger = db_logger

    def __init__(self) -> None:
        """Initialize the audit logger."""
        settings = get_settings()
        audit_file = Path(settings.logging.audit_file).expanduser()
        audit_file.parent.mkdir(parents=True, exist_ok=True)

        # Create a separate logger for audit events
        self._logger = logging.getLogger("aria.audit")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False  # Don't propagate to root logger

        # File handler for audit log
        handler = logging.handlers.RotatingFileHandler(
            audit_file,
            maxBytes=settings.logging.max_size_mb * 1024 * 1024,
            backupCount=settings.logging.backup_count,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

    def log(
        self,
        event: str,
        action_type: str,
        user_id: str | None = None,
        channel: str | None = None,
        details: dict[str, Any] | None = None,
        status: str = "info",
        **kwargs: Any,
    ) -> None:
        """
        Log an audit event.

        Args:
            event: Event name (e.g., "action_requested", "action_approved")
            action_type: Type of action (e.g., "read_files", "shell_commands")
            user_id: ID of the user who triggered the action
            channel: Channel where the action originated
            details: Additional event details
            status: Event status (info, warning, error)
            **kwargs: Additional fields to include
        """
        import orjson

        timestamp = datetime.now(timezone.utc).isoformat()

        log_entry = {
            "timestamp": timestamp,
            "event": event,
            "action_type": action_type,
            "user_id": user_id,
            "channel": channel,
            "status": status,
            "details": details or {},
            **kwargs,
        }

        self._logger.info(orjson.dumps(log_entry).decode())

        # Also write to database logger if registered
        if self.__class__._db_logger is not None:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self.__class__._db_logger.log(
                        event=event,
                        action_type=action_type,
                        user_id=user_id,
                        channel=channel,
                        status=status,
                        details=details,
                    )
                )
            except RuntimeError:
                pass  # No running event loop

    def action_requested(
        self,
        action_type: str,
        description: str,
        user_id: str | None = None,
        channel: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log an action request."""
        self.log(
            event="action_requested",
            action_type=action_type,
            user_id=user_id,
            channel=channel,
            details={"description": description},
            **kwargs,
        )

    def action_approved(
        self,
        action_type: str,
        approved_by: str,
        channel: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log an action approval."""
        self.log(
            event="action_approved",
            action_type=action_type,
            user_id=approved_by,
            channel=channel,
            status="info",
            **kwargs,
        )

    def action_denied(
        self,
        action_type: str,
        denied_by: str | None = None,
        reason: str | None = None,
        channel: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log an action denial."""
        self.log(
            event="action_denied",
            action_type=action_type,
            user_id=denied_by,
            channel=channel,
            status="warning",
            details={"reason": reason},
            **kwargs,
        )

    def action_executed(
        self,
        action_type: str,
        result: str,
        execution_time_ms: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log an action execution."""
        self.log(
            event="action_executed",
            action_type=action_type,
            status="info",
            details={"result": result, "execution_time_ms": execution_time_ms},
            **kwargs,
        )

    def action_failed(
        self,
        action_type: str,
        error: str,
        **kwargs: Any,
    ) -> None:
        """Log an action failure."""
        self.log(
            event="action_failed",
            action_type=action_type,
            status="error",
            details={"error": error},
            **kwargs,
        )

    def security_violation(
        self,
        action_type: str,
        violation_type: str,
        details: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log a security violation attempt."""
        self.log(
            event="security_violation",
            action_type=action_type,
            status="error",
            details={"violation_type": violation_type, **(details or {})},
            **kwargs,
        )


# Global audit logger instance
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
