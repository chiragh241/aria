"""Base skill interface for Aria."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class SkillResult:
    """Result of a skill execution."""

    success: bool = False
    output: Any = None
    error: str | None = None
    execution_time_ms: float = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class SkillCapability:
    """A capability provided by a skill."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    returns: dict[str, Any] = field(default_factory=dict)
    examples: list[dict[str, Any]] = field(default_factory=list)
    security_action: str | None = None  # Maps to security action type


class BaseSkill(ABC):
    """
    Abstract base class for all skills.

    Skills are modular capabilities that Aria can use to perform actions.
    Each skill can have multiple capabilities (functions).
    """

    # Skill metadata - override in subclasses
    name: str = "base_skill"
    description: str = "Base skill interface"
    version: str = "1.0.0"
    enabled: bool = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize the skill.

        Args:
            config: Skill-specific configuration
        """
        self.config = config or {}
        self._capabilities: dict[str, SkillCapability] = {}
        self._initialized = False

        # Register capabilities
        self._register_capabilities()

    @abstractmethod
    def _register_capabilities(self) -> None:
        """Register all capabilities provided by this skill."""
        pass

    def register_capability(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        returns: dict[str, Any] | None = None,
        examples: list[dict[str, Any]] | None = None,
        security_action: str | None = None,
    ) -> None:
        """
        Register a capability.

        Args:
            name: Capability name
            description: Description for LLM
            parameters: JSON schema for parameters
            returns: JSON schema for return value
            examples: Usage examples
            security_action: Security action type for approval
        """
        self._capabilities[name] = SkillCapability(
            name=name,
            description=description,
            parameters=parameters or {},
            returns=returns or {},
            examples=examples or [],
            security_action=security_action,
        )

    def get_capabilities(self) -> list[dict[str, Any]]:
        """Get all capabilities as LLM-friendly dictionaries."""
        return [
            {
                "name": cap.name,
                "description": cap.description,
                "parameters": cap.parameters,
                "returns": cap.returns,
                "examples": cap.examples,
            }
            for cap in self._capabilities.values()
        ]

    def get_capability(self, name: str) -> SkillCapability | None:
        """Get a specific capability."""
        return self._capabilities.get(name)

    async def initialize(self) -> None:
        """
        Initialize the skill asynchronously.

        Override in subclasses for async initialization.
        """
        self._initialized = True

    async def shutdown(self) -> None:
        """
        Shutdown the skill and clean up resources.

        Override in subclasses for cleanup.
        """
        self._initialized = False

    @abstractmethod
    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        """
        Execute a capability.

        Args:
            capability: Name of the capability to execute
            **kwargs: Capability-specific parameters

        Returns:
            SkillResult with execution outcome
        """
        pass

    def _measure_execution(self, start_time: datetime) -> float:
        """Calculate execution time in milliseconds."""
        return (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

    def _error_result(
        self,
        error: str,
        start_time: datetime | None = None,
    ) -> SkillResult:
        """Create an error result."""
        return SkillResult(
            success=False,
            error=error,
            execution_time_ms=self._measure_execution(start_time) if start_time else 0,
        )

    def _success_result(
        self,
        output: Any,
        start_time: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SkillResult:
        """Create a success result."""
        return SkillResult(
            success=True,
            output=output,
            execution_time_ms=self._measure_execution(start_time) if start_time else 0,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert skill metadata to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "enabled": self.enabled,
            "capabilities": self.get_capabilities(),
            "initialized": self._initialized,
        }
