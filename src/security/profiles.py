"""Security profile management."""

import fnmatch
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ActionResult(str, Enum):
    """Result of an action check."""

    AUTO = "auto"  # Automatically allow
    NOTIFY = "notify"  # Allow but notify user
    APPROVE = "approve"  # Require explicit approval
    DENY = "deny"  # Always deny


class RiskLevel(str, Enum):
    """Risk levels for actions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ActionRule:
    """Rule for a specific action type."""

    action: ActionResult = ActionResult.APPROVE
    description: str = ""
    allowed_paths: list[str] = field(default_factory=list)
    denied_paths: list[str] = field(default_factory=list)
    allowed_domains: list[str] = field(default_factory=list)
    denied_domains: list[str] = field(default_factory=list)
    safe_commands: list[str] = field(default_factory=list)
    denied_commands: list[str] = field(default_factory=list)

    def check_path(self, path: str) -> bool | None:
        """
        Check if a path is allowed by this rule.

        Returns:
            True if explicitly allowed
            False if explicitly denied
            None if no specific rule applies
        """
        # Expand path
        expanded_path = str(Path(path).expanduser().resolve())

        # Check denied paths first
        for pattern in self.denied_paths:
            expanded_pattern = str(Path(pattern).expanduser())
            if fnmatch.fnmatch(expanded_path, expanded_pattern) or expanded_path.startswith(
                expanded_pattern.rstrip("*")
            ):
                return False

        # Check allowed paths
        for pattern in self.allowed_paths:
            expanded_pattern = str(Path(pattern).expanduser())
            if fnmatch.fnmatch(expanded_path, expanded_pattern) or expanded_path.startswith(
                expanded_pattern.rstrip("*")
            ):
                return True

        return None

    def check_domain(self, domain: str) -> bool | None:
        """
        Check if a domain is allowed by this rule.

        Returns:
            True if explicitly allowed
            False if explicitly denied
            None if no specific rule applies
        """
        # Check denied domains first
        for pattern in self.denied_domains:
            if fnmatch.fnmatch(domain, pattern):
                return False

        # Check allowed domains
        for pattern in self.allowed_domains:
            if fnmatch.fnmatch(domain, pattern):
                return True

        return None

    def check_command(self, command: str) -> bool | None:
        """
        Check if a command is allowed by this rule.

        Returns:
            True if explicitly allowed (safe)
            False if explicitly denied
            None if no specific rule applies
        """
        # Get the base command (first word)
        base_command = command.strip().split()[0] if command.strip() else ""

        # Check denied commands first
        for denied in self.denied_commands:
            if denied in command:
                return False

        # Check safe commands
        if base_command in self.safe_commands:
            return True

        return None


@dataclass
class SecurityProfile:
    """A complete security profile."""

    name: str
    description: str = ""
    actions: dict[str, ActionRule] = field(default_factory=dict)

    def get_rule(self, action_type: str) -> ActionRule:
        """Get the rule for an action type, with fallback to default."""
        return self.actions.get(action_type, ActionRule(action=ActionResult.APPROVE))


class ProfileManager:
    """
    Manages security profiles for the application.

    Loads profiles from YAML configuration and provides
    methods to check actions against the active profile.
    """

    def __init__(self, profiles_file: str | Path | None = None) -> None:
        self.settings = get_settings()
        self._profiles: dict[str, SecurityProfile] = {}
        self._action_types: dict[str, dict[str, Any]] = {}
        self._active_profile_name: str = self.settings.security.active_profile

        # Load profiles
        if profiles_file:
            self._load_from_file(profiles_file)
        else:
            self._load_default_profiles()

    def _load_from_file(self, path: str | Path) -> None:
        """Load profiles from a YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning("Security profiles file not found", path=str(path))
            self._load_default_profiles()
            return

        with open(path) as f:
            data = yaml.safe_load(f)

        # Load action types metadata
        self._action_types = data.get("action_types", {})

        # Load profiles
        for name, profile_data in data.get("profiles", {}).items():
            actions = {}
            for action_name, action_data in profile_data.get("actions", {}).items():
                if isinstance(action_data, str):
                    actions[action_name] = ActionRule(
                        action=ActionResult(action_data),
                    )
                else:
                    actions[action_name] = ActionRule(
                        action=ActionResult(action_data.get("action", "approve")),
                        description=action_data.get("description", ""),
                        allowed_paths=action_data.get("allowed_paths", []),
                        denied_paths=action_data.get("denied_paths", []),
                        allowed_domains=action_data.get("allowed_domains", []),
                        denied_domains=action_data.get("denied_domains", []),
                        safe_commands=action_data.get("safe_commands", []),
                        denied_commands=action_data.get("denied_commands", []),
                    )

            self._profiles[name] = SecurityProfile(
                name=name,
                description=profile_data.get("description", ""),
                actions=actions,
            )

        logger.info("Loaded security profiles", count=len(self._profiles))

    def _load_default_profiles(self) -> None:
        """Load default built-in profiles."""
        # Paranoid profile
        self._profiles["paranoid"] = SecurityProfile(
            name="paranoid",
            description="Maximum security - all actions require approval",
            actions={
                action: ActionRule(action=ActionResult.APPROVE)
                for action in [
                    "read_files",
                    "write_files",
                    "delete_files",
                    "shell_commands",
                    "send_messages",
                    "send_emails",
                    "web_requests",
                    "calendar_read",
                    "calendar_write",
                    "skill_creation",
                ]
            },
        )

        # Balanced profile
        self._profiles["balanced"] = SecurityProfile(
            name="balanced",
            description="Balanced security for everyday use",
            actions={
                "read_files": ActionRule(
                    action=ActionResult.AUTO,
                    allowed_paths=["~/Documents", "~/Projects", "~/Downloads"],
                    denied_paths=["~/.ssh", "~/.gnupg", "~/.aws", "~/.config"],
                ),
                "write_files": ActionRule(
                    action=ActionResult.NOTIFY,
                    allowed_paths=["~/Documents", "~/Projects"],
                    denied_paths=["~/.ssh", "~/.gnupg", "/etc", "/usr"],
                ),
                "delete_files": ActionRule(action=ActionResult.APPROVE),
                "shell_commands": ActionRule(
                    action=ActionResult.APPROVE,
                    safe_commands=["ls", "pwd", "whoami", "date", "cat", "head", "tail", "grep", "find", "wc", "echo"],
                ),
                "send_messages": ActionRule(action=ActionResult.APPROVE),
                "send_emails": ActionRule(action=ActionResult.APPROVE),
                "web_requests": ActionRule(
                    action=ActionResult.AUTO,
                    allowed_domains=["*.google.com", "*.github.com", "*.stackoverflow.com", "*.wikipedia.org"],
                    denied_domains=["*.onion"],
                ),
                "calendar_read": ActionRule(action=ActionResult.AUTO),
                "calendar_write": ActionRule(action=ActionResult.NOTIFY),
                "skill_creation": ActionRule(action=ActionResult.APPROVE),
            },
        )

        # Trusted profile
        self._profiles["trusted"] = SecurityProfile(
            name="trusted",
            description="Minimal security for trusted environments",
            actions={
                "read_files": ActionRule(
                    action=ActionResult.AUTO,
                    denied_paths=["~/.ssh/id_*", "~/.gnupg/private*"],
                ),
                "write_files": ActionRule(
                    action=ActionResult.AUTO,
                    denied_paths=["/etc", "/usr", "/bin", "/sbin"],
                ),
                "delete_files": ActionRule(action=ActionResult.NOTIFY),
                "shell_commands": ActionRule(
                    action=ActionResult.NOTIFY,
                    denied_commands=["rm -rf /", "dd if=/dev/zero", "> /dev/sda", "mkfs", "fdisk"],
                ),
                "send_messages": ActionRule(action=ActionResult.NOTIFY),
                "send_emails": ActionRule(action=ActionResult.NOTIFY),
                "web_requests": ActionRule(action=ActionResult.AUTO),
                "calendar_read": ActionRule(action=ActionResult.AUTO),
                "calendar_write": ActionRule(action=ActionResult.AUTO),
                "skill_creation": ActionRule(action=ActionResult.NOTIFY),
            },
        )

        logger.info("Loaded default security profiles", count=len(self._profiles))

    @property
    def active_profile(self) -> SecurityProfile:
        """Get the currently active profile."""
        return self._profiles.get(
            self._active_profile_name,
            self._profiles.get("balanced", list(self._profiles.values())[0]),
        )

    def set_active_profile(self, profile_name: str) -> bool:
        """
        Set the active security profile.

        Args:
            profile_name: Name of the profile to activate

        Returns:
            True if successful, False if profile not found
        """
        if profile_name not in self._profiles:
            logger.warning("Profile not found", profile=profile_name)
            return False

        self._active_profile_name = profile_name
        logger.info("Active security profile changed", profile=profile_name)
        return True

    def list_profiles(self) -> list[dict[str, Any]]:
        """List all available profiles."""
        return [
            {
                "name": name,
                "description": profile.description,
                "active": name == self._active_profile_name,
            }
            for name, profile in self._profiles.items()
        ]

    def get_profile(self, name: str) -> SecurityProfile | None:
        """Get a specific profile by name."""
        return self._profiles.get(name)

    def get_action_metadata(self, action_type: str) -> dict[str, Any]:
        """Get metadata about an action type."""
        return self._action_types.get(action_type, {
            "risk_level": "high",
            "category": "unknown",
            "description": action_type,
        })

    def check_action(
        self,
        action_type: str,
        path: str | None = None,
        domain: str | None = None,
        command: str | None = None,
    ) -> tuple[ActionResult, str]:
        """
        Check an action against the active profile.

        Args:
            action_type: Type of action to check
            path: File path (for filesystem actions)
            domain: Domain (for web requests)
            command: Command (for shell actions)

        Returns:
            Tuple of (action_result, reason)
        """
        rule = self.active_profile.get_rule(action_type)

        # Check specific conditions
        if path:
            path_result = rule.check_path(path)
            if path_result is False:
                return ActionResult.DENY, f"Path '{path}' is denied by security policy"
            if path_result is True and rule.action == ActionResult.APPROVE:
                return ActionResult.AUTO, f"Path '{path}' is in allowed paths"

        if domain:
            domain_result = rule.check_domain(domain)
            if domain_result is False:
                return ActionResult.DENY, f"Domain '{domain}' is denied by security policy"
            if domain_result is True and rule.action == ActionResult.APPROVE:
                return ActionResult.AUTO, f"Domain '{domain}' is in allowed domains"

        if command:
            command_result = rule.check_command(command)
            if command_result is False:
                return ActionResult.DENY, f"Command is denied by security policy"
            if command_result is True and rule.action == ActionResult.APPROVE:
                return ActionResult.AUTO, f"Command is in safe commands list"

        return rule.action, rule.description or f"Action '{action_type}' requires {rule.action.value}"
