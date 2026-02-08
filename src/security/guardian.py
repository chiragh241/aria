"""Security Guardian - the approval engine for Aria."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..utils.config import get_settings
from ..utils.logging import get_audit_logger, get_logger
from .profiles import ActionResult, ProfileManager

if TYPE_CHECKING:
    from ..channels.base import BaseChannel

logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class ApprovalRequest:
    """A pending approval request."""

    id: str = field(default_factory=lambda: str(uuid4()))
    action_type: str = ""
    description: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    user_id: str = ""
    channel: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    timeout: int = 300
    status: str = "pending"  # pending, approved, denied, timeout
    approved_by: str | None = None
    responded_at: datetime | None = None


@dataclass
class ApprovalResult:
    """Result of an action check."""

    approved: bool = False
    requires_approval: bool = False
    auto_approved: bool = False
    description: str = ""
    reason: str = ""
    request_id: str | None = None


class SecurityGuardian:
    """
    The Security Guardian manages all action approval workflows.

    Responsibilities:
    - Check actions against security profiles
    - Request approval from users across channels
    - Track and timeout pending approvals
    - Log all security-relevant events
    """

    def __init__(
        self,
        profile_manager: ProfileManager | None = None,
    ) -> None:
        self.settings = get_settings()
        self.profile_manager = profile_manager or ProfileManager()

        # Approval tracking
        self._pending_requests: dict[str, ApprovalRequest] = {}
        self._approval_futures: dict[str, asyncio.Future[ApprovalResult]] = {}

        # Channel references for sending approval requests
        self._channels: dict[str, "BaseChannel"] = {}

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the security guardian."""
        logger.info("Security guardian initialized", profile=self.profile_manager.active_profile)

    def register_channel(self, name: str, channel: "BaseChannel") -> None:
        """Register a channel for sending approval requests."""
        self._channels[name] = channel

    async def check_action(
        self,
        action_type: str,
        details: dict[str, Any],
        user_id: str,
        channel: str,
    ) -> ApprovalResult:
        """
        Check if an action is allowed and request approval if needed.

        Args:
            action_type: Type of action (e.g., "read_files", "shell_commands")
            details: Action details for context
            user_id: User who initiated the action
            channel: Channel where the action was initiated

        Returns:
            ApprovalResult indicating whether to proceed
        """
        # Extract relevant info for checking
        path = details.get("path") or details.get("arguments", {}).get("path")
        domain = details.get("domain") or details.get("url", "").split("/")[2] if "://" in details.get("url", "") else None
        command = details.get("command") or details.get("arguments", {}).get("command")

        # Check against profile
        result, reason = self.profile_manager.check_action(
            action_type=action_type,
            path=path,
            domain=domain,
            command=command,
        )

        # Build description
        tool_name = details.get("tool", action_type)
        description = self._build_description(action_type, details)

        # Log the check
        audit_logger.action_requested(
            action_type=action_type,
            description=description,
            user_id=user_id,
            channel=channel,
        )

        # Handle based on result
        if result == ActionResult.DENY:
            audit_logger.action_denied(
                action_type=action_type,
                reason=reason,
                channel=channel,
            )
            return ApprovalResult(
                approved=False,
                requires_approval=False,
                description=description,
                reason=reason,
            )

        if result == ActionResult.AUTO:
            return ApprovalResult(
                approved=True,
                requires_approval=False,
                auto_approved=True,
                description=description,
                reason=reason,
            )

        if result == ActionResult.NOTIFY:
            # Auto-approve but notify user
            asyncio.create_task(self._notify_user(user_id, action_type, description))
            return ApprovalResult(
                approved=True,
                requires_approval=False,
                auto_approved=True,
                description=description,
                reason="Auto-approved with notification",
            )

        # APPROVE - need explicit approval
        return ApprovalResult(
            approved=False,
            requires_approval=True,
            description=description,
            reason=reason,
        )

    async def request_approval(
        self,
        action_type: str,
        description: str,
        details: dict[str, Any],
        user_id: str,
        channel: str,
        timeout: int | None = None,
    ) -> ApprovalResult:
        """
        Request explicit approval for an action.

        Args:
            action_type: Type of action
            description: Human-readable description
            details: Action details
            user_id: User to request approval from
            channel: Preferred channel for the request
            timeout: Override default timeout

        Returns:
            ApprovalResult with the decision
        """
        timeout = timeout or self.settings.security.approval_timeout

        # Create request
        request = ApprovalRequest(
            action_type=action_type,
            description=description,
            details=details,
            user_id=user_id,
            channel=channel,
            timeout=timeout,
        )

        # Store request
        async with self._lock:
            self._pending_requests[request.id] = request
            future: asyncio.Future[ApprovalResult] = asyncio.Future()
            self._approval_futures[request.id] = future

        logger.info(
            "Requesting approval",
            request_id=request.id,
            action_type=action_type,
            user_id=user_id,
        )

        # Send approval requests to configured channels
        await self._send_approval_requests(request)

        # Wait for response
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            # Timeout - deny by default
            async with self._lock:
                request.status = "timeout"
                self._pending_requests.pop(request.id, None)
                self._approval_futures.pop(request.id, None)

            audit_logger.action_denied(
                action_type=action_type,
                reason="Approval timeout",
                channel=channel,
            )

            return ApprovalResult(
                approved=False,
                requires_approval=True,
                description=description,
                reason="Approval request timed out",
                request_id=request.id,
            )

    async def _send_approval_requests(self, request: ApprovalRequest) -> None:
        """Send approval request to the originating channel.

        Only sends to the channel that initiated the action, because
        user IDs are channel-specific (a Slack user ID won't work on
        WhatsApp and vice versa).  The web dashboard always shows all
        pending requests via the REST API, so it doesn't need a
        channel-based approval message.
        """
        # Send to the originating channel (where the user is active)
        origin = request.channel
        channel = self._channels.get(origin)
        if channel and channel.is_connected:
            try:
                asyncio.create_task(
                    self._handle_channel_approval(
                        channel=channel,
                        channel_name=origin,
                        request=request,
                    )
                )
            except Exception as e:
                logger.error(
                    "Failed to send approval request to channel",
                    channel=origin,
                    error=str(e),
                )
        else:
            logger.warning(
                "Originating channel not available for approval",
                channel=origin,
                request_id=request.id,
            )

    async def _handle_channel_approval(
        self,
        channel: "BaseChannel",
        channel_name: str,
        request: ApprovalRequest,
    ) -> None:
        """Handle approval request for a specific channel."""
        try:
            result = await channel.request_approval(
                user_id=request.user_id,
                action_description=request.description,
                approval_id=request.id,
                timeout=request.timeout,
            )

            # Process result
            await self.handle_approval_response(
                request_id=request.id,
                approved=result.get("approved", False),
                approved_by=result.get("approved_by", ""),
                channel=channel_name,
            )
        except Exception as e:
            logger.error(
                "Channel approval request failed",
                channel=channel_name,
                error=str(e),
            )

    async def handle_approval_response(
        self,
        request_id: str,
        approved: bool,
        approved_by: str,
        channel: str,
    ) -> bool:
        """
        Handle an approval response from a channel.

        Args:
            request_id: The approval request ID
            approved: Whether the action was approved
            approved_by: Who approved/denied
            channel: Channel the response came from

        Returns:
            True if the response was processed
        """
        async with self._lock:
            request = self._pending_requests.get(request_id)
            future = self._approval_futures.get(request_id)

            if not request or not future or future.done():
                return False

            # Update request
            request.status = "approved" if approved else "denied"
            request.approved_by = approved_by
            request.responded_at = datetime.now(timezone.utc)

            # Log
            if approved:
                audit_logger.action_approved(
                    action_type=request.action_type,
                    approved_by=approved_by,
                    channel=channel,
                )
            else:
                audit_logger.action_denied(
                    action_type=request.action_type,
                    denied_by=approved_by,
                    channel=channel,
                )

            # Complete the future
            result = ApprovalResult(
                approved=approved,
                requires_approval=True,
                description=request.description,
                reason=f"{'Approved' if approved else 'Denied'} by {approved_by}",
                request_id=request_id,
            )
            future.set_result(result)

            # Clean up
            self._pending_requests.pop(request_id, None)
            self._approval_futures.pop(request_id, None)

        return True

    async def _notify_user(
        self,
        user_id: str,
        action_type: str,
        description: str,
    ) -> None:
        """Send a notification to the user about an auto-approved action."""
        for channel_name, channel in self._channels.items():
            if channel.is_connected:
                try:
                    await channel.send_message(
                        user_id=user_id,
                        content=f"📋 **Action Notification**\n\n{description}\n\n_This action was auto-approved by your security profile._",
                    )
                    return  # Only notify on one channel
                except Exception as e:
                    logger.warning(
                        "Failed to send notification",
                        channel=channel_name,
                        error=str(e),
                    )

    def _build_description(
        self,
        action_type: str,
        details: dict[str, Any],
    ) -> str:
        """Build a human-readable description of an action."""
        tool = details.get("tool", "")
        args = details.get("arguments", {})

        if action_type == "read_files":
            path = args.get("path", details.get("path", "unknown"))
            return f"Read file: `{path}`"

        elif action_type == "write_files":
            path = args.get("path", details.get("path", "unknown"))
            return f"Write to file: `{path}`"

        elif action_type == "delete_files":
            path = args.get("path", details.get("path", "unknown"))
            return f"Delete file: `{path}`"

        elif action_type == "shell_commands":
            command = args.get("command", details.get("command", "unknown"))
            return f"Execute command: `{command}`"

        elif action_type == "web_requests":
            url = args.get("url", details.get("url", "unknown"))
            return f"Web request to: `{url}`"

        elif action_type == "send_emails":
            to = args.get("to", details.get("to", "unknown"))
            subject = args.get("subject", "")
            return f"Send email to: `{to}` - Subject: {subject}"

        elif action_type == "send_messages":
            to = args.get("to", details.get("to", "unknown"))
            return f"Send message to: `{to}`"

        elif action_type in ("calendar_read", "calendar_write"):
            return f"Calendar operation: {tool}"

        elif action_type == "code_edit":
            edits = details.get("edits", [])
            if edits:
                fps = ", ".join(e.get("file_path", "?") for e in edits if isinstance(e, dict))
            else:
                fps = details.get("file_path", "unknown")
            reason = details.get("reason", "")
            return f"Self-healing code fix: {fps}" + (f" — {reason}" if reason else "")

        return f"Action: {action_type} - {tool}"

    def get_pending_requests(self, user_id: str | None = None) -> list[ApprovalRequest]:
        """Get all pending approval requests, optionally filtered by user."""
        requests = list(self._pending_requests.values())
        if user_id:
            requests = [r for r in requests if r.user_id == user_id]
        return requests

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending approval request."""
        if request_id in self._pending_requests:
            request = self._pending_requests.pop(request_id)
            future = self._approval_futures.pop(request_id, None)

            request.status = "cancelled"

            if future and not future.done():
                future.set_result(
                    ApprovalResult(
                        approved=False,
                        requires_approval=True,
                        description=request.description,
                        reason="Request cancelled",
                        request_id=request_id,
                    )
                )

            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get guardian statistics."""
        return {
            "pending_requests": len(self._pending_requests),
            "active_profile": self.profile_manager.active_profile.name,
            "available_profiles": [p["name"] for p in self.profile_manager.list_profiles()],
        }
