"""SMS messaging skill using Twilio."""

from datetime import datetime, timezone
from typing import Any

from ..base import BaseSkill, SkillResult


class SMSSkill(BaseSkill):
    """
    Skill for SMS messaging via Twilio.

    Requires Twilio account credentials.
    """

    name = "sms"
    description = "SMS messaging via Twilio"
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.account_sid = config.get("account_sid", "")
        self.auth_token = config.get("auth_token", "")
        self.from_number = config.get("from_number", "")
        self._client: Any = None

    def _register_capabilities(self) -> None:
        """Register SMS capabilities."""
        self.register_capability(
            name="send",
            description="Send an SMS message",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient phone number (E.164 format)"},
                    "body": {"type": "string", "description": "Message body"},
                    "from_number": {"type": "string", "description": "Sender number (optional)"},
                },
                "required": ["to", "body"],
            },
            security_action="send_messages",
        )

        self.register_capability(
            name="send_bulk",
            description="Send SMS to multiple recipients",
            parameters={
                "type": "object",
                "properties": {
                    "recipients": {"type": "array", "items": {"type": "string"}, "description": "Phone numbers"},
                    "body": {"type": "string", "description": "Message body"},
                },
                "required": ["recipients", "body"],
            },
            security_action="send_messages",
        )

    async def initialize(self) -> None:
        """Initialize Twilio client."""
        if not all([self.account_sid, self.auth_token, self.from_number]):
            self._initialized = False
            return

        try:
            from twilio.rest import Client

            self._client = Client(self.account_sid, self.auth_token)
            self._initialized = True
        except ImportError:
            self._initialized = False

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        """Execute an SMS capability."""
        start_time = datetime.now(timezone.utc)

        if not self._initialized:
            return self._error_result(
                "SMS not configured. Please set Twilio credentials.",
                start_time,
            )

        handlers = {
            "send": self._send,
            "send_bulk": self._send_bulk,
        }

        handler = handlers.get(capability)
        if not handler:
            return self._error_result(f"Unknown capability: {capability}", start_time)

        try:
            result = await handler(**kwargs)
            return self._success_result(result, start_time)
        except Exception as e:
            return self._error_result(str(e), start_time)

    async def _send(
        self,
        to: str,
        body: str,
        from_number: str | None = None,
    ) -> dict[str, Any]:
        """Send an SMS message."""
        import asyncio

        loop = asyncio.get_event_loop()

        # Twilio client is synchronous, run in executor
        message = await loop.run_in_executor(
            None,
            lambda: self._client.messages.create(
                body=body,
                from_=from_number or self.from_number,
                to=to,
            ),
        )

        return {
            "sent": True,
            "sid": message.sid,
            "to": to,
            "status": message.status,
        }

    async def _send_bulk(
        self,
        recipients: list[str],
        body: str,
    ) -> dict[str, Any]:
        """Send SMS to multiple recipients."""
        results = []
        errors = []

        for recipient in recipients:
            try:
                result = await self._send(to=recipient, body=body)
                results.append(result)
            except Exception as e:
                errors.append({"to": recipient, "error": str(e)})

        return {
            "sent_count": len(results),
            "error_count": len(errors),
            "results": results,
            "errors": errors,
        }
