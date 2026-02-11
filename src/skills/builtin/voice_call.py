"""Voice call skill (Twilio)."""

from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from ..base import BaseSkill, SkillResult


class VoiceCallSkill(BaseSkill):
    """
    Initiate and control voice calls via Twilio (or compatible). Requires a public TwiML URL
    (e.g. Aria's /api/voice/twiml) so Twilio can fetch speech content when the call connects.
    """

    name = "voice_call"
    description = "Place voice calls via Twilio and speak a message (TTS)"
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        c = config or {}
        twilio = c.get("twilio", c)
        self.account_sid = twilio.get("account_sid", "") or c.get("account_sid", "")
        self.auth_token = twilio.get("auth_token", "") or c.get("auth_token", "")
        self.from_number = twilio.get("from_number", "") or c.get("from_number", "")
        self.twiml_base_url = (twilio.get("twiml_base_url", "") or c.get("twiml_base_url", "")).rstrip("/")

    def _register_capabilities(self) -> None:
        """Register voice call capabilities."""
        self.register_capability(
            name="initiate_call",
            description="Place an outbound voice call and speak a message (TTS). Requires Twilio and a public TwiML URL.",
            parameters={
                "type": "object",
                "properties": {
                    "to_number": {"type": "string", "description": "E.164 phone number to call"},
                    "message": {"type": "string", "description": "Message to speak when the call is answered"},
                },
                "required": ["to_number", "message"],
            },
        )

    async def initialize(self) -> None:
        """Check config."""
        self._initialized = bool(
            self.account_sid and self.auth_token and self.from_number and self.twiml_base_url
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        """Execute voice call capability."""
        start_time = datetime.now(timezone.utc)
        if not self._initialized:
            return self._error_result(
                "Voice call not configured. Set account_sid, auth_token, from_number, and twiml_base_url (public URL to /api/voice/twiml).",
                start_time,
            )
        if capability == "initiate_call":
            return await self._initiate_call(start_time, **kwargs)
        return self._error_result(f"Unknown capability: {capability}", start_time)

    async def _initiate_call(
        self,
        start_time: datetime,
        to_number: str,
        message: str,
    ) -> SkillResult:
        """Create an outbound Twilio call; Twilio will GET twiml_url and speak the message."""
        try:
            import httpx
        except ImportError:
            return self._error_result("httpx required for voice calls", start_time)

        url_twiml = f"{self.twiml_base_url}?message={quote(message)}"
        api_url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Calls.json"
        payload = {
            "From": self.from_number,
            "To": to_number,
            "Url": url_twiml,
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    api_url,
                    auth=(self.account_sid, self.auth_token),
                    data=payload,
                    timeout=15.0,
                )
        except Exception as e:
            return self._error_result(f"Twilio request failed: {e}", start_time)

        if resp.status_code >= 400:
            return self._error_result(
                f"Twilio error {resp.status_code}: {resp.text[:300]}",
                start_time,
            )
        data = resp.json()
        call_sid = data.get("sid")
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return SkillResult(
            success=True,
            output={"call_sid": call_sid, "status": data.get("status"), "to": to_number},
            execution_time_ms=elapsed,
        )

    def _error_result(self, error: str, start_time: datetime) -> SkillResult:
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return SkillResult(success=False, error=error, execution_time_ms=elapsed)
