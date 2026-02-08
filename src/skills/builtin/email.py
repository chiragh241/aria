"""Email skill using SMTP/IMAP."""

from datetime import datetime, timezone
from typing import Any

from ..base import BaseSkill, SkillResult


class EmailSkill(BaseSkill):
    """
    Skill for sending and reading emails.

    Uses SMTP for sending and IMAP for reading.
    """

    name = "email"
    description = "Email sending and reading"
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.smtp_host = config.get("smtp_host", "")
        self.smtp_port = config.get("smtp_port", 587)
        self.imap_host = config.get("imap_host", "")
        self.imap_port = config.get("imap_port", 993)
        self.username = config.get("username", "")
        self.password = config.get("password", "")

    def _register_capabilities(self) -> None:
        """Register email capabilities."""
        self.register_capability(
            name="send",
            description="Send an email",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "array", "items": {"type": "string"}, "description": "Recipient emails"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                    "html": {"type": "boolean", "default": False, "description": "Send as HTML"},
                    "cc": {"type": "array", "items": {"type": "string"}, "description": "CC recipients"},
                    "bcc": {"type": "array", "items": {"type": "string"}, "description": "BCC recipients"},
                    "attachments": {"type": "array", "items": {"type": "string"}, "description": "File paths to attach"},
                },
                "required": ["to", "subject", "body"],
            },
            security_action="send_emails",
        )

        self.register_capability(
            name="read_inbox",
            description="Read emails from inbox",
            parameters={
                "type": "object",
                "properties": {
                    "folder": {"type": "string", "default": "INBOX"},
                    "limit": {"type": "integer", "default": 10},
                    "unread_only": {"type": "boolean", "default": False},
                },
            },
            security_action="read_files",
        )

        self.register_capability(
            name="search",
            description="Search emails",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "folder": {"type": "string", "default": "INBOX"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
            security_action="read_files",
        )

    async def initialize(self) -> None:
        """Initialize email connections."""
        self._initialized = bool(self.smtp_host and self.username)

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        """Execute an email capability."""
        start_time = datetime.now(timezone.utc)

        if not self._initialized:
            return self._error_result(
                "Email not configured. Please set SMTP/IMAP settings.",
                start_time,
            )

        handlers = {
            "send": self._send,
            "read_inbox": self._read_inbox,
            "search": self._search,
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
        to: list[str],
        subject: str,
        body: str,
        html: bool = False,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        attachments: list[str] | None = None,
    ) -> dict[str, Any]:
        """Send an email."""
        import asyncio
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders
        from pathlib import Path

        import aiosmtplib

        msg = MIMEMultipart()
        msg["From"] = self.username
        msg["To"] = ", ".join(to)
        msg["Subject"] = subject

        if cc:
            msg["Cc"] = ", ".join(cc)

        # Add body
        content_type = "html" if html else "plain"
        msg.attach(MIMEText(body, content_type))

        # Add attachments
        if attachments:
            for file_path in attachments:
                path = Path(file_path).expanduser()
                if path.exists():
                    with open(path, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            "Content-Disposition",
                            f"attachment; filename={path.name}",
                        )
                        msg.attach(part)

        # Send
        all_recipients = to + (cc or []) + (bcc or [])

        await aiosmtplib.send(
            msg,
            hostname=self.smtp_host,
            port=self.smtp_port,
            username=self.username,
            password=self.password,
            start_tls=True,
        )

        return {
            "sent": True,
            "to": to,
            "subject": subject,
            "recipients_count": len(all_recipients),
        }

    async def _read_inbox(
        self,
        folder: str = "INBOX",
        limit: int = 10,
        unread_only: bool = False,
    ) -> dict[str, Any]:
        """Read emails from inbox."""
        import aioimaplib

        client = aioimaplib.IMAP4_SSL(self.imap_host, self.imap_port)
        await client.wait_hello_from_server()
        await client.login(self.username, self.password)
        await client.select(folder)

        # Search for emails
        criteria = "UNSEEN" if unread_only else "ALL"
        _, data = await client.search(criteria)

        email_ids = data[0].split()[-limit:]
        emails = []

        for email_id in reversed(email_ids):
            _, msg_data = await client.fetch(email_id.decode(), "(RFC822)")
            if msg_data:
                import email

                msg = email.message_from_bytes(msg_data[1])
                emails.append({
                    "id": email_id.decode(),
                    "from": msg.get("From"),
                    "to": msg.get("To"),
                    "subject": msg.get("Subject"),
                    "date": msg.get("Date"),
                })

        await client.logout()

        return {
            "folder": folder,
            "count": len(emails),
            "emails": emails,
        }

    async def _search(
        self,
        query: str,
        folder: str = "INBOX",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search emails."""
        # Simplified search - would need proper IMAP search implementation
        return {
            "query": query,
            "folder": folder,
            "results": [],
            "message": "Search requires full IMAP implementation",
        }
