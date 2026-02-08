"""Slack channel implementation using Bolt SDK."""

import asyncio
from typing import Any

from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from ..utils.config import get_settings
from ..utils.logging import get_logger
from .base import Attachment, BaseChannel, Message, MessageType

logger = get_logger(__name__)


class SlackChannel(BaseChannel):
    """
    Slack channel using the Bolt SDK with Socket Mode.

    Features:
    - Real-time messaging via Socket Mode
    - Support for DMs and channel mentions
    - Reaction-based approvals
    - File/image handling
    - Threading support
    """

    def __init__(self) -> None:
        super().__init__("slack")
        self.settings = get_settings()

        # Initialize Slack app
        self.app = AsyncApp(
            token=self.settings.slack_bot_token or self.settings.channels.slack.bot_token,
            # Socket mode doesn't need signing secret
        )

        self.client: AsyncWebClient = self.app.client
        self._handler: AsyncSocketModeHandler | None = None
        self._bot_user_id: str | None = None

        # Pending approval requests
        self._pending_approvals: dict[str, asyncio.Future[dict[str, Any]]] = {}

        # Set up event handlers
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Set up Slack event handlers."""

        @self.app.event("message")
        async def handle_message(event: dict[str, Any], say: Any) -> None:
            """Handle incoming messages."""
            await self._handle_message_event(event)

        @self.app.event("app_mention")
        async def handle_mention(event: dict[str, Any], say: Any) -> None:
            """Handle @mentions of the bot."""
            await self._handle_message_event(event)

        @self.app.event("reaction_added")
        async def handle_reaction(event: dict[str, Any]) -> None:
            """Handle reactions for approval flow."""
            await self._handle_reaction_event(event)

    async def _handle_message_event(self, event: dict[str, Any]) -> None:
        """Process an incoming message event."""
        # Ignore bot messages and message edits
        if event.get("bot_id") or event.get("subtype") in ["message_changed", "message_deleted"]:
            return

        # Ignore messages from ourselves
        if event.get("user") == self._bot_user_id:
            return

        # Build message object
        message = await self._build_message(event)

        logger.info(
            "Received Slack message",
            user=message.user_id,
            channel_id=event.get("channel"),
            content_preview=message.content[:80] if message.content else "",
            handler_count=len(self._message_handlers),
        )

        # Dispatch to handlers
        await self._dispatch_message(message)

    async def _build_message(self, event: dict[str, Any]) -> Message:
        """Build a Message object from a Slack event."""
        user_id = event.get("user", "")
        content = event.get("text", "")
        channel_id = event.get("channel", "")
        thread_ts = event.get("thread_ts")
        message_ts = event.get("ts", "")

        # Get user info
        user_name = None
        try:
            user_info = await self.client.users_info(user=user_id)
            if user_info.get("ok"):
                user_name = user_info["user"].get("real_name") or user_info["user"].get("name")
        except Exception:
            pass

        # Process attachments (files)
        attachments = []
        for file_info in event.get("files", []):
            attachment_type = self._get_attachment_type(file_info.get("mimetype", ""))
            attachments.append(
                Attachment(
                    type=attachment_type,
                    url=file_info.get("url_private"),
                    filename=file_info.get("name"),
                    mime_type=file_info.get("mimetype"),
                    size=file_info.get("size"),
                    metadata={"file_id": file_info.get("id")},
                )
            )

        return Message(
            id=message_ts,
            channel=self.name,
            user_id=user_id,
            user_name=user_name,
            content=content,
            message_type=MessageType.TEXT if not attachments else attachments[0].type,
            attachments=attachments,
            thread_id=thread_ts,
            metadata={
                "channel_id": channel_id,
                "ts": message_ts,
            },
            raw=event,
        )

    def _get_attachment_type(self, mime_type: str) -> MessageType:
        """Map MIME type to attachment type."""
        if mime_type.startswith("image/"):
            return MessageType.IMAGE
        elif mime_type.startswith("audio/"):
            return MessageType.AUDIO
        elif mime_type.startswith("video/"):
            return MessageType.VIDEO
        return MessageType.FILE

    async def _handle_reaction_event(self, event: dict[str, Any]) -> None:
        """Handle reaction events for approval flow."""
        reaction = event.get("reaction", "")
        message_ts = event.get("item", {}).get("ts", "")
        user_id = event.get("user", "")

        # Ignore reactions from the bot itself (e.g. the prompt reactions we add)
        if user_id == self._bot_user_id:
            return

        # Check if this is a pending approval
        if message_ts in self._pending_approvals:
            future = self._pending_approvals[message_ts]

            # Map reactions to approval status
            if reaction in ["white_check_mark", "+1", "thumbsup", "heavy_check_mark"]:
                future.set_result({
                    "approved": True,
                    "approved_by": user_id,
                    "reaction": reaction,
                })
            elif reaction in ["x", "-1", "thumbsdown", "no_entry"]:
                future.set_result({
                    "approved": False,
                    "approved_by": user_id,
                    "reaction": reaction,
                })

    async def start(self) -> None:
        """Start the Slack channel."""
        if self._connected:
            return

        app_token = self.settings.slack_app_token or self.settings.channels.slack.app_token
        if not app_token:
            raise ValueError("Slack app token not configured")

        # Get bot user ID
        try:
            auth_result = await self.client.auth_test()
            self._bot_user_id = auth_result.get("user_id")
            logger.info("Slack bot authenticated", bot_user_id=self._bot_user_id)
        except Exception as e:
            logger.error("Failed to authenticate Slack bot", error=str(e))
            raise

        # Start Socket Mode handler
        self._handler = AsyncSocketModeHandler(self.app, app_token)
        await self._handler.connect_async()

        self._connected = True
        logger.info("Slack channel started")

    async def stop(self) -> None:
        """Stop the Slack channel."""
        if not self._connected:
            return

        if self._handler:
            await self._handler.close_async()
            self._handler = None

        self._connected = False
        logger.info("Slack channel stopped")

    async def send_message(
        self,
        user_id: str,
        content: str,
        reply_to: str | None = None,
        thread_id: str | None = None,
        attachments: list[Attachment] | None = None,
    ) -> str | None:
        """Send a message to a Slack user or channel."""
        try:
            # If user_id looks like a channel/DM ID, use it directly
            # Otherwise, open a DM with the user
            if user_id.startswith(("C", "G", "D")):
                channel = user_id
            else:
                # Open DM channel
                dm_result = await self.client.conversations_open(users=[user_id])
                channel = dm_result["channel"]["id"]

            # Build message kwargs
            kwargs: dict[str, Any] = {
                "channel": channel,
                "text": content,
            }

            if thread_id:
                kwargs["thread_ts"] = thread_id
            elif reply_to:
                kwargs["thread_ts"] = reply_to

            # Send message
            result = await self.client.chat_postMessage(**kwargs)

            # Handle attachments (file uploads)
            if attachments:
                for attachment in attachments:
                    if attachment.data or attachment.path:
                        await self._upload_file(
                            channel=channel,
                            attachment=attachment,
                            thread_ts=result.get("ts"),
                        )

            return result.get("ts")

        except Exception as e:
            logger.error("Failed to send Slack message", error=str(e))
            return None

    async def _upload_file(
        self,
        channel: str,
        attachment: Attachment,
        thread_ts: str | None = None,
    ) -> None:
        """Upload a file to Slack."""
        kwargs: dict[str, Any] = {
            "channels": channel,
            "filename": attachment.filename or "file",
        }

        if thread_ts:
            kwargs["thread_ts"] = thread_ts

        if attachment.data:
            kwargs["content"] = attachment.data
        elif attachment.path:
            kwargs["file"] = attachment.path

        await self.client.files_upload_v2(**kwargs)

    async def send_reaction(
        self,
        message_id: str,
        reaction: str,
        channel_id: str = "",
    ) -> bool:
        """Add a reaction to a message.

        Args:
            message_id: Slack message timestamp (ts)
            reaction: Emoji name without colons (e.g., "eyes", "white_check_mark")
            channel_id: Slack channel ID where the message lives
        """
        if not channel_id:
            logger.warning("send_reaction requires channel_id")
            return False
        try:
            await self.client.reactions_add(
                channel=channel_id,
                timestamp=message_id,
                name=reaction,
            )
            return True
        except Exception as e:
            logger.debug("Slack reaction failed", error=str(e))
            return False

    async def request_approval(
        self,
        user_id: str,
        action_description: str,
        approval_id: str,
        timeout: int = 300,
    ) -> dict[str, Any]:
        """Request approval via Slack message with reaction buttons."""
        # Open DM with user
        try:
            dm_result = await self.client.conversations_open(users=[user_id])
            channel = dm_result["channel"]["id"]
        except Exception as e:
            logger.error("Failed to open DM for approval", error=str(e))
            return {"approved": False, "error": str(e)}

        # Send approval request with blocks
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"🔐 *Approval Required*\n\n{action_description}",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "React with ✅ to approve or ❌ to deny",
                    }
                ],
            },
        ]

        try:
            result = await self.client.chat_postMessage(
                channel=channel,
                text=f"Approval Required: {action_description}",
                blocks=blocks,
            )
            message_ts = result["ts"]

            # Add reaction prompts
            await self.client.reactions_add(
                channel=channel,
                timestamp=message_ts,
                name="white_check_mark",
            )
            await self.client.reactions_add(
                channel=channel,
                timestamp=message_ts,
                name="x",
            )

        except Exception as e:
            logger.error("Failed to send approval request", error=str(e))
            return {"approved": False, "error": str(e)}

        # Wait for reaction
        future: asyncio.Future[dict[str, Any]] = asyncio.Future()
        self._pending_approvals[message_ts] = future

        try:
            result = await asyncio.wait_for(future, timeout=timeout)

            # Update message to show result
            status = "✅ Approved" if result["approved"] else "❌ Denied"
            await self.client.chat_update(
                channel=channel,
                ts=message_ts,
                text=f"{action_description}\n\n{status}",
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"🔐 *Approval Request*\n\n{action_description}\n\n*Status:* {status}",
                        },
                    }
                ],
            )

            return result

        except asyncio.TimeoutError:
            # Update message to show timeout
            await self.client.chat_update(
                channel=channel,
                ts=message_ts,
                text=f"{action_description}\n\n⏱️ Timed out",
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"🔐 *Approval Request*\n\n{action_description}\n\n*Status:* ⏱️ Timed out",
                        },
                    }
                ],
            )

            return {"approved": False, "timeout": True}

        finally:
            self._pending_approvals.pop(message_ts, None)

    async def send_typing_indicator(self, user_id: str) -> None:
        """Slack doesn't have a typing indicator for bots."""
        pass

    async def get_user_info(self, user_id: str) -> dict[str, Any] | None:
        """Get Slack user information."""
        try:
            result = await self.client.users_info(user=user_id)
            if result.get("ok"):
                user = result["user"]
                return {
                    "id": user["id"],
                    "name": user.get("name"),
                    "real_name": user.get("real_name"),
                    "email": user.get("profile", {}).get("email"),
                    "avatar": user.get("profile", {}).get("image_192"),
                    "is_admin": user.get("is_admin", False),
                    "timezone": user.get("tz"),
                }
        except Exception as e:
            logger.error("Failed to get user info", error=str(e))
        return None

    async def download_attachment(self, attachment: Attachment) -> bytes | None:
        """Download a file from Slack."""
        if attachment.data:
            return attachment.data

        if not attachment.url:
            return None

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.settings.slack_bot_token}"}
                async with session.get(attachment.url, headers=headers) as response:
                    if response.status == 200:
                        return await response.read()
        except Exception as e:
            logger.error("Failed to download attachment", error=str(e))

        return None
