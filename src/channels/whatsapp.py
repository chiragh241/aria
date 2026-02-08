"""WhatsApp channel using whatsapp-web.js Node.js bridge."""

import asyncio
import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

from ..utils.config import get_settings
from ..utils.logging import get_logger
from .base import Attachment, BaseChannel, Message, MessageType

logger = get_logger(__name__)


class WhatsAppChannel(BaseChannel):
    """
    WhatsApp channel using whatsapp-web.js via a Node.js bridge.

    The bridge runs as a separate Node.js process and communicates
    via HTTP/WebSocket for real-time messaging.

    Features:
    - QR code authentication
    - Text and media messages
    - Voice note transcription
    - Group chat support
    """

    def __init__(self) -> None:
        super().__init__("whatsapp")
        self.settings = get_settings()
        # Allow env var overrides for Docker networking
        self.bridge_host = os.environ.get("WHATSAPP_BRIDGE_HOST") or self.settings.channels.whatsapp.bridge_host
        self.bridge_port = int(os.environ.get("WHATSAPP_BRIDGE_PORT", 0)) or self.settings.channels.whatsapp.bridge_port
        self.session_path = Path(self.settings.channels.whatsapp.session_path).expanduser()

        self._base_url = f"http://{self.bridge_host}:{self.bridge_port}"
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._listen_task: asyncio.Task[None] | None = None

        # Allowlist: only process messages from these numbers (empty = allow all)
        self._allowed_numbers: set[str] = set()
        for num in self.settings.channels.whatsapp.allowed_numbers:
            # Store normalized: digits only (no + prefix, no @c.us suffix)
            cleaned = num.strip().lstrip("+").split("@")[0]
            if cleaned:
                self._allowed_numbers.add(cleaned)
        if self._allowed_numbers:
            logger.info("WhatsApp allowlist active", allowed=list(self._allowed_numbers))

        # Pending approval requests
        self._pending_approvals: dict[str, asyncio.Future[dict[str, Any]]] = {}

        # Track message IDs we've sent to avoid echo loops
        self._sent_message_ids: set[str] = set()

    async def start(self) -> None:
        """Start the WhatsApp channel."""
        if self._connected:
            return

        # Check if bridge is running
        if not await self._check_bridge():
            logger.warning("WhatsApp bridge not running. Starting bridge...")
            if not await self._start_bridge():
                raise RuntimeError("Failed to start WhatsApp bridge")

        # Create HTTP session
        self._session = aiohttp.ClientSession()

        # Try WebSocket first for real-time messages, fall back to polling
        try:
            self._ws = await self._session.ws_connect(
                f"ws://{self.bridge_host}:{self.bridge_port}/ws",
                timeout=aiohttp.ClientTimeout(total=5),
            )
            self._listen_task = asyncio.create_task(self._listen_messages())
            self._connected = True
            logger.info("WhatsApp channel connected via WebSocket")
        except Exception as e:
            logger.warning("WebSocket connection failed, falling back to polling", error=str(e))
            # Fall back to HTTP polling for messages
            self._listen_task = asyncio.create_task(self._poll_messages())
            self._connected = True
            logger.info("WhatsApp channel connected via polling")

    async def stop(self) -> None:
        """Stop the WhatsApp channel."""
        if not self._connected:
            return

        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            await self._ws.close()

        if self._session:
            await self._session.close()

        self._connected = False
        logger.info("WhatsApp channel stopped")

    async def _check_bridge(self) -> bool:
        """Check if the bridge is running."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self._base_url}/status", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def _start_bridge(self) -> bool:
        """Start the Node.js bridge process."""
        bridge_path = Path(__file__).parent.parent.parent / "whatsapp-bridge"

        if not bridge_path.exists():
            logger.error("WhatsApp bridge not found", path=str(bridge_path))
            return False

        try:
            # Start the bridge process
            process = await asyncio.create_subprocess_exec(
                "node",
                "index.js",
                cwd=str(bridge_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait for bridge to be ready
            for _ in range(30):  # Wait up to 30 seconds
                await asyncio.sleep(1)
                if await self._check_bridge():
                    return True

            logger.error("Bridge did not start in time")
            return False

        except Exception as e:
            logger.error("Failed to start bridge", error=str(e))
            return False

    async def _listen_messages(self) -> None:
        """Listen for incoming messages from the bridge via WebSocket."""
        if not self._ws:
            return

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_bridge_message(data)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON from bridge")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("WebSocket error", error=str(self._ws.exception()))
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("WebSocket listener error, switching to polling", error=str(e))
            # Auto-fallback to polling if WebSocket dies
            self._ws = None
            self._listen_task = asyncio.create_task(self._poll_messages())

    async def _poll_messages(self) -> None:
        """Poll the bridge for new messages via HTTP (fallback when WebSocket unavailable)."""
        logger.info("Starting HTTP message polling for WhatsApp bridge")
        last_seen_id: str | None = None

        while self._connected and self._session:
            try:
                async with self._session.get(
                    f"{self._base_url}/messages",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        messages = data if isinstance(data, list) else data.get("messages", [])
                        for msg_data in messages:
                            msg_id = msg_data.get("id", {}).get("id", "") if isinstance(msg_data.get("id"), dict) else str(msg_data.get("id", ""))
                            if last_seen_id and msg_id == last_seen_id:
                                continue
                            # Process as a bridge message event
                            await self._handle_bridge_message({"type": "message", "message": msg_data, "from": msg_data.get("from", {}), "chatId": msg_data.get("chatId", "")})
                        if messages:
                            last_msg = messages[-1]
                            last_seen_id = last_msg.get("id", {}).get("id", "") if isinstance(last_msg.get("id"), dict) else str(last_msg.get("id", ""))
                    elif resp.status == 404:
                        # Bridge doesn't have /messages endpoint, just wait
                        pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Poll error (will retry)", error=str(e))

            await asyncio.sleep(2)  # Poll every 2 seconds

    def _is_number_allowed(self, sender_id: str) -> bool:
        """Check if a sender is in the allowlist. Empty allowlist = allow all."""
        if not self._allowed_numbers:
            return True
        # Normalize sender: strip @c.us / @g.us suffix, strip +
        normalized = sender_id.strip().lstrip("+").split("@")[0]
        return normalized in self._allowed_numbers

    async def _handle_bridge_message(self, data: dict[str, Any]) -> None:
        """Handle a message from the bridge."""
        event_type = data.get("type")

        if event_type == "message":
            msg_data = data.get("message", {})

            # Skip messages we sent (prevents echo loop)
            msg_id_obj = msg_data.get("id", {})
            msg_serialized = msg_id_obj.get("_serialized", "") if isinstance(msg_id_obj, dict) else str(msg_id_obj)
            if msg_serialized and msg_serialized in self._sent_message_ids:
                self._sent_message_ids.discard(msg_serialized)
                return

            from_info = data.get("from", {})
            sender_id = from_info.get("id", "") if isinstance(from_info, dict) else str(from_info)
            is_from_me = msg_data.get("fromMe", False)
            chat_id = data.get("chatId", "")

            if is_from_me:
                # For messages WE sent: only process if it's the "Message Yourself"
                # chat (chatId matches our own number). Skip messages to other people.
                chat_number = chat_id.split("@")[0]
                if not self._is_number_allowed(chat_number):
                    # User is messaging someone else — not a command to the bot
                    return
            else:
                # For incoming messages from others: check allowlist
                if not self._is_number_allowed(sender_id):
                    logger.warning(
                        "WhatsApp message blocked by allowlist",
                        sender=sender_id,
                        allowed=list(self._allowed_numbers),
                    )
                    return

            logger.info(
                "WhatsApp message accepted",
                sender=sender_id,
                from_me=is_from_me,
                chat_id=chat_id,
                body_preview=(msg_data.get("body", "") or "")[:60],
            )

            message = await self._build_message(data)
            await self._dispatch_message(message)

        elif event_type == "qr":
            # QR code for authentication
            qr_code = data.get("qr")
            logger.info("WhatsApp QR code received. Scan to authenticate.")
            # Print QR info to terminal so user can find it
            print("\n" + "=" * 60)
            print("  WHATSAPP QR CODE - Scan with your phone")
            print("=" * 60)
            print(f"  View QR in browser: {self._base_url}/qr")
            print(f"  Or in the dashboard: Settings > Channels > WhatsApp")
            print("=" * 60 + "\n")

        elif event_type == "ready":
            logger.info("WhatsApp client ready")

        elif event_type == "authenticated":
            logger.info("WhatsApp authenticated")

        elif event_type == "reaction":
            await self._handle_reaction(data)

    async def _build_message(self, data: dict[str, Any]) -> Message:
        """Build a Message from bridge data."""
        msg_data = data.get("message", {})
        from_info = data.get("from", {})

        # Determine message type
        msg_type = MessageType.TEXT
        attachments = []

        if msg_data.get("hasMedia"):
            media_type = msg_data.get("type", "")
            if "image" in media_type:
                msg_type = MessageType.IMAGE
            elif "audio" in media_type or "ptt" in media_type:
                msg_type = MessageType.AUDIO
            elif "video" in media_type:
                msg_type = MessageType.VIDEO
            else:
                msg_type = MessageType.FILE

            # Create attachment
            attachments.append(
                Attachment(
                    type=msg_type,
                    url=msg_data.get("mediaUrl"),
                    mime_type=msg_data.get("mimetype"),
                    filename=msg_data.get("filename"),
                    metadata={
                        "media_key": msg_data.get("mediaKey"),
                        "is_voice": msg_data.get("isVoice", False),
                    },
                )
            )

        # Get the serialized message ID (needed for quoting/replying)
        msg_id_obj = msg_data.get("id", {})
        if isinstance(msg_id_obj, dict):
            msg_id = msg_id_obj.get("_serialized", msg_id_obj.get("id", ""))
        else:
            msg_id = str(msg_id_obj)

        return Message(
            id=msg_id,
            channel=self.name,
            user_id=from_info.get("id", ""),
            user_name=from_info.get("pushname") or from_info.get("name"),
            content=msg_data.get("body", ""),
            message_type=msg_type,
            attachments=attachments,
            reply_to=msg_data.get("quotedMsgId"),
            metadata={
                "chat_id": data.get("chatId"),
                "is_group": data.get("isGroup", False),
                "timestamp": msg_data.get("timestamp"),
            },
            raw=data,
        )

    async def _handle_reaction(self, data: dict[str, Any]) -> None:
        """Handle a reaction event."""
        msg_id = data.get("messageId")
        reaction = data.get("reaction")
        user_id = data.get("userId")

        # Check for pending approval
        if msg_id in self._pending_approvals:
            future = self._pending_approvals[msg_id]

            # Map reactions to approval
            if reaction in ["👍", "✅", "✓", "👌"]:
                future.set_result({
                    "approved": True,
                    "approved_by": user_id,
                    "reaction": reaction,
                })
            elif reaction in ["👎", "❌", "✗", "🚫"]:
                future.set_result({
                    "approved": False,
                    "approved_by": user_id,
                    "reaction": reaction,
                })

    async def send_message(
        self,
        user_id: str,
        content: str,
        reply_to: str | None = None,
        thread_id: str | None = None,
        attachments: list[Attachment] | None = None,
    ) -> str | None:
        """Send a message to a WhatsApp user or group."""
        if not self._session:
            return None

        try:
            payload: dict[str, Any] = {
                "chatId": user_id,
                "content": content,
            }

            if reply_to:
                payload["quotedMessageId"] = reply_to

            # Handle attachments
            if attachments:
                for attachment in attachments:
                    if attachment.type == MessageType.AUDIO:
                        # Send as voice note
                        payload["sendAsVoice"] = True

                    if attachment.data:
                        payload["media"] = base64.b64encode(attachment.data).decode()
                        payload["mediaType"] = attachment.mime_type
                        payload["filename"] = attachment.filename
                    elif attachment.path:
                        with open(attachment.path, "rb") as f:
                            payload["media"] = base64.b64encode(f.read()).decode()
                        payload["mediaType"] = attachment.mime_type
                        payload["filename"] = attachment.filename or Path(attachment.path).name

            async with self._session.post(
                f"{self._base_url}/send",
                json=payload,
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    msg_id = result.get("messageId")
                    # Track so we skip this message when it echoes back via message_create
                    if msg_id:
                        self._sent_message_ids.add(msg_id)
                    logger.info("WhatsApp message sent", chat_id=user_id, message_id=msg_id)
                    return msg_id
                else:
                    body = await resp.text()
                    logger.error("Failed to send WhatsApp message", status=resp.status, body=body[:200], chat_id=user_id)
                    return None

        except Exception as e:
            logger.error("WhatsApp send error", error=str(e))
            return None

    async def send_reaction(
        self,
        message_id: str,
        reaction: str,
    ) -> bool:
        """Add a reaction to a message."""
        if not self._session:
            return False

        try:
            async with self._session.post(
                f"{self._base_url}/react",
                json={"messageId": message_id, "reaction": reaction},
            ) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error("WhatsApp reaction error", error=str(e))
            return False

    async def request_approval(
        self,
        user_id: str,
        action_description: str,
        approval_id: str,
        timeout: int = 300,
    ) -> dict[str, Any]:
        """Request approval via WhatsApp message."""
        # Normalize user_id to a valid chatId (strip device suffix like :23)
        # "17785129687:23@c.us" → "17785129687@c.us"
        chat_id = user_id
        if ":" in chat_id and "@" in chat_id:
            number_part = chat_id.split(":")[0]
            domain_part = chat_id.split("@")[-1]
            chat_id = f"{number_part}@{domain_part}"

        # Send approval request message
        message = f"""🔐 *Approval Required*

{action_description}

React with:
👍 to approve
👎 to deny"""

        message_id = await self.send_message(chat_id, message)
        if not message_id:
            return {"approved": False, "error": "Failed to send approval request"}

        # Wait for reaction
        future: asyncio.Future[dict[str, Any]] = asyncio.Future()
        self._pending_approvals[message_id] = future

        try:
            result = await asyncio.wait_for(future, timeout=timeout)

            # Update message with result
            status = "✅ Approved" if result["approved"] else "❌ Denied"
            await self.send_message(
                chat_id,
                f"Request {status}",
                reply_to=message_id,
            )

            return result

        except asyncio.TimeoutError:
            await self.send_message(
                chat_id,
                "⏱️ Request timed out",
                reply_to=message_id,
            )
            return {"approved": False, "timeout": True}

        finally:
            self._pending_approvals.pop(message_id, None)

    async def send_typing_indicator(self, user_id: str) -> None:
        """Send typing indicator."""
        if not self._session:
            return

        try:
            await self._session.post(
                f"{self._base_url}/typing",
                json={"chatId": user_id, "typing": True},
            )
        except Exception:
            pass

    async def get_user_info(self, user_id: str) -> dict[str, Any] | None:
        """Get WhatsApp user information."""
        if not self._session:
            return None

        try:
            async with self._session.get(
                f"{self._base_url}/contact/{user_id}",
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass
        return None

    async def download_attachment(self, attachment: Attachment) -> bytes | None:
        """Download media from WhatsApp."""
        if attachment.data:
            return attachment.data

        if not self._session or not attachment.metadata.get("media_key"):
            return None

        try:
            async with self._session.post(
                f"{self._base_url}/download",
                json={"mediaKey": attachment.metadata["media_key"]},
            ) as resp:
                if resp.status == 200:
                    return await resp.read()
        except Exception as e:
            logger.error("Failed to download WhatsApp media", error=str(e))

        return None

    async def send_voice_note(
        self,
        user_id: str,
        audio_path: str,
        reply_to: str | None = None,
    ) -> str | None:
        """Send an audio file as a voice note."""
        path = Path(audio_path).expanduser()
        if not path.exists():
            return None

        attachment = Attachment(
            type=MessageType.AUDIO,
            path=str(path),
            mime_type="audio/ogg; codecs=opus",
            metadata={"is_voice": True},
        )

        return await self.send_message(
            user_id=user_id,
            content="",
            reply_to=reply_to,
            attachments=[attachment],
        )

    def get_qr_code_url(self) -> str:
        """Get URL to view QR code for authentication."""
        return f"{self._base_url}/qr"
