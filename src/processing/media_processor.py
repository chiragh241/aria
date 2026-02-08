"""Media auto-understanding — auto-describe images and transcribe audio.

When a user sends an image or voice note via WhatsApp/Slack, this module
automatically processes the attachment and injects a description as context
so the AI understands what was shared without being explicitly asked.
"""

import base64
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Supported MIME types for auto-processing
IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/heic"}
AUDIO_TYPES = {"audio/ogg", "audio/mpeg", "audio/mp4", "audio/webm", "audio/wav", "audio/x-m4a"}


async def process_attachments(
    attachments: list[dict[str, Any]],
    skill_registry: Any,
) -> str | None:
    """Process message attachments and return context string.

    Each attachment dict should have:
      - "type": MIME type (e.g., "image/jpeg")
      - "data": base64-encoded content OR
      - "path": local file path OR
      - "url": URL to fetch

    Returns formatted context string or None.
    """
    if not attachments or not skill_registry:
        return None

    results: list[str] = []

    for att in attachments:
        mime_type = att.get("type", "").lower()
        try:
            if mime_type in IMAGE_TYPES:
                desc = await _describe_image(att, skill_registry)
                if desc:
                    results.append(f"[Image: {desc}]")
            elif mime_type in AUDIO_TYPES:
                text = await _transcribe_audio(att, skill_registry)
                if text:
                    results.append(f"[Voice message: \"{text}\"]")
        except Exception as e:
            logger.warning("Failed to process attachment", type=mime_type, error=str(e))

    if not results:
        return None

    return "\n\n---\nAttachment context (auto-processed):\n" + "\n".join(results) + "\n---"


async def _describe_image(attachment: dict[str, Any], skill_registry: Any) -> str | None:
    """Use the image skill to describe an image."""
    skill = skill_registry.get_skill("image")
    if not skill:
        return None

    # Get image data
    data = attachment.get("data")
    path = attachment.get("path")

    if path and not data:
        try:
            raw = Path(path).read_bytes()
            data = base64.b64encode(raw).decode()
        except Exception as e:
            logger.warning("Failed to read image file", path=path, error=str(e))
            return None

    if not data:
        return None

    try:
        result = await skill.execute("analyze", image_data=data, prompt="Describe this image briefly.")
        if hasattr(result, "output"):
            output = result.output
            if isinstance(output, dict):
                return output.get("description", output.get("text", str(output)))
            return str(output)
    except Exception as e:
        logger.debug("Image analysis failed", error=str(e))

    return None


async def _transcribe_audio(attachment: dict[str, Any], skill_registry: Any) -> str | None:
    """Use the STT skill to transcribe audio."""
    skill = skill_registry.get_skill("stt")
    if not skill:
        return None

    data = attachment.get("data")
    path = attachment.get("path")

    if path and not data:
        try:
            raw = Path(path).read_bytes()
            data = base64.b64encode(raw).decode()
        except Exception as e:
            logger.warning("Failed to read audio file", path=path, error=str(e))
            return None

    if not data:
        return None

    try:
        audio_bytes = base64.b64decode(data)
        fmt = attachment.get("format", "ogg")
        result = await skill.execute("transcribe", audio_data=audio_bytes, format=fmt)
        if hasattr(result, "output"):
            output = result.output
            if isinstance(output, dict):
                return output.get("text", str(output))
            return str(output)
    except Exception as e:
        logger.debug("Audio transcription failed", error=str(e))

    return None


def detect_attachments_in_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract attachment info from message metadata.

    Different channels store attachments differently:
    - WhatsApp: metadata["media"] with {mimetype, data, filename}
    - Slack: metadata["files"] with {mimetype, url_private, name}
    """
    attachments = []

    # WhatsApp media
    if "media" in metadata:
        media = metadata["media"]
        if isinstance(media, dict):
            media = [media]
        for m in media:
            att = {
                "type": m.get("mimetype", m.get("type", "")),
                "data": m.get("data"),
                "path": m.get("path"),
                "url": m.get("url"),
                "filename": m.get("filename", ""),
            }
            if att["type"]:
                attachments.append(att)

    # Slack files
    if "files" in metadata:
        for f in metadata["files"]:
            att = {
                "type": f.get("mimetype", f.get("filetype", "")),
                "url": f.get("url_private", f.get("url", "")),
                "filename": f.get("name", ""),
            }
            if att["type"]:
                attachments.append(att)

    # Generic attachments field
    if "attachments" in metadata:
        for a in metadata["attachments"]:
            if isinstance(a, dict) and a.get("type"):
                attachments.append(a)

    return attachments
