"""Text-to-speech skill using edge-tts."""

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..base import BaseSkill, SkillResult


class TTSSkill(BaseSkill):
    """
    Skill for text-to-speech conversion.

    Uses edge-tts (Microsoft Azure TTS) for high-quality,
    free text-to-speech synthesis.
    """

    name = "tts"
    description = "Text-to-speech synthesis"
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.default_voice = config.get("voice", "en-US-AriaNeural")
        self._available_voices: list[dict[str, Any]] = []

    def _register_capabilities(self) -> None:
        """Register TTS capabilities."""
        self.register_capability(
            name="speak",
            description="Convert text to speech and save as audio file",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to convert to speech"},
                    "voice": {"type": "string", "description": "Voice name to use"},
                    "output_path": {"type": "string", "description": "Output file path (MP3)"},
                    "rate": {"type": "string", "description": "Speech rate (e.g., '+20%', '-10%')"},
                    "volume": {"type": "string", "description": "Volume (e.g., '+50%', '-20%')"},
                },
                "required": ["text"],
            },
        )

        self.register_capability(
            name="list_voices",
            description="List available voices",
            parameters={
                "type": "object",
                "properties": {
                    "language": {"type": "string", "description": "Filter by language code (e.g., 'en', 'es')"},
                },
            },
        )

    async def initialize(self) -> None:
        """Initialize TTS and fetch available voices."""
        try:
            import edge_tts

            self._available_voices = await edge_tts.list_voices()
            self._initialized = True
        except ImportError:
            self._initialized = False
        except Exception:
            self._initialized = True  # Still try to work

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        """Execute a TTS capability."""
        start_time = datetime.now(timezone.utc)

        if not self._initialized:
            return self._error_result(
                "TTS not initialized. Install edge-tts: pip install edge-tts",
                start_time,
            )

        if capability == "speak":
            return await self._speak(start_time, **kwargs)
        elif capability == "list_voices":
            return await self._list_voices(start_time, **kwargs)
        else:
            return self._error_result(f"Unknown capability: {capability}", start_time)

    async def _speak(
        self,
        start_time: datetime,
        text: str,
        voice: str | None = None,
        output_path: str | None = None,
        rate: str | None = None,
        volume: str | None = None,
    ) -> SkillResult:
        """Convert text to speech."""
        try:
            import edge_tts

            voice = voice or self.default_voice

            # Create output path if not specified
            if output_path:
                output_file = Path(output_path).expanduser()
                output_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_file = Path(tempfile.mktemp(suffix=".mp3"))

            # Create TTS communicate object
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate or "+0%",
                volume=volume or "+0%",
            )

            # Generate audio
            await communicate.save(str(output_file))

            return self._success_result(
                {
                    "output_path": str(output_file),
                    "voice": voice,
                    "text_length": len(text),
                    "file_size": output_file.stat().st_size,
                },
                start_time,
            )

        except Exception as e:
            return self._error_result(str(e), start_time)

    async def _list_voices(
        self,
        start_time: datetime,
        language: str | None = None,
    ) -> SkillResult:
        """List available voices."""
        try:
            import edge_tts

            if not self._available_voices:
                self._available_voices = await edge_tts.list_voices()

            voices = self._available_voices

            # Filter by language if specified
            if language:
                language = language.lower()
                voices = [
                    v for v in voices
                    if v.get("Locale", "").lower().startswith(language)
                ]

            # Simplify voice data
            simplified = [
                {
                    "name": v.get("ShortName"),
                    "locale": v.get("Locale"),
                    "gender": v.get("Gender"),
                    "friendly_name": v.get("FriendlyName"),
                }
                for v in voices
            ]

            return self._success_result(
                {
                    "voices": simplified,
                    "count": len(simplified),
                    "filtered_by": language,
                },
                start_time,
            )

        except Exception as e:
            return self._error_result(str(e), start_time)
