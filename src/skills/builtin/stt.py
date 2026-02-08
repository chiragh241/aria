"""Speech-to-text skill using Whisper."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...utils.logging import get_logger
from ..base import BaseSkill, SkillResult

logger = get_logger(__name__)


class STTSkill(BaseSkill):
    """
    Skill for speech-to-text transcription.

    Uses OpenAI's Whisper model for accurate,
    local speech recognition.
    """

    name = "stt"
    description = "Speech-to-text transcription"
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.model_name = config.get("model", "base")
        self._model: Any = None

    def _register_capabilities(self) -> None:
        """Register STT capabilities."""
        self.register_capability(
            name="transcribe",
            description="Transcribe audio to text",
            parameters={
                "type": "object",
                "properties": {
                    "audio_path": {"type": "string", "description": "Path to audio file"},
                    "language": {"type": "string", "description": "Language code (optional, auto-detect if not specified)"},
                    "task": {"type": "string", "enum": ["transcribe", "translate"], "default": "transcribe"},
                },
                "required": ["audio_path"],
            },
            security_action="read_files",
        )

        self.register_capability(
            name="transcribe_bytes",
            description="Transcribe audio from bytes",
            parameters={
                "type": "object",
                "properties": {
                    "audio_data": {"type": "string", "description": "Base64 encoded audio data"},
                    "format": {"type": "string", "description": "Audio format (mp3, wav, etc.)"},
                    "language": {"type": "string", "description": "Language code"},
                },
                "required": ["audio_data"],
            },
        )

    async def initialize(self) -> None:
        """Initialize Whisper model."""
        try:
            import whisper

            logger.info("STT: loading Whisper model", model=self.model_name)
            self._model = whisper.load_model(self.model_name)
            self._initialized = True
            logger.info("STT: Whisper model loaded")
        except ImportError as e:
            logger.error("STT: whisper not installed", error=str(e))
            self._initialized = False
        except Exception as e:
            logger.error("STT: failed to load model", error=str(e), exc_info=True)
            self._initialized = False

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        """Execute an STT capability."""
        start_time = datetime.now(timezone.utc)
        logger.info("STT: execute", capability=capability, has_audio=bool(kwargs.get("audio_data")))

        if not self._initialized:
            logger.warning("STT: not initialized — whisper not loaded")
            return self._error_result(
                "STT not initialized. Install whisper: pip install openai-whisper",
                start_time,
            )

        if capability == "transcribe":
            return await self._transcribe(start_time, **kwargs)
        elif capability == "transcribe_bytes":
            return await self._transcribe_bytes(start_time, **kwargs)
        else:
            return self._error_result(f"Unknown capability: {capability}", start_time)

    async def _transcribe(
        self,
        start_time: datetime,
        audio_path: str,
        language: str | None = None,
        task: str = "transcribe",
    ) -> SkillResult:
        """Transcribe an audio file."""
        try:
            import asyncio

            audio_file = Path(audio_path).expanduser()
            if not audio_file.exists():
                return self._error_result(f"Audio file not found: {audio_path}", start_time)

            # Run transcription in executor to avoid blocking
            loop = asyncio.get_event_loop()

            options = {
                "task": task,
                "fp16": False,  # Use FP32 for CPU compatibility
            }
            if language:
                options["language"] = language

            result = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(str(audio_file), **options),
            )

            return self._success_result(
                {
                    "text": result["text"],
                    "language": result.get("language"),
                    "segments": [
                        {
                            "start": s["start"],
                            "end": s["end"],
                            "text": s["text"],
                        }
                        for s in result.get("segments", [])
                    ],
                },
                start_time,
            )

        except Exception as e:
            return self._error_result(str(e), start_time)

    async def _transcribe_bytes(
        self,
        start_time: datetime,
        audio_data: str,
        format: str = "mp3",
        language: str | None = None,
    ) -> SkillResult:
        """Transcribe audio from base64 bytes."""
        try:
            import base64

            audio_bytes = base64.b64decode(audio_data)
            logger.info("STT: transcribe_bytes", format=format, size_bytes=len(audio_bytes))

            with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name

            try:
                result = await self._transcribe(
                    start_time,
                    audio_path=temp_path,
                    language=language,
                )
                if result.success:
                    logger.info("STT: transcribe_bytes success")
                else:
                    logger.warning("STT: transcribe_bytes failed", error=result.error)
                return result
            finally:
                Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            logger.error("STT: transcribe_bytes error", error=str(e), exc_info=True)
            return self._error_result(str(e), start_time)

    async def shutdown(self) -> None:
        """Shutdown and free model memory."""
        self._model = None
        self._initialized = False
