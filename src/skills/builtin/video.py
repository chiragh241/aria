"""Video processing skill using ffmpeg."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..base import BaseSkill, SkillResult


class VideoSkill(BaseSkill):
    """
    Skill for video processing using ffmpeg.

    Supports conversion, trimming, audio extraction,
    and thumbnail generation.
    """

    name = "video"
    description = "Video processing and conversion"
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.ffmpeg_path = config.get("ffmpeg_path", "ffmpeg")
        self.ffprobe_path = config.get("ffprobe_path", "ffprobe")

    def _register_capabilities(self) -> None:
        """Register video capabilities."""
        self.register_capability(
            name="convert",
            description="Convert video to different format",
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Input video path"},
                    "output_path": {"type": "string", "description": "Output video path"},
                    "codec": {"type": "string", "description": "Video codec (h264, h265, vp9)"},
                    "audio_codec": {"type": "string", "description": "Audio codec (aac, mp3)"},
                    "resolution": {"type": "string", "description": "Resolution (1920x1080, 1280x720)"},
                    "bitrate": {"type": "string", "description": "Video bitrate (e.g., 2M)"},
                },
                "required": ["input_path", "output_path"],
            },
            security_action="write_files",
        )

        self.register_capability(
            name="trim",
            description="Trim video to specific time range",
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Input video path"},
                    "output_path": {"type": "string", "description": "Output video path"},
                    "start": {"type": "string", "description": "Start time (HH:MM:SS or seconds)"},
                    "end": {"type": "string", "description": "End time (HH:MM:SS or seconds)"},
                    "duration": {"type": "string", "description": "Duration (alternative to end)"},
                },
                "required": ["input_path", "output_path", "start"],
            },
            security_action="write_files",
        )

        self.register_capability(
            name="extract_audio",
            description="Extract audio from video",
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Input video path"},
                    "output_path": {"type": "string", "description": "Output audio path (mp3, wav, etc.)"},
                    "codec": {"type": "string", "description": "Audio codec"},
                    "bitrate": {"type": "string", "description": "Audio bitrate (e.g., 192k)"},
                },
                "required": ["input_path", "output_path"],
            },
            security_action="write_files",
        )

        self.register_capability(
            name="thumbnail",
            description="Create thumbnail from video",
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Input video path"},
                    "output_path": {"type": "string", "description": "Output image path"},
                    "time": {"type": "string", "description": "Time position (HH:MM:SS or seconds)", "default": "00:00:01"},
                    "width": {"type": "integer", "description": "Thumbnail width"},
                },
                "required": ["input_path", "output_path"],
            },
            security_action="write_files",
        )

        self.register_capability(
            name="get_info",
            description="Get video information",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Video path"},
                },
                "required": ["path"],
            },
            security_action="read_files",
        )

        self.register_capability(
            name="compress",
            description="Compress video to reduce file size",
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Input video path"},
                    "output_path": {"type": "string", "description": "Output video path"},
                    "target_size_mb": {"type": "number", "description": "Target file size in MB"},
                    "quality": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
                },
                "required": ["input_path", "output_path"],
            },
            security_action="write_files",
        )

    async def initialize(self) -> None:
        """Check if ffmpeg is available."""
        try:
            process = await asyncio.create_subprocess_exec(
                self.ffmpeg_path,
                "-version",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await process.wait()
            self._initialized = process.returncode == 0
        except Exception:
            self._initialized = False

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        """Execute a video capability."""
        start_time = datetime.now(timezone.utc)

        if not self._initialized:
            return self._error_result(
                "ffmpeg not available. Please install ffmpeg.",
                start_time,
            )

        handlers = {
            "convert": self._convert,
            "trim": self._trim,
            "extract_audio": self._extract_audio,
            "thumbnail": self._thumbnail,
            "get_info": self._get_info,
            "compress": self._compress,
        }

        handler = handlers.get(capability)
        if not handler:
            return self._error_result(f"Unknown capability: {capability}", start_time)

        try:
            result = await handler(**kwargs)
            return self._success_result(result, start_time)
        except Exception as e:
            return self._error_result(str(e), start_time)

    async def _run_ffmpeg(self, args: list[str]) -> tuple[int, str, str]:
        """Run ffmpeg with arguments."""
        process = await asyncio.create_subprocess_exec(
            self.ffmpeg_path,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return (
            process.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )

    async def _convert(
        self,
        input_path: str,
        output_path: str,
        codec: str | None = None,
        audio_codec: str | None = None,
        resolution: str | None = None,
        bitrate: str | None = None,
    ) -> dict[str, Any]:
        """Convert video format."""
        input_file = Path(input_path).expanduser()
        output_file = Path(output_path).expanduser()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        args = ["-i", str(input_file), "-y"]

        if codec:
            args.extend(["-c:v", codec])
        if audio_codec:
            args.extend(["-c:a", audio_codec])
        if resolution:
            args.extend(["-s", resolution])
        if bitrate:
            args.extend(["-b:v", bitrate])

        args.append(str(output_file))

        returncode, stdout, stderr = await self._run_ffmpeg(args)

        if returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr}")

        return {
            "input_path": str(input_file),
            "output_path": str(output_file),
            "file_size": output_file.stat().st_size,
        }

    async def _trim(
        self,
        input_path: str,
        output_path: str,
        start: str,
        end: str | None = None,
        duration: str | None = None,
    ) -> dict[str, Any]:
        """Trim video."""
        input_file = Path(input_path).expanduser()
        output_file = Path(output_path).expanduser()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        args = ["-i", str(input_file), "-ss", start, "-y"]

        if end:
            args.extend(["-to", end])
        elif duration:
            args.extend(["-t", duration])

        args.extend(["-c", "copy", str(output_file)])

        returncode, stdout, stderr = await self._run_ffmpeg(args)

        if returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr}")

        return {
            "input_path": str(input_file),
            "output_path": str(output_file),
            "start": start,
            "end": end,
            "duration": duration,
        }

    async def _extract_audio(
        self,
        input_path: str,
        output_path: str,
        codec: str | None = None,
        bitrate: str | None = None,
    ) -> dict[str, Any]:
        """Extract audio from video."""
        input_file = Path(input_path).expanduser()
        output_file = Path(output_path).expanduser()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        args = ["-i", str(input_file), "-vn", "-y"]

        if codec:
            args.extend(["-c:a", codec])
        if bitrate:
            args.extend(["-b:a", bitrate])

        args.append(str(output_file))

        returncode, stdout, stderr = await self._run_ffmpeg(args)

        if returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr}")

        return {
            "input_path": str(input_file),
            "output_path": str(output_file),
            "file_size": output_file.stat().st_size,
        }

    async def _thumbnail(
        self,
        input_path: str,
        output_path: str,
        time: str = "00:00:01",
        width: int | None = None,
    ) -> dict[str, Any]:
        """Create thumbnail from video."""
        input_file = Path(input_path).expanduser()
        output_file = Path(output_path).expanduser()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        args = ["-i", str(input_file), "-ss", time, "-vframes", "1", "-y"]

        if width:
            args.extend(["-vf", f"scale={width}:-1"])

        args.append(str(output_file))

        returncode, stdout, stderr = await self._run_ffmpeg(args)

        if returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr}")

        return {
            "input_path": str(input_file),
            "output_path": str(output_file),
            "time": time,
        }

    async def _get_info(self, path: str) -> dict[str, Any]:
        """Get video information."""
        video_path = Path(path).expanduser()

        process = await asyncio.create_subprocess_exec(
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {stderr.decode()}")

        data = json.loads(stdout.decode())

        # Extract relevant info
        format_info = data.get("format", {})
        streams = data.get("streams", [])

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        return {
            "path": str(video_path),
            "format": format_info.get("format_name"),
            "duration": float(format_info.get("duration", 0)),
            "size": int(format_info.get("size", 0)),
            "bitrate": int(format_info.get("bit_rate", 0)),
            "video": {
                "codec": video_stream.get("codec_name") if video_stream else None,
                "width": video_stream.get("width") if video_stream else None,
                "height": video_stream.get("height") if video_stream else None,
                "fps": eval(video_stream.get("r_frame_rate", "0/1")) if video_stream else None,
            } if video_stream else None,
            "audio": {
                "codec": audio_stream.get("codec_name") if audio_stream else None,
                "sample_rate": int(audio_stream.get("sample_rate", 0)) if audio_stream else None,
                "channels": audio_stream.get("channels") if audio_stream else None,
            } if audio_stream else None,
        }

    async def _compress(
        self,
        input_path: str,
        output_path: str,
        target_size_mb: float | None = None,
        quality: str = "medium",
    ) -> dict[str, Any]:
        """Compress video."""
        input_file = Path(input_path).expanduser()
        output_file = Path(output_path).expanduser()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Get original info
        info = await self._get_info(input_path)
        duration = info["duration"]

        # Calculate target bitrate if size specified
        if target_size_mb and duration > 0:
            target_bits = target_size_mb * 8 * 1024 * 1024
            target_bitrate = int(target_bits / duration * 0.9)  # 90% for overhead
            bitrate = f"{target_bitrate // 1000}k"
        else:
            # Use quality preset
            quality_bitrates = {
                "low": "500k",
                "medium": "1500k",
                "high": "3000k",
            }
            bitrate = quality_bitrates.get(quality, "1500k")

        args = [
            "-i", str(input_file),
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", bitrate,
            "-c:a", "aac",
            "-b:a", "128k",
            "-y",
            str(output_file),
        ]

        returncode, stdout, stderr = await self._run_ffmpeg(args)

        if returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr}")

        return {
            "input_path": str(input_file),
            "output_path": str(output_file),
            "original_size": info["size"],
            "compressed_size": output_file.stat().st_size,
            "compression_ratio": info["size"] / output_file.stat().st_size if output_file.stat().st_size > 0 else 0,
        }
