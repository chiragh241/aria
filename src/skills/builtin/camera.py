"""Camera capture and screenshot skill."""

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...utils.logging import get_logger
from ..base import BaseSkill, SkillResult

logger = get_logger(__name__)


class CameraSkill(BaseSkill):
    """
    Skill for taking screenshots (desktop) and optionally photos from a local camera.
    On the server, take_screenshot uses OS tools (e.g. screencapture on macOS); take_photo
    requires a device with camera or a paired client that can capture and upload.
    """

    name = "camera"
    description = "Take screenshots or photos (desktop screenshot; device camera when available)"
    version = "1.0.0"

    def _register_capabilities(self) -> None:
        """Register camera capabilities."""
        self.register_capability(
            name="take_screenshot",
            description="Capture a screenshot of the desktop and save to a file",
            parameters={
                "type": "object",
                "properties": {
                    "output_path": {"type": "string", "description": "Path to save the image (PNG or JPG)"},
                    "format": {"type": "string", "enum": ["png", "jpg"], "default": "png"},
                },
                "required": ["output_path"],
            },
            security_action="write_files",
        )
        self.register_capability(
            name="take_photo",
            description="Capture a photo from the default camera (if available on this machine). On headless servers use a client that uploads images.",
            parameters={
                "type": "object",
                "properties": {
                    "output_path": {"type": "string", "description": "Path to save the photo"},
                },
                "required": ["output_path"],
            },
            security_action="write_files",
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        """Execute a camera capability."""
        start_time = datetime.now(timezone.utc)
        if capability == "take_screenshot":
            return await self._take_screenshot(start_time, **kwargs)
        if capability == "take_photo":
            return await self._take_photo(start_time, **kwargs)
        return self._error_result(f"Unknown capability: {capability}", start_time)

    async def _take_screenshot(
        self,
        start_time: datetime,
        output_path: str,
        format: str = "png",
    ) -> SkillResult:
        """Capture desktop screenshot. Uses screencapture on macOS, scrot on Linux if available."""
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        if format == "jpg":
            if not str(out).lower().endswith((".jpg", ".jpeg")):
                out = out.with_suffix(".jpg")
        else:
            if not str(out).lower().endswith(".png"):
                out = out.with_suffix(".png")

        import platform
        system = platform.system()
        try:
            if system == "Darwin":
                # -x: no sound, -o: open (optional), we write to path
                cmd = ["screencapture", "-x", str(out)]
                ret, _ = await _run_cmd(cmd)
                if ret != 0:
                    return self._error_result(f"screencapture failed with code {ret}", start_time)
            elif system == "Linux":
                # scrot or gnome-screenshot
                for cmd in [["scrot", "-o", str(out)], ["gnome-screenshot", "-f", str(out)]]:
                    ret, _ = await _run_cmd(cmd)
                    if ret == 0:
                        break
                else:
                    return self._error_result(
                        "Screenshot not available: install scrot or gnome-screenshot on Linux",
                        start_time,
                    )
            else:
                return self._error_result(f"Screenshot not implemented for {system}", start_time)
        except FileNotFoundError as e:
            return self._error_result(f"Screenshot tool not found: {e}", start_time)
        except Exception as e:
            logger.exception("take_screenshot failed")
            return self._error_result(str(e), start_time)

        if not out.exists():
            return self._error_result("Screenshot file was not created", start_time)
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return SkillResult(
            success=True,
            output={"path": str(out), "size_bytes": out.stat().st_size},
            execution_time_ms=elapsed,
        )

    async def _take_photo(
        self,
        start_time: datetime,
        output_path: str,
    ) -> SkillResult:
        """Try to capture from default camera (e.g. OpenCV). If not available, return helpful message."""
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        if not str(out).lower().endswith((".jpg", ".jpeg", ".png")):
            out = out.with_suffix(".jpg")

        try:
            import cv2  # type: ignore
        except ImportError:
            return self._error_result(
                "Camera capture requires opencv-python. For device camera use a client that captures and uploads.",
                start_time,
            )
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return self._error_result(
                    "No camera available on this machine. Use take_screenshot for desktop capture or a client that supports camera upload.",
                    start_time,
                )
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return self._error_result("Failed to read frame from camera", start_time)
            cv2.imwrite(str(out), frame)
        except Exception as e:
            logger.warning("take_photo failed", error=str(e))
            return self._error_result(str(e), start_time)

        if not out.exists():
            return self._error_result("Photo file was not created", start_time)
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return SkillResult(
            success=True,
            output={"path": str(out), "size_bytes": out.stat().st_size},
            execution_time_ms=elapsed,
        )

    def _error_result(self, error: str, start_time: datetime) -> SkillResult:
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return SkillResult(success=False, error=error, execution_time_ms=elapsed)


async def _run_cmd(cmd: list[str]) -> tuple[int, bytes]:
    """Run a subprocess and return (returncode, stderr)."""
    import asyncio
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    return (proc.returncode or 0, stderr or b"")
