"""Image processing skill using Pillow."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..base import BaseSkill, SkillResult


class ImageSkill(BaseSkill):
    """
    Skill for image manipulation and processing.

    Uses Pillow for image operations like resize,
    crop, format conversion, and more.
    """

    name = "image"
    description = "Image manipulation and processing"
    version = "1.0.0"

    def _register_capabilities(self) -> None:
        """Register image capabilities."""
        self.register_capability(
            name="resize",
            description="Resize an image",
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Input image path"},
                    "output_path": {"type": "string", "description": "Output image path"},
                    "width": {"type": "integer", "description": "Target width"},
                    "height": {"type": "integer", "description": "Target height"},
                    "maintain_aspect": {"type": "boolean", "default": True},
                },
                "required": ["input_path", "output_path"],
            },
            security_action="write_files",
        )

        self.register_capability(
            name="crop",
            description="Crop an image",
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Input image path"},
                    "output_path": {"type": "string", "description": "Output image path"},
                    "left": {"type": "integer", "description": "Left coordinate"},
                    "top": {"type": "integer", "description": "Top coordinate"},
                    "right": {"type": "integer", "description": "Right coordinate"},
                    "bottom": {"type": "integer", "description": "Bottom coordinate"},
                },
                "required": ["input_path", "output_path", "left", "top", "right", "bottom"],
            },
            security_action="write_files",
        )

        self.register_capability(
            name="convert",
            description="Convert image format",
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Input image path"},
                    "output_path": {"type": "string", "description": "Output image path"},
                    "format": {"type": "string", "description": "Target format (PNG, JPEG, etc.)"},
                    "quality": {"type": "integer", "description": "Quality for JPEG (1-100)"},
                },
                "required": ["input_path", "output_path"],
            },
            security_action="write_files",
        )

        self.register_capability(
            name="rotate",
            description="Rotate an image",
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Input image path"},
                    "output_path": {"type": "string", "description": "Output image path"},
                    "degrees": {"type": "number", "description": "Rotation angle in degrees"},
                    "expand": {"type": "boolean", "default": True, "description": "Expand to fit rotated image"},
                },
                "required": ["input_path", "output_path", "degrees"],
            },
            security_action="write_files",
        )

        self.register_capability(
            name="add_watermark",
            description="Add text watermark to image",
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Input image path"},
                    "output_path": {"type": "string", "description": "Output image path"},
                    "text": {"type": "string", "description": "Watermark text"},
                    "position": {"type": "string", "enum": ["center", "bottom-right", "bottom-left", "top-right", "top-left"]},
                    "opacity": {"type": "number", "description": "Opacity (0-1)"},
                },
                "required": ["input_path", "output_path", "text"],
            },
            security_action="write_files",
        )

        self.register_capability(
            name="get_info",
            description="Get image information",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Image path"},
                },
                "required": ["path"],
            },
            security_action="read_files",
        )

        self.register_capability(
            name="thumbnail",
            description="Create a thumbnail",
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Input image path"},
                    "output_path": {"type": "string", "description": "Output image path"},
                    "size": {"type": "integer", "description": "Maximum dimension (width or height)"},
                },
                "required": ["input_path", "output_path", "size"],
            },
            security_action="write_files",
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        """Execute an image capability."""
        start_time = datetime.now(timezone.utc)

        handlers = {
            "resize": self._resize,
            "crop": self._crop,
            "convert": self._convert,
            "rotate": self._rotate,
            "add_watermark": self._add_watermark,
            "get_info": self._get_info,
            "thumbnail": self._thumbnail,
        }

        handler = handlers.get(capability)
        if not handler:
            return self._error_result(f"Unknown capability: {capability}", start_time)

        try:
            result = await handler(**kwargs)
            return self._success_result(result, start_time)
        except ImportError:
            return self._error_result("Pillow not installed. Run: pip install Pillow", start_time)
        except Exception as e:
            return self._error_result(str(e), start_time)

    async def _resize(
        self,
        input_path: str,
        output_path: str,
        width: int | None = None,
        height: int | None = None,
        maintain_aspect: bool = True,
    ) -> dict[str, Any]:
        """Resize an image."""
        from PIL import Image

        img = Image.open(Path(input_path).expanduser())
        original_size = img.size

        if maintain_aspect:
            if width and height:
                img.thumbnail((width, height), Image.Resampling.LANCZOS)
            elif width:
                ratio = width / img.width
                img = img.resize((width, int(img.height * ratio)), Image.Resampling.LANCZOS)
            elif height:
                ratio = height / img.height
                img = img.resize((int(img.width * ratio), height), Image.Resampling.LANCZOS)
        else:
            if width and height:
                img = img.resize((width, height), Image.Resampling.LANCZOS)

        output_file = Path(output_path).expanduser()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_file)

        return {
            "input_path": input_path,
            "output_path": str(output_file),
            "original_size": original_size,
            "new_size": img.size,
        }

    async def _crop(
        self,
        input_path: str,
        output_path: str,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> dict[str, Any]:
        """Crop an image."""
        from PIL import Image

        img = Image.open(Path(input_path).expanduser())
        cropped = img.crop((left, top, right, bottom))

        output_file = Path(output_path).expanduser()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_file)

        return {
            "input_path": input_path,
            "output_path": str(output_file),
            "original_size": img.size,
            "crop_box": (left, top, right, bottom),
            "new_size": cropped.size,
        }

    async def _convert(
        self,
        input_path: str,
        output_path: str,
        format: str | None = None,
        quality: int = 85,
    ) -> dict[str, Any]:
        """Convert image format."""
        from PIL import Image

        img = Image.open(Path(input_path).expanduser())

        output_file = Path(output_path).expanduser()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Determine format from extension if not specified
        if not format:
            format = output_file.suffix[1:].upper()

        # Convert RGBA to RGB for JPEG
        if format.upper() in ("JPEG", "JPG") and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        save_kwargs: dict[str, Any] = {}
        if format.upper() in ("JPEG", "JPG"):
            save_kwargs["quality"] = quality

        img.save(output_file, format=format, **save_kwargs)

        return {
            "input_path": input_path,
            "output_path": str(output_file),
            "format": format,
            "size": img.size,
        }

    async def _rotate(
        self,
        input_path: str,
        output_path: str,
        degrees: float,
        expand: bool = True,
    ) -> dict[str, Any]:
        """Rotate an image."""
        from PIL import Image

        img = Image.open(Path(input_path).expanduser())
        rotated = img.rotate(degrees, expand=expand, resample=Image.Resampling.BICUBIC)

        output_file = Path(output_path).expanduser()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        rotated.save(output_file)

        return {
            "input_path": input_path,
            "output_path": str(output_file),
            "degrees": degrees,
            "original_size": img.size,
            "new_size": rotated.size,
        }

    async def _add_watermark(
        self,
        input_path: str,
        output_path: str,
        text: str,
        position: str = "bottom-right",
        opacity: float = 0.5,
    ) -> dict[str, Any]:
        """Add watermark to image."""
        from PIL import Image, ImageDraw, ImageFont

        img = Image.open(Path(input_path).expanduser()).convert("RGBA")

        # Create watermark layer
        watermark = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)

        # Use default font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
        except Exception:
            font = ImageFont.load_default()

        # Calculate text size
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Calculate position
        margin = 20
        positions = {
            "center": ((img.width - text_width) // 2, (img.height - text_height) // 2),
            "bottom-right": (img.width - text_width - margin, img.height - text_height - margin),
            "bottom-left": (margin, img.height - text_height - margin),
            "top-right": (img.width - text_width - margin, margin),
            "top-left": (margin, margin),
        }
        pos = positions.get(position, positions["bottom-right"])

        # Draw text
        alpha = int(255 * opacity)
        draw.text(pos, text, font=font, fill=(255, 255, 255, alpha))

        # Composite
        result = Image.alpha_composite(img, watermark)

        output_file = Path(output_path).expanduser()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result.convert("RGB").save(output_file)

        return {
            "input_path": input_path,
            "output_path": str(output_file),
            "watermark_text": text,
            "position": position,
        }

    async def _get_info(self, path: str) -> dict[str, Any]:
        """Get image information."""
        from PIL import Image
        from PIL.ExifTags import TAGS

        img_path = Path(path).expanduser()
        img = Image.open(img_path)

        info = {
            "path": str(img_path),
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "width": img.width,
            "height": img.height,
            "file_size": img_path.stat().st_size,
        }

        # Extract EXIF data if available
        try:
            exif = img._getexif()
            if exif:
                info["exif"] = {
                    TAGS.get(k, k): str(v)
                    for k, v in exif.items()
                    if k in TAGS
                }
        except Exception:
            pass

        return info

    async def _thumbnail(
        self,
        input_path: str,
        output_path: str,
        size: int,
    ) -> dict[str, Any]:
        """Create a thumbnail."""
        from PIL import Image

        img = Image.open(Path(input_path).expanduser())
        img.thumbnail((size, size), Image.Resampling.LANCZOS)

        output_file = Path(output_path).expanduser()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_file)

        return {
            "input_path": input_path,
            "output_path": str(output_file),
            "size": img.size,
        }
