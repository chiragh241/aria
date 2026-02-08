"""Built-in skills for Aria."""

from .filesystem import FilesystemSkill
from .shell import ShellSkill
from .browser import BrowserSkill
from .calendar import CalendarSkill
from .email import EmailSkill
from .sms import SMSSkill
from .tts import TTSSkill
from .stt import STTSkill
from .image import ImageSkill
from .video import VideoSkill
from .documents import DocumentsSkill

__all__ = [
    "FilesystemSkill",
    "ShellSkill",
    "BrowserSkill",
    "CalendarSkill",
    "EmailSkill",
    "SMSSkill",
    "TTSSkill",
    "STTSkill",
    "ImageSkill",
    "VideoSkill",
    "DocumentsSkill",
]
