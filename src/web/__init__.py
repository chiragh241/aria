"""Web dashboard components for Aria."""

from .api import create_app
from .websocket import WebSocketManager

__all__ = ["create_app", "WebSocketManager"]
