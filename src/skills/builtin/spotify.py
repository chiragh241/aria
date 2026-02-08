"""Spotify integration skill — control playback, suggest playlists."""

import os
from typing import Any

from ..base import BaseSkill, SkillResult
from ...features.registry import is_feature_enabled
from ...utils.logging import get_logger

logger = get_logger(__name__)


class SpotifySkill(BaseSkill):
    """Control Spotify playback and suggest playlists. Requires SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET."""

    name = "spotify"
    description = "Play, pause, skip, search playlists; suggest music based on mood"
    version = "1.0.0"
    enabled = False

    def _get_config(self) -> dict[str, str]:
        return {
            "client_id": self.config.get("client_id") or os.environ.get("SPOTIFY_CLIENT_ID", ""),
            "client_secret": self.config.get("client_secret") or os.environ.get("SPOTIFY_CLIENT_SECRET", ""),
        }

    def _register_capabilities(self) -> None:
        self.register_capability(
            name="play",
            description="Play or resume playback",
            parameters={"type": "object", "properties": {"uri": {"type": "string", "description": "Track/album URI"}}},
        )
        self.register_capability(
            name="pause",
            description="Pause playback",
            parameters={"type": "object", "properties": {}},
        )
        self.register_capability(
            name="next",
            description="Skip to next track",
            parameters={"type": "object", "properties": {}},
        )
        self.register_capability(
            name="search",
            description="Search for tracks or playlists",
            parameters={"type": "object", "properties": {"query": {"type": "string"}, "type": {"type": "string", "enum": ["track", "playlist", "album"]}}}),
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        if not is_feature_enabled("spotify_integration"):
            return SkillResult(success=False, error="Spotify integration is disabled")
        cfg = self._get_config()
        if not cfg.get("client_id") or not cfg.get("client_secret"):
            return SkillResult(
                success=False,
                error="Spotify credentials not configured. Add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.",
            )
        return SkillResult(
            success=True,
            output="Spotify integration is configured. Full implementation coming soon.",
        )
