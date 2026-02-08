"""Personality adapter for emotional intelligence."""

from .sentiment import SentimentResult
from ..memory.user_profile import UserProfile
from ..utils.logging import get_logger

logger = get_logger(__name__)


class PersonalityAdapter:
    """Adapts system prompt tone based on sentiment, channel, and user profile."""

    _TONE_GUIDANCE = {
        "happy": "The user seems to be in a good mood. Match their energy — be upbeat and enthusiastic where appropriate.",
        "excited": "The user is excited! Share their enthusiasm and be energetic in your responses.",
        "sad": "The user seems down. Be warm, empathetic, and supportive. Acknowledge their feelings before diving into tasks.",
        "frustrated": "The user seems frustrated. Be patient, empathetic, and solution-focused. Validate their frustration, then help them move forward. Avoid being overly cheerful.",
        "anxious": "The user seems stressed or anxious. Be calm, reassuring, and structured. Break things down into manageable steps. Offer to help prioritize.",
        "neutral": "",  # No special guidance
    }

    _CHANNEL_GUIDANCE = {
        "slack": "This is a work/Slack channel. Keep responses professional but friendly. Use Slack-appropriate formatting.",
        "whatsapp": "This is a personal WhatsApp chat. Be more casual and conversational. Use shorter messages.",
        "web": "",  # Default, no special guidance
        "websocket": "",
    }

    def adapt_system_prompt(
        self,
        base_prompt: str,
        sentiment: SentimentResult,
        channel: str = "",
        user_profile: UserProfile | None = None,
    ) -> str:
        """Adapt the system prompt based on context."""
        additions = []

        # Add tone guidance based on sentiment
        tone = self._TONE_GUIDANCE.get(sentiment.mood, "")
        if tone and sentiment.confidence > 0.5:
            additions.append(f"\n## Current Tone\n{tone}")

        # Add channel-specific guidance
        channel_hint = self._CHANNEL_GUIDANCE.get(channel, "")
        if channel_hint:
            additions.append(f"\n## Channel Context\n{channel_hint}")

        # Add communication style from profile
        if user_profile and user_profile.communication_style:
            additions.append(
                f"\n## User Communication Style\nThe user prefers {user_profile.communication_style} communication."
            )

        if not additions:
            return base_prompt

        return base_prompt + "\n" + "\n".join(additions)
