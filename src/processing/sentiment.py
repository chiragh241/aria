"""Sentiment analysis for emotional intelligence."""

from dataclasses import dataclass, field
from collections import deque
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""

    polarity: float = 0.0  # -1 (negative) to 1 (positive)
    mood: str = "neutral"  # happy, sad, frustrated, neutral, excited, anxious
    confidence: float = 0.5
    keywords: list[str] = field(default_factory=list)


class SentimentAnalyzer:
    """Lightweight keyword-based sentiment analyzer with mood tracking."""

    _MOOD_KEYWORDS: dict[str, list[str]] = {
        "happy": [
            "happy", "great", "awesome", "wonderful", "fantastic", "love",
            "excellent", "amazing", "perfect", "glad", "pleased", "delighted",
            "thrilled", "yay", "hooray", "excited", "joy", "celebrate",
        ],
        "sad": [
            "sad", "unhappy", "depressed", "miserable", "heartbroken", "crying",
            "tears", "grief", "loss", "mourning", "miss", "lonely", "blue",
            "down", "gloomy", "sorrow",
        ],
        "frustrated": [
            "frustrated", "annoyed", "angry", "furious", "irritated", "mad",
            "rage", "pissed", "hate", "terrible", "awful", "worst",
            "broken", "doesn't work", "not working", "failing", "stuck",
            "bug", "error", "crash", "wtf", "ugh", "damn", "crap",
        ],
        "excited": [
            "excited", "can't wait", "pumped", "stoked", "eager", "hyped",
            "looking forward", "thrilling", "incredible", "mind-blowing",
            "game-changer", "breakthrough",
        ],
        "anxious": [
            "anxious", "worried", "nervous", "scared", "afraid", "fearful",
            "concerned", "uneasy", "panic", "stress", "stressed", "overwhelmed",
            "deadline", "urgent", "help me", "sos", "emergency",
        ],
    }

    _POLARITY_MAP = {
        "happy": 0.7,
        "excited": 0.8,
        "neutral": 0.0,
        "anxious": -0.4,
        "sad": -0.7,
        "frustrated": -0.8,
    }

    def __init__(self) -> None:
        self._mood_history: dict[str, deque] = {}  # user_id -> rolling window of moods

    def analyze(self, text: str, user_id: str | None = None) -> SentimentResult:
        """Analyze sentiment of text using keyword matching."""
        if not text:
            return SentimentResult()

        text_lower = text.lower()
        words = set(text_lower.split())

        # Count keyword matches per mood
        scores: dict[str, float] = {}
        matched_keywords: list[str] = []

        for mood, keywords in self._MOOD_KEYWORDS.items():
            count = 0
            for kw in keywords:
                if " " in kw:
                    # Multi-word keyword
                    if kw in text_lower:
                        count += 1
                        matched_keywords.append(kw)
                else:
                    if kw in words:
                        count += 1
                        matched_keywords.append(kw)
            if count > 0:
                scores[mood] = count

        # Determine dominant mood
        if not scores:
            mood = "neutral"
            confidence = 0.5
        else:
            mood = max(scores, key=scores.get)
            total_matches = sum(scores.values())
            confidence = min(0.95, 0.5 + (scores[mood] / max(total_matches, 1)) * 0.4)

        polarity = self._POLARITY_MAP.get(mood, 0.0)

        result = SentimentResult(
            polarity=polarity,
            mood=mood,
            confidence=confidence,
            keywords=matched_keywords[:5],
        )

        # Track mood history per user
        if user_id:
            if user_id not in self._mood_history:
                self._mood_history[user_id] = deque(maxlen=10)
            self._mood_history[user_id].append(mood)

        return result

    def get_mood_trend(self, user_id: str) -> list[str]:
        """Get recent mood history for a user."""
        if user_id in self._mood_history:
            return list(self._mood_history[user_id])
        return []

    def get_dominant_mood(self, user_id: str) -> str:
        """Get the most common recent mood for a user."""
        history = self.get_mood_trend(user_id)
        if not history:
            return "neutral"
        from collections import Counter
        return Counter(history).most_common(1)[0][0]
