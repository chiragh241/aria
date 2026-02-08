"""User profile system for persistent long-term memory about users."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class UserProfile:
    """Persistent profile for a user."""

    user_id: str = ""
    name: str = ""
    preferred_name: str = ""
    timezone: str = ""
    language: str = "en"
    interests: list[str] = field(default_factory=list)
    communication_style: str = ""  # e.g. "casual", "formal", "technical"
    important_people: dict[str, str] = field(default_factory=dict)  # name -> relationship
    important_dates: dict[str, str] = field(default_factory=dict)  # label -> date
    preferences: dict[str, Any] = field(default_factory=dict)
    facts: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "preferred_name": self.preferred_name,
            "timezone": self.timezone,
            "language": self.language,
            "interests": self.interests,
            "communication_style": self.communication_style,
            "important_people": self.important_people,
            "important_dates": self.important_dates,
            "preferences": self.preferences,
            "facts": self.facts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserProfile":
        return cls(
            user_id=data.get("user_id", ""),
            name=data.get("name", ""),
            preferred_name=data.get("preferred_name", ""),
            timezone=data.get("timezone", ""),
            language=data.get("language", "en"),
            interests=data.get("interests", []),
            communication_style=data.get("communication_style", ""),
            important_people=data.get("important_people", {}),
            important_dates=data.get("important_dates", {}),
            preferences=data.get("preferences", {}),
            facts=data.get("facts", []),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
        )


class UserProfileManager:
    """Manages persistent user profiles."""

    def __init__(self, data_dir: str | None = None) -> None:
        settings = get_settings()
        base = Path(data_dir or settings.aria.data_dir).expanduser()
        self._profiles_dir = base / "user_profiles"
        self._profiles_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, UserProfile] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all profiles from disk."""
        for path in self._profiles_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                profile = UserProfile.from_dict(data)
                self._cache[profile.user_id] = profile
            except Exception as e:
                logger.warning("Failed to load user profile", path=str(path), error=str(e))

    def _persist(self, profile: UserProfile) -> None:
        """Save profile to disk."""
        safe_id = profile.user_id.replace(":", "_").replace("/", "_").replace("@", "_at_")
        path = self._profiles_dir / f"{safe_id}.json"
        try:
            path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to persist user profile", user_id=profile.user_id, error=str(e))

    def get_profile(self, user_id: str) -> UserProfile:
        """Get or create a user profile."""
        if user_id not in self._cache:
            self._cache[user_id] = UserProfile(user_id=user_id)
            self._persist(self._cache[user_id])
        return self._cache[user_id]

    def update_profile(self, user_id: str, **fields: Any) -> UserProfile:
        """Update specific fields on a user profile."""
        profile = self.get_profile(user_id)
        for key, value in fields.items():
            if hasattr(profile, key):
                current = getattr(profile, key)
                if isinstance(current, dict) and isinstance(value, dict):
                    current.update(value)
                elif isinstance(current, list) and isinstance(value, list):
                    for item in value:
                        if item not in current:
                            current.append(item)
                else:
                    setattr(profile, key, value)
        profile.updated_at = datetime.now(timezone.utc).isoformat()
        self._persist(profile)
        return profile

    def add_fact(self, user_id: str, fact: str) -> None:
        """Add a fact to the user's profile."""
        profile = self.get_profile(user_id)
        # Avoid duplicates
        if fact not in profile.facts:
            profile.facts.append(fact)
            profile.updated_at = datetime.now(timezone.utc).isoformat()
            self._persist(profile)

    def remove_fact(self, user_id: str, fact_index: int) -> bool:
        """Remove a fact by index."""
        profile = self.get_profile(user_id)
        if 0 <= fact_index < len(profile.facts):
            profile.facts.pop(fact_index)
            profile.updated_at = datetime.now(timezone.utc).isoformat()
            self._persist(profile)
            return True
        return False

    def remove_fact_by_query(self, user_id: str, query: str) -> tuple[bool, int]:
        """Remove facts matching query (substring match). Returns (success, count_removed)."""
        profile = self.get_profile(user_id)
        q = query.lower()
        to_remove = [i for i, f in enumerate(profile.facts) if q in f.lower()]
        for i in reversed(to_remove):
            profile.facts.pop(i)
        if to_remove:
            profile.updated_at = datetime.now(timezone.utc).isoformat()
            self._persist(profile)
        return True, len(to_remove)

    def get_context_for_llm(self, user_id: str) -> str:
        """Format profile as context string for the LLM system prompt."""
        profile = self.get_profile(user_id)
        parts = []

        if profile.preferred_name or profile.name:
            parts.append(f"User's name: {profile.preferred_name or profile.name}")
        if profile.timezone:
            parts.append(f"Timezone: {profile.timezone}")
        if profile.language and profile.language != "en":
            parts.append(f"Language: {profile.language}")
        if profile.communication_style:
            parts.append(f"Communication style: {profile.communication_style}")
        if profile.interests:
            parts.append(f"Interests: {', '.join(profile.interests)}")
        if profile.important_people:
            people = [f"{name} ({rel})" for name, rel in profile.important_people.items()]
            parts.append(f"Important people: {', '.join(people)}")
        if profile.important_dates:
            dates = [f"{label}: {date}" for label, date in profile.important_dates.items()]
            parts.append(f"Important dates: {', '.join(dates)}")
        if profile.preferences:
            prefs = [f"{k}: {v}" for k, v in profile.preferences.items()]
            parts.append(f"Preferences: {', '.join(prefs)}")
        if profile.facts:
            parts.append("Known facts about the user:")
            for fact in profile.facts[-20:]:  # Last 20 facts
                parts.append(f"  - {fact}")

        if not parts:
            return ""

        return "\n\n## User Profile\n" + "\n".join(parts) + "\n"

    def list_profiles(self) -> list[dict[str, Any]]:
        """List all profiles."""
        return [p.to_dict() for p in self._cache.values()]
