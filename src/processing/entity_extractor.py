"""Entity extraction from text using regex patterns."""

import re
from dataclasses import dataclass, field
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Entity:
    """An extracted entity."""

    type: str  # PERSON, DATE, LOCATION, PROJECT, ORGANIZATION, CONTACT_INFO
    value: str
    context: str = ""
    confidence: float = 0.8


class EntityExtractor:
    """Extract structured entities from text using regex patterns."""

    # Common name patterns (first + last name, or "my X's name is Y")
    _NAME_PATTERNS = [
        r"(?:my\s+(?:wife|husband|partner|girlfriend|boyfriend|friend|mom|dad|mother|father|brother|sister|son|daughter|boss|manager|colleague|coworker)(?:'s| is)\s+(?:name is\s+)?)([\w]+(?:\s+[\w]+)?)",
        r"(?:call me\s+)([\w]+)",
        r"(?:my name is\s+)([\w]+(?:\s+[\w]+)?)",
        r"(?:I'm\s+)([\w]+(?:\s+[\w]+)?)",
        r"(?:(?:wife|husband|partner|girlfriend|boyfriend|friend|mom|dad|mother|father|brother|sister|son|daughter|boss|manager|colleague)\s+(?:named|called)\s+)([\w]+(?:\s+[\w]+)?)",
    ]

    # Relationship patterns
    _RELATIONSHIP_PATTERNS = [
        (r"my\s+(wife|husband|partner|girlfriend|boyfriend|friend|mom|dad|mother|father|brother|sister|son|daughter|boss|manager|colleague|coworker)(?:'s|\s+is)\s+(?:name is\s+)?([\w]+(?:\s+[\w]+)?)", "relationship"),
        (r"my\s+(wife|husband|partner|girlfriend|boyfriend|friend|mom|dad|mother|father|brother|sister|son|daughter|boss|manager|colleague|coworker)\s+([\w]+(?:\s+[\w]+)?)(?:\s+and\s+|\s*[.,]|$)", "relationship"),
    ]

    # Date patterns
    _DATE_PATTERNS = [
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?)\b",
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+\d{4})?)\b",
    ]

    # Contact info patterns
    _CONTACT_PATTERNS = [
        (r"\b([\w.+-]+@[\w-]+\.[\w.-]+)\b", "email"),
        (r"\b(\+?1?\d{10,15})\b", "phone"),
        (r"\b(\(\d{3}\)\s*\d{3}[-.]?\d{4})\b", "phone"),
        (r"\b(\d{3}[-.]?\d{3}[-.]?\d{4})\b", "phone"),
    ]

    # Location patterns (basic)
    _LOCATION_PATTERNS = [
        r"(?:I\s+live\s+in|I'm\s+from|I'm\s+in|located\s+in|based\s+in)\s+([\w\s,]+?)(?:\.|,|$|\s+and\s+)",
        r"(?:my\s+(?:home|office|address)\s+is\s+(?:in\s+)?)([\w\s,]+?)(?:\.|,|$)",
    ]

    # Important date patterns
    _IMPORTANT_DATE_PATTERNS = [
        (r"(?:my\s+)?birthday\s+is\s+(?:on\s+)?(.+?)(?:\.|,|$)", "birthday"),
        (r"(?:my\s+)?anniversary\s+is\s+(?:on\s+)?(.+?)(?:\.|,|$)", "anniversary"),
    ]

    # Preference patterns — "I prefer X", "I like X", "use X", "always X", etc.
    _PREFERENCE_PATTERNS = [
        (r"(?:use\s+)(?:the\s+)?(metric|imperial|celsius|fahrenheit)(?:\s+units?)?(?:\.|,|$)",
         "units"),
        (r"(?:I\s+)?(?:prefer|like)\s+(?:to\s+)?(?:use\s+)?(metric|imperial|celsius|fahrenheit)(?:\s+units?)?(?:\.|,|$)",
         "units"),
        (r"(?:I\s+)?(?:am|'m)\s+(vegetarian|vegan|pescatarian)(?:\.|,|$)",
         "diet"),
        (r"(?:I\s+)?(?:prefer|like)\s+(short|brief|concise|detailed|long)\s+(?:responses?|replies?)(?:\.|,|$)",
         "response_style"),
        (r"(?:please\s+)?(?:always\s+)?(?:use\s+)(short|brief|concise|detailed|long)\s+(?:responses?|replies?)(?:\s+please)?(?:\.|,|$)",
         "response_style"),
        (r"(?:my\s+)?timezone\s+is\s+(?:['\"]?)([\w/]+)(?:['\"]?)(?:\.|,|$)",
         "timezone"),
        (r"(?:I\s+)?(?:speak|use)\s+([\w\-]+)(?:\s+as\s+my\s+language)?(?:\.|,|$)",
         "language"),
        (r"(?:I\s+)?(?:don't|do\s+not)\s+(?:like|want)\s+([\w\s]+?)(?:\.|,|$)",
         "dislike"),
        (r"(?:I\s+)?(?:work\s+)(?:in\s+)?(?:the\s+)?([\w\s]+?)(?:\s+industry|field)?(?:\.|,|$)",
         "occupation"),
        (r"(?:I\s+)?prefer\s+(?:to\s+)?(?:use\s+)?([\w\s]+?)(?:\s+over\s+[\w\s]+)?(?:\.|,|$)",
         "preference"),
        (r"(?:I\s+)?like\s+(?:to\s+)?(?:use\s+)?([\w\s]+?)(?:\.|,|$)", "preference"),
        (r"(?:please\s+)?(?:always\s+)?(?:use\s+)([\w\s]+?)(?:\.|,|$)", "preference"),
    ]

    def extract(self, text: str) -> list[Entity]:
        """Extract entities from text using regex patterns."""
        entities: list[Entity] = []
        text_lower = text.lower()

        # Extract names and relationships
        for pattern in self._NAME_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1).strip()
                if len(name) > 1 and not name.lower() in ("the", "a", "an", "is", "my"):
                    entities.append(Entity(
                        type="PERSON",
                        value=name,
                        context=match.group(0).strip(),
                        confidence=0.8,
                    ))

        # Extract relationships (person + their role)
        for pattern, _ in self._RELATIONSHIP_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                relationship = match.group(1).strip()
                name = match.group(2).strip()
                entities.append(Entity(
                    type="PERSON",
                    value=name,
                    context=f"{relationship}: {name}",
                    confidence=0.9,
                ))

        # Extract dates
        for pattern in self._DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    type="DATE",
                    value=match.group(1).strip(),
                    context=text[max(0, match.start()-20):match.end()+20].strip(),
                    confidence=0.9,
                ))

        # Extract important dates with labels
        for pattern, label in self._IMPORTANT_DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    type="DATE",
                    value=match.group(1).strip(),
                    context=f"{label}: {match.group(1).strip()}",
                    confidence=0.9,
                ))

        # Extract contact info
        for pattern, contact_type in self._CONTACT_PATTERNS:
            for match in re.finditer(pattern, text):
                entities.append(Entity(
                    type="CONTACT_INFO",
                    value=match.group(1).strip(),
                    context=contact_type,
                    confidence=0.95,
                ))

        # Extract locations
        for pattern in self._LOCATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                location = match.group(1).strip().rstrip(".")
                if len(location) > 1:
                    entities.append(Entity(
                        type="LOCATION",
                        value=location,
                        context=match.group(0).strip(),
                        confidence=0.7,
                    ))

        # Extract preferences
        for pattern, pref_key in self._PREFERENCE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = match.group(1).strip()
                if len(value) > 0 and len(value) < 80:
                    entities.append(Entity(
                        type="PREFERENCE",
                        value=value,
                        context=pref_key,
                        confidence=0.85,
                    ))

        return entities

    def extract_profile_updates(self, text: str) -> dict[str, Any]:
        """Extract profile-relevant updates from text.

        Returns a dict of profile fields to update.
        """
        updates: dict[str, Any] = {}
        entities = self.extract(text)

        for entity in entities:
            if entity.type == "PERSON":
                # Check if it's the user's own name
                if any(p in entity.context.lower() for p in ["call me", "my name is", "i'm"]):
                    updates["preferred_name"] = entity.value
                # Check if it's a relationship
                elif ":" in entity.context:
                    rel, name = entity.context.split(":", 1)
                    if "important_people" not in updates:
                        updates["important_people"] = {}
                    updates["important_people"][name.strip()] = rel.strip()

            elif entity.type == "LOCATION":
                if "preferences" not in updates:
                    updates["preferences"] = {}
                updates["preferences"]["location"] = entity.value

            elif entity.type == "DATE" and ":" in entity.context:
                label, date_val = entity.context.split(":", 1)
                if "important_dates" not in updates:
                    updates["important_dates"] = {}
                updates["important_dates"][label.strip()] = date_val.strip()

            elif entity.type == "PREFERENCE":
                pref_key = entity.context.lower().replace(" ", "_")
                if pref_key == "timezone":
                    updates["timezone"] = entity.value
                elif pref_key == "language":
                    updates["language"] = entity.value[:10]  # e.g. "en", "fr"
                else:
                    if "preferences" not in updates:
                        updates["preferences"] = {}
                    if pref_key == "preference":
                        key = "pref_" + entity.value.lower().replace(" ", "_")[:30]
                    else:
                        key = pref_key
                    updates["preferences"][key] = entity.value

        return updates

    def extract_facts_from_conversation(self, user_text: str, assistant_text: str = "") -> list[str]:
        """
        Extract facts to remember from the user message.
        Used for automatic storing without explicit 'remember' skill call.
        """
        facts: list[str] = []

        # User states facts about themselves
        fact_patterns = [
            (r"remember\s+that\s+(?:I\s+)?([^.!?]+?)(?:\.|!|\?|$)", 1),
            (r"(?:I(?:'m| am)\s+)(?:a\s+)?([\w\s]{3,50}?)(?:\.|,|$)", 1),
            (r"my\s+([\w\s]+?)\s+is\s+([^.!?,]{2,40}?)(?:\.|,|$)", 2),  # "my job is X" -> "job: X"
            (r"note\s+that\s+(?:I\s+)?([^.!?]+?)(?:\.|!|\?|$)", 1),
        ]
        for pat, group_idx in fact_patterns:
            for m in re.finditer(pat, user_text, re.IGNORECASE):
                if group_idx == 2:
                    fact = f"{m.group(1).strip()}: {m.group(2).strip()}"
                else:
                    fact = m.group(1).strip()
                if 5 <= len(fact) <= 120 and fact not in facts:
                    facts.append(fact)

        return facts[:3]  # Limit to 3 per exchange to avoid noise
