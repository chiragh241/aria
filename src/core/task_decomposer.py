"""Decomposes large tasks into parallel subtasks for multi-bot execution."""

import re
from dataclasses import dataclass
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Subtask:
    """A single subtask for parallel execution."""

    description: str
    entity: str  # e.g. "Paris", "Japan"
    entity_type: str  # e.g. "city", "country", "topic"
    index: int


@dataclass
class DecompositionResult:
    """Result of decomposing a task."""

    task_type: str  # e.g. "itinerary", "research", "compare"
    subtasks: list[Subtask]
    template: str  # e.g. "Research itinerary for {entity}"
    can_parallelize: bool


def _extract_entities_simple(text: str) -> list[str]:
    """Extract entity names (cities, countries) using heuristics."""
    entities: list[str] = []
    skip = {"the", "and", "for", "to", "in", "of", "with", "a", "an", "my", "our"}

    # "itinerary for Paris, London, Tokyo" - after "for", "to", "in"
    for prefix in (" for ", " to ", " in ", ":"):
        if prefix in text.lower():
            idx = text.lower().find(prefix)
            rest = text[idx + len(prefix) :].strip().lstrip(":")
            if rest:
                parts = re.split(r"\s*,\s*|\s+and\s+", rest)
                for p in parts:
                    p = p.strip().strip(".")
                    if p and len(p) >= 2 and len(p) < 50 and p.lower() not in skip:
                        entities.append(p)
            if entities:
                break

    # "Paris, London, Tokyo" - explicit comma/and list
    if not entities and re.search(r"[A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)*\s*,\s*", text):
        # Find the list part (often mid-sentence)
        parts = re.split(r"\s*,\s*|\s+and\s+", text)
        for p in parts:
            p = p.strip()
            # Must look like a place name (capitalized, reasonable length)
            if p and 2 <= len(p) <= 45 and (p[0].isupper() or p.isupper()) and p.lower() not in skip:
                entities.append(p)

    # Deduplicate while preserving order
    seen = set()
    out = []
    for e in entities:
        k = e.lower().strip()
        if k not in seen:
            seen.add(k)
            out.append(e.strip())

    return out[:10]  # Cap at 10 subtasks


def _detect_task_type(text: str) -> str:
    """Detect if task is itinerary, compare, research, etc."""
    t = text.lower()
    if any(kw in t for kw in ["itinerary", "trip", "travel", "visit", "tour", "itineraries"]):
        return "itinerary"
    if any(kw in t for kw in ["compare", "comparison", "vs", "versus", "different"]):
        return "compare"
    if any(kw in t for kw in ["research", "find", "search", "look up"]):
        return "research"
    return "general"


def decompose(
    task: str,
    llm_router: Any = None,
) -> DecompositionResult:
    """
    Decompose a task into parallel subtasks when possible.

    Returns DecompositionResult with subtasks when decomposition makes sense,
    or a single subtask when it doesn't.
    """
    task_type = _detect_task_type(task)
    entities = _extract_entities_simple(task)

    if not entities or len(entities) < 2:
        # Single subtask - no need to parallelize
        return DecompositionResult(
            task_type=task_type,
            subtasks=[Subtask(description=task, entity="", entity_type="", index=0)],
            template="",
            can_parallelize=False,
        )

    # Determine template based on task type
    if task_type == "itinerary":
        template = "Research 1-day itinerary and top things to do in {entity}"
    elif task_type == "compare":
        template = "Research and summarize key facts about {entity}"
    else:
        template = "Research {entity} for: " + task[:80]

    subtasks = []
    for i, entity in enumerate(entities):
        desc = template.format(entity=entity)
        entity_type = "city" if task_type == "itinerary" else "entity"
        subtasks.append(
            Subtask(description=desc, entity=entity, entity_type=entity_type, index=i)
        )

    return DecompositionResult(
        task_type=task_type,
        subtasks=subtasks,
        template=template,
        can_parallelize=len(subtasks) >= 2,
    )
