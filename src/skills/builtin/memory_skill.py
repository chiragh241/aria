"""Memory skill for explicit remember/recall/forget operations."""

from datetime import datetime, timezone
from typing import Any

from ..base import BaseSkill, SkillResult
from ...utils.logging import get_logger

logger = get_logger(__name__)


class MemorySkill(BaseSkill):
    """Skill for explicit memory management — remember, recall, forget facts."""

    name = "memory"
    description = "Remember facts, recall memories, and manage what the assistant knows about you"
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._profile_manager = None
        self._vector_memory = None

    def set_profile_manager(self, pm: Any) -> None:
        self._profile_manager = pm

    def set_vector_memory(self, vm: Any) -> None:
        self._vector_memory = vm

    def _register_capabilities(self) -> None:
        self.register_capability(
            name="remember",
            description="Store a fact or piece of information about the user. Use this when the user says 'remember that...' or shares personal information they want you to retain.",
            parameters={
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact to remember, e.g. 'My wife's name is Sarah' or 'I prefer dark mode'",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "The user ID (auto-filled from context)",
                    },
                },
                "required": ["fact"],
            },
        )
        self.register_capability(
            name="recall",
            description="Search memory for facts about the user or past conversations. Use when the user asks 'what do you know about...' or 'do you remember...'",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in memory",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "The user ID (auto-filled from context)",
                    },
                },
                "required": ["query"],
            },
        )
        self.register_capability(
            name="forget",
            description="Remove a specific stored fact by index or by query. Use fact_index for exact index, or query to remove all facts matching a phrase.",
            parameters={
                "type": "object",
                "properties": {
                    "fact_index": {
                        "type": "integer",
                        "description": "The index of the fact to remove (0-based). Use -1 if using query instead.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Remove all facts containing this phrase (e.g. 'wife' removes 'My wife is Sarah')",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "The user ID",
                    },
                },
            },
        )
        self.register_capability(
            name="list_memories",
            description="List all stored facts and profile information for the user",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID",
                    },
                },
            },
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        start = datetime.now(timezone.utc)

        if capability == "remember":
            return await self._remember(kwargs.get("fact", ""), kwargs.get("user_id", "default"), start)
        elif capability == "recall":
            return await self._recall(kwargs.get("query", ""), kwargs.get("user_id", "default"), start)
        elif capability == "forget":
            query = kwargs.get("query", "")
            fact_index = kwargs.get("fact_index", -1)
            return await self._forget(fact_index, query, kwargs.get("user_id", "default"), start)
        elif capability == "list_memories":
            return await self._list_memories(kwargs.get("user_id", "default"), start)
        else:
            return self._error_result(f"Unknown capability: {capability}", start)

    async def _remember(self, fact: str, user_id: str, start: datetime) -> SkillResult:
        if not fact:
            return self._error_result("No fact provided to remember", start)

        if self._profile_manager:
            self._profile_manager.add_fact(user_id, fact)

        # Also store in vector memory for semantic search
        if self._vector_memory and self._vector_memory.available:
            try:
                await self._vector_memory.add_document(
                    content=fact,
                    metadata={"type": "user_fact", "user_id": user_id},
                )
            except Exception:
                pass

        return self._success_result(f"Remembered: {fact}", start)

    async def _recall(self, query: str, user_id: str, start: datetime) -> SkillResult:
        results = []

        # Search profile facts
        if self._profile_manager:
            profile = self._profile_manager.get_profile(user_id)
            matching_facts = [
                f for f in profile.facts
                if query.lower() in f.lower()
            ]
            if matching_facts:
                results.append("Profile facts:\n" + "\n".join(f"  - {f}" for f in matching_facts))

            # Search important people
            for name, rel in profile.important_people.items():
                if query.lower() in name.lower() or query.lower() in rel.lower():
                    results.append(f"Person: {name} ({rel})")

        # Search vector memory
        if self._vector_memory and self._vector_memory.available:
            try:
                memories = await self._vector_memory.search(query, top_k=5)
                if memories:
                    memory_lines = []
                    for m in memories:
                        if m.get("score", 0) > 0.5:
                            memory_lines.append(f"  - {m['content'][:200]}")
                    if memory_lines:
                        results.append("Related memories:\n" + "\n".join(memory_lines))
            except Exception:
                pass

        if not results:
            return self._success_result(f"No memories found matching '{query}'", start)

        return self._success_result("\n\n".join(results), start)

    async def _forget(self, fact_index: int, query: str, user_id: str, start: datetime) -> SkillResult:
        if not self._profile_manager:
            return self._error_result("Profile manager not available", start)

        if query:
            _, count = self._profile_manager.remove_fact_by_query(user_id, query)
            return self._success_result(f"Forgot {count} fact(s) matching '{query}'", start)
        if fact_index >= 0:
            if self._profile_manager.remove_fact(user_id, fact_index):
                return self._success_result(f"Forgot fact at index {fact_index}", start)
            return self._error_result(f"Invalid fact index: {fact_index}", start)
        return self._error_result("Provide fact_index or query to forget", start)

    async def _list_memories(self, user_id: str, start: datetime) -> SkillResult:
        if not self._profile_manager:
            return self._success_result("No profile manager available", start)

        profile = self._profile_manager.get_profile(user_id)
        parts = []

        if profile.preferred_name or profile.name:
            parts.append(f"Name: {profile.preferred_name or profile.name}")
        if profile.timezone:
            parts.append(f"Timezone: {profile.timezone}")
        if profile.interests:
            parts.append(f"Interests: {', '.join(profile.interests)}")
        if profile.important_people:
            people = [f"{n} ({r})" for n, r in profile.important_people.items()]
            parts.append(f"Important people: {', '.join(people)}")
        if profile.important_dates:
            dates = [f"{l}: {d}" for l, d in profile.important_dates.items()]
            parts.append(f"Important dates: {', '.join(dates)}")
        if profile.facts:
            parts.append("Stored facts:")
            for i, fact in enumerate(profile.facts):
                parts.append(f"  [{i}] {fact}")

        if not parts:
            return self._success_result("No memories stored yet.", start)

        return self._success_result("\n".join(parts), start)
