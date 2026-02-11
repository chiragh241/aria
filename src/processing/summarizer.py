"""Conversation summarizer for compressing old context."""

from typing import Any

from ..core.llm_router import LLMMessage, LLMRouter
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ConversationSummarizer:
    """Summarizes conversations to preserve context when trimming."""

    def __init__(self, llm_router: LLMRouter | None = None) -> None:
        self._llm_router = llm_router

    def set_llm_router(self, router: LLMRouter) -> None:
        self._llm_router = router

    async def summarize(self, messages: list[LLMMessage], max_tokens: int = 200) -> str:
        """Summarize a list of messages into a concise summary.

        Uses simple extraction when few messages (saves LLM tokens); uses LLM for longer conversations.
        Falls back to simple extraction if no LLM is available.
        """
        if not messages:
            return ""

        # Filter to user/assistant messages only
        conversation = [m for m in messages if m.role in ("user", "assistant")]
        if not conversation:
            return ""

        # For small conversation, use fast extractive summary (no LLM call)
        if len(conversation) <= 10:
            return self._simple_summarize(conversation, max_tokens)

        # Try LLM-based summary for longer conversations
        if self._llm_router:
            try:
                return await self._llm_summarize(conversation, max_tokens)
            except Exception as e:
                logger.debug("LLM summarize failed, using fallback", error=str(e))

        return self._simple_summarize(conversation, max_tokens)

    async def _llm_summarize(self, messages: list[LLMMessage], max_tokens: int) -> str:
        """Use LLM to create a summary."""
        # Build the conversation text
        conv_text = "\n".join(
            f"{m.role.upper()}: {m.content[:300]}" for m in messages[-20:]  # Last 20 messages
        )

        summary_messages = [
            LLMMessage(
                role="system",
                content="You are a conversation summarizer. Create a brief, factual summary of the key topics, decisions, and facts mentioned. Focus on information that would be useful to remember for future conversations. Be concise.",
            ),
            LLMMessage(
                role="user",
                content=f"Summarize this conversation in {max_tokens} tokens or less:\n\n{conv_text}",
            ),
        ]

        response = await self._llm_router.generate(
            messages=summary_messages,
            task_type=None,
        )
        return response.content.strip()

    def _simple_summarize(self, messages: list[LLMMessage], max_tokens: int) -> str:
        """Simple extractive summary without LLM."""
        topics = []
        for msg in messages:
            if msg.role == "user" and msg.content:
                # Take first sentence of each user message as a topic
                first_sentence = msg.content.split(".")[0].strip()
                if len(first_sentence) > 10 and first_sentence not in topics:
                    topics.append(first_sentence[:100])

        if not topics:
            return ""

        summary = "Previous conversation covered: " + "; ".join(topics[:10])
        return summary[:max_tokens * 4]  # Rough char estimate

    async def extract_action_items(self, messages: list[LLMMessage]) -> list[str]:
        """Extract action items from a conversation."""
        if not messages or not self._llm_router:
            return []

        conv_text = "\n".join(
            f"{m.role.upper()}: {m.content[:300]}"
            for m in messages[-20:]
            if m.role in ("user", "assistant")
        )

        try:
            extract_messages = [
                LLMMessage(
                    role="system",
                    content="Extract action items, tasks, or promises from this conversation. Return each on a new line. If none, return 'None'.",
                ),
                LLMMessage(role="user", content=conv_text),
            ]
            response = await self._llm_router.generate(messages=extract_messages)
            items = [
                line.strip().lstrip("- ").lstrip("* ")
                for line in response.content.strip().split("\n")
                if line.strip() and line.strip().lower() != "none"
            ]
            return items[:10]
        except Exception as e:
            logger.debug("Action item extraction failed", error=str(e))
            return []

    async def extract_key_decisions(self, messages: list[LLMMessage]) -> list[str]:
        """Extract key decisions made in a conversation."""
        if not messages or not self._llm_router:
            return []

        conv_text = "\n".join(
            f"{m.role.upper()}: {m.content[:300]}"
            for m in messages[-20:]
            if m.role in ("user", "assistant")
        )

        try:
            extract_messages = [
                LLMMessage(
                    role="system",
                    content="Extract key decisions or conclusions from this conversation. Return each on a new line. If none, return 'None'.",
                ),
                LLMMessage(role="user", content=conv_text),
            ]
            response = await self._llm_router.generate(messages=extract_messages)
            decisions = [
                line.strip().lstrip("- ").lstrip("* ")
                for line in response.content.strip().split("\n")
                if line.strip() and line.strip().lower() != "none"
            ]
            return decisions[:10]
        except Exception as e:
            logger.debug("Decision extraction failed", error=str(e))
            return []
