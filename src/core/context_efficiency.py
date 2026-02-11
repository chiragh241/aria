"""Context efficiency: turn normalization, token limits, proportional truncation."""

from __future__ import annotations

from typing import Any, Callable

from ..utils.config import get_settings
from ..utils.logging import get_logger
from .llm_router import LLMMessage

logger = get_logger(__name__)

# Max share of context window a single tool result should occupy
MAX_TOOL_RESULT_CONTEXT_SHARE = 0.3
CHARS_PER_TOKEN_ESTIMATE = 4
MIN_KEEP_CHARS = 2000
TRUNCATION_SUFFIX = "\n\n... [truncated]"


def estimate_tokens_messages(messages: list[LLMMessage], count_tokens: Callable[[str], int] | None = None) -> int:
    """Estimate total token count for a list of messages."""
    if count_tokens:
        total = 0
        for m in messages:
            total += count_tokens(m.content or "")
        return total
    total_chars = sum(len(m.content or "") for m in messages)
    return total_chars // CHARS_PER_TOKEN_ESTIMATE


def normalize_turns(messages: list[LLMMessage]) -> list[LLMMessage]:
    """
    Merge consecutive same-role messages (Anthropic/Gemini expect alternating user/assistant).
    Reduces message count and token usage.
    """
    if not messages:
        return []
    out: list[LLMMessage] = []
    for m in messages:
        if not m.role or not m.content:
            out.append(m)
            continue
        if out and out[-1].role == m.role:
            # Merge with previous
            prev = out[-1]
            combined = (prev.content or "") + "\n\n" + (m.content or "")
            out[-1] = LLMMessage(
                role=prev.role,
                content=combined.strip(),
                name=prev.name or m.name,
                tool_calls=prev.tool_calls or m.tool_calls,
                tool_call_id=prev.tool_call_id or m.tool_call_id,
            )
        else:
            out.append(m)
    return out


def limit_history_by_turns(
    messages: list[LLMMessage],
    max_turns: int,
) -> list[LLMMessage]:
    """
    Keep only the last N user turns and their associated assistant/tool replies.
    System message(s) at the start are always kept. If max_turns <= 0, return messages unchanged.
    """
    if max_turns <= 0 or not messages:
        return list(messages)
    system_msgs = [m for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]
    if not rest:
        return list(messages)
    user_count = 0
    start_index = len(rest)
    for i in range(len(rest) - 1, -1, -1):
        if rest[i].role == "user":
            user_count += 1
            if user_count > max_turns:
                return system_msgs + rest[start_index:]
            start_index = i
    return system_msgs + rest[start_index:]


def get_history_limit_for_channel(channel: str, user_id: str) -> int | None:
    """Return per-channel history turn limit if configured, else None (use default max_turns)."""
    settings = get_settings()
    limits = getattr(settings.memory, "channel_history_limits", None) or {}
    if not limits:
        return None
    # Keys can be "web", "slack", "slack:dm", or "channel:user_id"
    key = channel
    if key in limits:
        return limits[key]
    key2 = f"{channel}:{user_id}"
    if key2 in limits:
        return limits[key2]
    # Try base channel (e.g. "slack" when channel is "slack:dm:U123")
    base = channel.split(":")[0] if ":" in channel else channel
    if base in limits:
        return limits[base]
    return None


def proportional_max_tool_result_chars(context_window_tokens: int) -> int:
    """Max chars for a single tool result (30% of context window, ~4 chars/token)."""
    settings = get_settings()
    max_chars = getattr(settings.memory.short_term, "max_tool_result_chars", 4000)
    if not getattr(settings.memory.short_term, "use_proportional_tool_result", True):
        return max_chars
    share_tokens = int(context_window_tokens * MAX_TOOL_RESULT_CONTEXT_SHARE)
    proportional = share_tokens * CHARS_PER_TOKEN_ESTIMATE
    return min(max_chars, max(proportional, MIN_KEEP_CHARS))


def truncate_tool_result_text(text: str, max_chars: int) -> str:
    """Truncate tool result preserving the beginning; try to cut at newline."""
    if len(text) <= max_chars:
        return text
    keep = max(MIN_KEEP_CHARS, max_chars - len(TRUNCATION_SUFFIX))
    cut = keep
    last_nl = text.rfind("\n", 0, keep + 1)
    if last_nl > keep * 0.8:
        cut = last_nl
    return text[:cut] + TRUNCATION_SUFFIX


def cap_tool_result_for_persist(content: str) -> str:
    """Cap tool result content when persisting to disk (hard limit)."""
    settings = get_settings()
    max_chars = getattr(
        settings.memory.short_term,
        "max_tool_result_chars_at_persist",
        400_000,
    )
    if len(content) <= max_chars:
        return content
    return truncate_tool_result_text(content, max_chars)


def rag_head_tail_trim(text: str, max_chars: int, head_ratio: float = 0.7, tail_ratio: float = 0.2) -> str:
    """Trim long injected context to max_chars keeping head and tail."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head_c = int(max_chars * head_ratio)
    tail_c = int(max_chars * tail_ratio)
    head = text[:head_c]
    tail = text[-tail_c:] if tail_c > 0 else ""
    sep = "\n\n... [trimmed] ...\n\n"
    return head + sep + tail


def is_likely_context_overflow_error(exc: BaseException) -> bool:
    """Heuristic: API error likely due to context window overflow."""
    msg = str(exc).lower()
    return (
        "context" in msg
        and ("overflow" in msg or "length" in msg or "too long" in msg or "maximum" in msg or "limit" in msg)
    ) or "input tokens" in msg or "token limit" in msg


def has_oversized_tool_results(
    messages: list[LLMMessage],
    context_window_tokens: int,
    max_share: float = MAX_TOOL_RESULT_CONTEXT_SHARE,
) -> bool:
    """True if any tool message exceeds the proportional limit (used for truncate-and-retry)."""
    max_chars = int(context_window_tokens * max_share) * CHARS_PER_TOKEN_ESTIMATE
    for m in messages:
        if m.role == "tool" and m.content and len(m.content) > max_chars:
            return True
    return False


def apply_tool_result_cap_to_messages(
    messages: list[LLMMessage],
    context_window_tokens: int,
) -> list[LLMMessage]:
    """Return a new list with tool result contents truncated to proportional limit (in-memory, for retry)."""
    max_chars = proportional_max_tool_result_chars(context_window_tokens)
    out: list[LLMMessage] = []
    for m in messages:
        if m.role == "tool" and m.content and len(m.content) > max_chars:
            out.append(LLMMessage(role="tool", content=truncate_tool_result_text(m.content, max_chars), tool_call_id=m.tool_call_id))
        else:
            out.append(m)
    return out


def apply_tool_result_cap_to_context_in_place(messages: list[LLMMessage], context_window_tokens: int) -> None:
    """Mutate messages list: truncate tool result contents to proportional limit (for overflow retry)."""
    max_chars = proportional_max_tool_result_chars(context_window_tokens)
    for i, m in enumerate(messages):
        if m.role == "tool" and m.content and len(m.content) > max_chars:
            messages[i] = LLMMessage(role="tool", content=truncate_tool_result_text(m.content, max_chars), tool_call_id=m.tool_call_id)
