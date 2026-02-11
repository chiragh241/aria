"""Tests for context_efficiency helpers."""

from src.core.context_efficiency import (
    normalize_turns,
    limit_history_by_turns,
    truncate_tool_result_text,
    rag_head_tail_trim,
    is_likely_context_overflow_error,
)
from src.core.llm_router import LLMMessage


def test_normalize_turns_merges_consecutive_same_role():
    msgs = [
        LLMMessage(role="user", content="a"),
        LLMMessage(role="user", content="b"),
        LLMMessage(role="assistant", content="c"),
    ]
    out = normalize_turns(msgs)
    assert len(out) == 2
    assert out[0].role == "user" and "a" in (out[0].content or "") and "b" in (out[0].content or "")
    assert out[1].role == "assistant" and out[1].content == "c"


def test_limit_history_by_turns_keeps_system():
    msgs = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="u1"),
        LLMMessage(role="assistant", content="a1"),
    ]
    out = limit_history_by_turns(msgs, max_turns=1)
    assert any(m.role == "system" for m in out)
    assert out[0].content == "sys"


def test_truncate_tool_result_text():
    long_text = "x" * 5000
    # Use max_chars above MIN_KEEP_CHARS (2000) so truncation applies
    out = truncate_tool_result_text(long_text, max_chars=2500)
    assert len(out) <= 2520
    assert "[truncated]" in out
    assert out.startswith("x")


def test_rag_head_tail_trim():
    text = "a" * 500 + "b" * 500
    out = rag_head_tail_trim(text, max_chars=200, head_ratio=0.5, tail_ratio=0.5)
    assert len(out) <= 250
    assert "a" in out and "b" in out


def test_is_likely_context_overflow_error():
    assert is_likely_context_overflow_error(Exception("context length exceeded")) is True
    assert is_likely_context_overflow_error(Exception("rate limit")) is False
