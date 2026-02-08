"""LLM usage and cost tracking for API calls and tokens."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class UsageEntry:
    """Single usage record."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    latency_ms: float = 0.0
    task_type: str = ""


# Approximate cost per 1M tokens (USD) - update as needed
COST_PER_1M = {
    ("anthropic", "claude-sonnet"): {"input": 3.0, "output": 15.0},
    ("anthropic", "claude-opus"): {"input": 15.0, "output": 75.0},
    ("anthropic", "claude-haiku"): {"input": 0.25, "output": 1.25},
    ("anthropic", "default"): {"input": 3.0, "output": 15.0},
    ("ollama", "default"): {"input": 0.0, "output": 0.0},
    ("openai", "default"): {"input": 2.0, "output": 6.0},
}


class UsageTracker:
    """Track LLM usage for cost and latency dashboards."""

    def __init__(self) -> None:
        self._entries: list[UsageEntry] = []
        self._max_entries = 10000
        self._file: Path | None = None
        self._load()

    def _load(self) -> None:
        """Load persisted usage from disk."""
        try:
            settings = get_settings()
            path = settings.get_data_path("usage.json")
            if path.exists():
                data = json.loads(path.read_text())
                self._entries = [
                    UsageEntry(
                        provider=e.get("provider", ""),
                        model=e.get("model", ""),
                        input_tokens=e.get("input_tokens", 0),
                        output_tokens=e.get("output_tokens", 0),
                        timestamp=e.get("timestamp", ""),
                        latency_ms=e.get("latency_ms", 0),
                        task_type=e.get("task_type", ""),
                    )
                    for e in data.get("entries", [])[-self._max_entries :]
                ]
                self._file = path
        except Exception as e:
            logger.debug("Could not load usage history", error=str(e))
            self._file = get_settings().get_data_path("usage.json")

    def _persist(self) -> None:
        """Persist usage to disk."""
        if not self._file:
            self._file = get_settings().get_data_path("usage.json")
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "entries": [
                    {
                        "provider": e.provider,
                        "model": e.model,
                        "input_tokens": e.input_tokens,
                        "output_tokens": e.output_tokens,
                        "timestamp": e.timestamp,
                        "latency_ms": e.latency_ms,
                        "task_type": e.task_type,
                    }
                    for e in self._entries[-self._max_entries :]
                ]
            }
            self._file.write_text(json.dumps(data, indent=0))
        except Exception as e:
            logger.warning("Could not persist usage", error=str(e))

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0,
        task_type: str = "",
    ) -> None:
        """Record a usage entry."""
        self._entries.append(
            UsageEntry(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                task_type=task_type,
            )
        )
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]
        self._persist()

    def get_stats(self) -> dict[str, Any]:
        """Get usage statistics for dashboard."""
        total_input = sum(e.input_tokens for e in self._entries)
        total_output = sum(e.output_tokens for e in self._entries)
        total_tokens = total_input + total_output

        by_provider: dict[str, dict[str, Any]] = {}
        for e in self._entries:
            if e.provider not in by_provider:
                by_provider[e.provider] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "calls": 0,
                    "latency_ms_total": 0.0,
                }
            by_provider[e.provider]["input_tokens"] += e.input_tokens
            by_provider[e.provider]["output_tokens"] += e.output_tokens
            by_provider[e.provider]["calls"] += 1
            by_provider[e.provider]["latency_ms_total"] += e.latency_ms

        cost_estimate = 0.0
        for prov, data in by_provider.items():
            key = (prov, "default")
            if prov == "anthropic":
                key = (prov, "claude-sonnet")
            rates = COST_PER_1M.get(key, COST_PER_1M.get((prov, "default"), {"input": 0, "output": 0}))
            cost_estimate += (data["input_tokens"] / 1e6) * rates["input"]
            cost_estimate += (data["output_tokens"] / 1e6) * rates["output"]

        latencies = [e.latency_ms for e in self._entries if e.latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "total_calls": len(self._entries),
            "cost_estimate_usd": round(cost_estimate, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "by_provider": by_provider,
        }


_usage_tracker: UsageTracker | None = None


def get_usage_tracker() -> UsageTracker:
    """Get singleton usage tracker."""
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = UsageTracker()
    return _usage_tracker
