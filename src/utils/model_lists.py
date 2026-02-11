"""Fetch dynamic model lists from LLM provider APIs.

Used by the setup wizard and Settings UI to show current models.
Fallback to static lists when API is unavailable or key not set.
"""

from __future__ import annotations

import os
from typing import Any

# Cache for one request (no in-memory long-lived cache to avoid stale data)
_anthropic_cache: list[dict[str, str]] | None = None
_openrouter_cache: list[dict[str, Any]] | None = None


def fetch_anthropic_models(api_key: str | None = None) -> list[dict[str, str]]:
    """Fetch Claude models from Anthropic API. Requires API key."""
    global _anthropic_cache
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return _anthropic_fallback()

    try:
        import httpx
        resp = httpx.get(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return _anthropic_fallback()
        data = resp.json()
        models = []
        for m in data.get("data", []):
            mid = m.get("id") or m.get("model_id")
            name = m.get("display_name") or mid or ""
            if mid:
                models.append({"id": mid, "name": name})
        if models:
            _anthropic_cache = models
            return models
    except Exception:
        pass
    return _anthropic_fallback()


def _anthropic_fallback() -> list[dict[str, str]]:
    """Static list when API unavailable. Update periodically from docs."""
    if _anthropic_cache:
        return _anthropic_cache
    return [
        {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4 (recommended)"},
        {"id": "claude-opus-4-5-20251101", "name": "Claude Opus 4.5"},
        {"id": "claude-3-5-haiku-20241022", "name": "Claude Haiku 3.5"},
        {"id": "claude-opus-4-6", "name": "Claude Opus 4.6"},
    ]


def fetch_openrouter_models(
    api_key: str | None = None,
    free_only: bool = False,
) -> list[dict[str, Any]]:
    """Fetch models from OpenRouter. Optional API key (list may work without)."""
    global _openrouter_cache
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    try:
        import httpx
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        resp = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=15,
        )
        if resp.status_code != 200:
            return _openrouter_fallback(free_only)
        data = resp.json()
        out = []
        for m in data.get("data", []):
            mid = m.get("id")
            if not mid:
                continue
            name = m.get("name") or mid
            pricing = m.get("pricing") or {}
            prompt_cost = str(pricing.get("prompt") or "0")
            completion_cost = str(pricing.get("completion") or "0")
            is_free = prompt_cost == "0" and completion_cost == "0"
            if free_only:
                if not is_free:
                    continue
            else:
                if is_free:
                    continue  # paid list: only paid models
            out.append({"id": mid, "name": name, "free": is_free})
        if out:
            _openrouter_cache = out
            return out
    except Exception:
        pass
    return _openrouter_fallback(free_only)


def _openrouter_fallback(free_only: bool) -> list[dict[str, Any]]:
    """Static OpenRouter list when API unavailable."""
    if _openrouter_cache and (not free_only or any(m.get("free") for m in _openrouter_cache)):
        return [m for m in _openrouter_cache if not free_only or m.get("free")] or _openrouter_cache
    paid = [
        {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet", "free": False},
        {"id": "anthropic/claude-3.5-haiku", "name": "Claude 3.5 Haiku", "free": False},
        {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini", "free": False},
        {"id": "openai/gpt-4o", "name": "GPT-4o", "free": False},
        {"id": "google/gemini-2.0-flash-001", "name": "Gemini 2.0 Flash", "free": False},
        {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Llama 3.3 70B", "free": False},
    ]
    free = [
        {"id": "openrouter/free", "name": "Free Models Router", "free": True},
    ]
    return free if free_only else paid


def fetch_gemini_models(api_key: str | None = None) -> list[dict[str, str]]:
    """Fetch Gemini models from Google Generative Language API. Requires API key."""
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        return _gemini_fallback()
    try:
        import httpx
        resp = httpx.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": key},
            timeout=15,
        )
        if resp.status_code != 200:
            return _gemini_fallback()
        data = resp.json()
        out = []
        for m in data.get("models", []):
            raw_name = m.get("name") or ""
            if not raw_name.startswith("models/"):
                continue
            mid = raw_name.replace("models/", "", 1)
            display = m.get("displayName") or mid
            out.append({"id": mid, "name": display})
        if out:
            return out
    except Exception:
        pass
    return _gemini_fallback()


def _gemini_fallback() -> list[dict[str, str]]:
    """Static Gemini list when API unavailable."""
    return [
        {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash"},
        {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
        {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"},
        {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro"},
        {"id": "gemini-1.5-flash-8b", "name": "Gemini 1.5 Flash 8B"},
    ]


def fetch_nvidia_models(api_key: str | None = None) -> list[dict[str, str]]:
    """Fetch NIM models from NVIDIA integrate API. GET /v1/models works without auth."""
    try:
        import httpx
        headers = {"Content-Type": "application/json"}
        if api_key or os.environ.get("NVIDIA_API_KEY"):
            key = api_key or os.environ.get("NVIDIA_API_KEY", "")
            headers["Authorization"] = f"Bearer {key}"
        resp = httpx.get(
            "https://integrate.api.nvidia.com/v1/models",
            headers=headers,
            timeout=15,
        )
        if resp.status_code != 200:
            return _nvidia_fallback()
        data = resp.json()
        out = []
        for m in data.get("data", []):
            mid = m.get("id")
            if not mid:
                continue
            # id is e.g. "moonshotai/kimi-k2-thinking"; use as name a readable label
            name = mid.split("/")[-1].replace("-", " ").title() if "/" in mid else mid
            out.append({"id": mid, "name": name})
        if out:
            return out
    except Exception:
        pass
    return _nvidia_fallback()


def _nvidia_fallback() -> list[dict[str, str]]:
    """Static list when API unavailable."""
    return [
        {"id": "moonshotai/kimi-k2-thinking", "name": "Kimi K2 Thinking (reasoning)"},
        {"id": "moonshotai/kimi-k2.5", "name": "Kimi K2.5 (Moonshot)"},
        {"id": "meta/llama-3.3-70b-instruct", "name": "Llama 3.3 70B"},
        {"id": "mistralai/mixtral-8x22b-instruct", "name": "Mixtral 8x22B"},
    ]


def fetch_openai_models(api_key: str | None = None) -> list[dict[str, str]]:
    """Fetch OpenAI chat models. Requires API key."""
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return _openai_fallback()
    try:
        import httpx
        resp = httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=15,
        )
        if resp.status_code != 200:
            return _openai_fallback()
        data = resp.json()
        out = []
        for m in data.get("data", []):
            mid = m.get("id")
            if not mid or mid.startswith("gpt-4") or mid.startswith("gpt-3.5") or "chat" in mid.lower():
                if mid and ("gpt-4" in mid or "gpt-3.5" in mid):
                    out.append({"id": mid, "name": m.get("id", mid)})
        if out:
            return sorted(out, key=lambda x: x["id"])
    except Exception:
        pass
    return _openai_fallback()


def _openai_fallback() -> list[dict[str, str]]:
    return [
        {"id": "gpt-4o", "name": "GPT-4o"},
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
    ]
