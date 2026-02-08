"""Webhook skill for sending HTTP requests."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from ..base import BaseSkill, SkillResult
from ...utils.config import get_settings
from ...utils.logging import get_logger

logger = get_logger(__name__)


class WebhookSkill(BaseSkill):
    """Send HTTP requests to webhooks. Can save and reuse webhook shortcuts."""

    name = "webhook"
    description = "Send HTTP requests to URLs (webhooks)"
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._data_path: Path | None = None
        self._webhooks: dict[str, dict] = {}

    def _get_data_path(self) -> Path:
        if self._data_path is None:
            settings = get_settings()
            base = Path(settings.aria.data_dir).expanduser()
            self._data_path = base / "webhooks.json"
            self._data_path.parent.mkdir(parents=True, exist_ok=True)
        return self._data_path

    def _load(self) -> None:
        path = self._get_data_path()
        if path.exists():
            try:
                self._webhooks = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._webhooks = {}
        else:
            self._webhooks = {}

    def _save(self) -> None:
        path = self._get_data_path()
        path.write_text(json.dumps(self._webhooks, indent=2), encoding="utf-8")

    async def initialize(self) -> None:
        self._load()
        await super().initialize()

    def _register_capabilities(self) -> None:
        self.register_capability(
            name="fire",
            description="Send an HTTP request to a URL",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Target URL"},
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"], "default": "POST"},
                    "body": {"type": "object", "description": "JSON body (for POST/PUT)"},
                },
                "required": ["url"],
            },
        )
        self.register_capability(
            name="list_webhooks",
            description="List saved webhook shortcuts",
            parameters={"type": "object", "properties": {}},
        )
        self.register_capability(
            name="save_webhook",
            description="Save a named webhook for reuse",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "url": {"type": "string"},
                    "method": {"type": "string", "default": "POST"},
                },
                "required": ["name", "url"],
            },
        )
        self.register_capability(
            name="delete_webhook",
            description="Remove a saved webhook",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        start = datetime.now(timezone.utc)
        self._load()

        if capability == "fire":
            return await self._fire(kwargs, start)
        elif capability == "list_webhooks":
            return await self._list_webhooks(start)
        elif capability == "save_webhook":
            return await self._save_webhook(kwargs, start)
        elif capability == "delete_webhook":
            return await self._delete_webhook(kwargs.get("name", ""), start)
        return self._error_result(f"Unknown capability: {capability}", start)

    async def _fire(self, kwargs: dict, start: datetime) -> SkillResult:
        url = kwargs.get("url", "")
        if not url:
            return self._error_result("URL required", start)

        # Check if it's a saved webhook name
        if url in self._webhooks:
            wh = self._webhooks[url]
            url = wh.get("url", url)
            kwargs.setdefault("method", wh.get("method", "POST"))
            if "body" not in kwargs and "body" in wh:
                kwargs["body"] = wh["body"]

        method = (kwargs.get("method") or "POST").upper()
        body = kwargs.get("body")

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                if method == "GET":
                    resp = await client.get(url)
                elif method == "POST":
                    resp = await client.post(url, json=body if body else None)
                elif method == "PUT":
                    resp = await client.put(url, json=body if body else None)
                elif method == "DELETE":
                    resp = await client.delete(url)
                else:
                    return self._error_result(f"Unsupported method: {method}", start)

                return self._success_result(
                    f"Request sent. Status: {resp.status_code}",
                    start,
                )
        except Exception as e:
            return self._error_result(f"Request failed: {e}", start)

    async def _list_webhooks(self, start: datetime) -> SkillResult:
        if not self._webhooks:
            return self._success_result("No webhooks saved.", start)
        lines = [f"- {k}: {v.get('url', '')}" for k, v in self._webhooks.items()]
        return self._success_result("\n".join(lines), start)

    async def _save_webhook(self, kwargs: dict, start: datetime) -> SkillResult:
        name = kwargs.get("name", "").strip()
        url = kwargs.get("url", "").strip()
        if not name or not url:
            return self._error_result("name and url required", start)
        self._webhooks[name] = {
            "url": url,
            "method": kwargs.get("method", "POST"),
            "body": kwargs.get("body"),
        }
        self._save()
        return self._success_result(f"Saved webhook: {name}", start)

    async def _delete_webhook(self, name: str, start: datetime) -> SkillResult:
        if name in self._webhooks:
            del self._webhooks[name]
            self._save()
            return self._success_result(f"Deleted webhook: {name}", start)
        return self._error_result(f"Webhook not found: {name}", start)
