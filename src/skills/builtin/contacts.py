"""Contacts skill for storing and searching contacts."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..base import BaseSkill, SkillResult
from ...utils.config import get_settings
from ...utils.logging import get_logger

logger = get_logger(__name__)


class ContactsSkill(BaseSkill):
    """Store and search contacts (name, phone, email, notes)."""

    name = "contacts"
    description = "Add, search, and manage contacts"
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._data_path: Path | None = None
        self._contacts: list[dict[str, Any]] = []

    def _get_data_path(self) -> Path:
        if self._data_path is None:
            settings = get_settings()
            base = Path(settings.aria.data_dir).expanduser()
            self._data_path = base / "contacts.json"
            self._data_path.parent.mkdir(parents=True, exist_ok=True)
        return self._data_path

    def _load(self) -> None:
        path = self._get_data_path()
        if path.exists():
            try:
                self._contacts = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._contacts = []
        else:
            self._contacts = []

    def _save(self) -> None:
        path = self._get_data_path()
        path.write_text(json.dumps(self._contacts, indent=2), encoding="utf-8")

    async def initialize(self) -> None:
        self._load()
        await super().initialize()

    def _register_capabilities(self) -> None:
        self.register_capability(
            name="add_contact",
            description="Add a new contact with name, phone, email, notes",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Contact name"},
                    "phone": {"type": "string", "description": "Phone number"},
                    "email": {"type": "string", "description": "Email address"},
                    "notes": {"type": "string", "description": "Additional notes"},
                },
                "required": ["name"],
            },
        )
        self.register_capability(
            name="find_contact",
            description="Search contacts by name, phone, or email",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term"},
                },
                "required": ["query"],
            },
        )
        self.register_capability(
            name="list_contacts",
            description="List all stored contacts",
            parameters={"type": "object", "properties": {}},
        )
        self.register_capability(
            name="update_contact",
            description="Update an existing contact by name",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "phone": {"type": "string"},
                    "email": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["name"],
            },
        )
        self.register_capability(
            name="delete_contact",
            description="Remove a contact by name",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        start = datetime.now(timezone.utc)
        self._load()

        if capability == "add_contact":
            return await self._add_contact(kwargs, start)
        elif capability == "find_contact":
            return await self._find_contact(kwargs.get("query", ""), start)
        elif capability == "list_contacts":
            return await self._list_contacts(start)
        elif capability == "update_contact":
            return await self._update_contact(kwargs, start)
        elif capability == "delete_contact":
            return await self._delete_contact(kwargs.get("name", ""), start)
        return self._error_result(f"Unknown capability: {capability}", start)

    async def _add_contact(self, kwargs: dict, start: datetime) -> SkillResult:
        name = kwargs.get("name", "").strip()
        if not name:
            return self._error_result("Name is required", start)
        contact = {
            "name": name,
            "phone": kwargs.get("phone", "").strip(),
            "email": kwargs.get("email", "").strip(),
            "notes": kwargs.get("notes", "").strip(),
        }
        self._contacts.append(contact)
        self._save()
        return self._success_result(f"Added contact: {name}", start)

    async def _find_contact(self, query: str, start: datetime) -> SkillResult:
        if not query:
            return self._success_result("No query provided", start)
        q = query.lower()
        matches = [
            c for c in self._contacts
            if q in c.get("name", "").lower()
            or q in c.get("phone", "")
            or q in c.get("email", "").lower()
        ]
        if not matches:
            return self._success_result(f"No contacts found for '{query}'", start)
        lines = [f"- {c['name']}: {c.get('phone', '')} {c.get('email', '')}" for c in matches[:10]]
        return self._success_result("\n".join(lines), start)

    async def _list_contacts(self, start: datetime) -> SkillResult:
        if not self._contacts:
            return self._success_result("No contacts stored.", start)
        lines = [f"- {c['name']}: {c.get('phone', '')} {c.get('email', '')}" for c in self._contacts[:50]]
        return self._success_result("\n".join(lines), start)

    async def _update_contact(self, kwargs: dict, start: datetime) -> SkillResult:
        name = kwargs.get("name", "").strip()
        for c in self._contacts:
            if c.get("name", "").lower() == name.lower():
                if "phone" in kwargs and kwargs["phone"]:
                    c["phone"] = kwargs["phone"]
                if "email" in kwargs and kwargs["email"]:
                    c["email"] = kwargs["email"]
                if "notes" in kwargs:
                    c["notes"] = kwargs["notes"]
                self._save()
                return self._success_result(f"Updated contact: {name}", start)
        return self._error_result(f"Contact not found: {name}", start)

    async def _delete_contact(self, name: str, start: datetime) -> SkillResult:
        for i, c in enumerate(self._contacts):
            if c.get("name", "").lower() == name.lower():
                self._contacts.pop(i)
                self._save()
                return self._success_result(f"Deleted contact: {name}", start)
        return self._error_result(f"Contact not found: {name}", start)
