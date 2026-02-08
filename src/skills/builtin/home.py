"""Home Assistant skill for smart home control."""

import os
from datetime import datetime, timezone
from typing import Any

import httpx

from ..base import BaseSkill, SkillResult
from ...utils.logging import get_logger

logger = get_logger(__name__)


class HomeSkill(BaseSkill):
    """Control smart home devices via Home Assistant REST API."""

    name = "home"
    description = "Control lights, switches, thermostat via Home Assistant"
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._base_url = os.environ.get("HA_URL", "").rstrip("/")
        self._token = os.environ.get("HA_TOKEN", "")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def _available(self) -> bool:
        return bool(self._base_url and self._token)

    def _register_capabilities(self) -> None:
        self.register_capability(
            name="list_devices",
            description="List all Home Assistant entities (lights, switches, sensors)",
            parameters={"type": "object", "properties": {}},
        )
        self.register_capability(
            name="get_state",
            description="Get the current state of an entity",
            parameters={
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string", "description": "Entity ID (e.g. light.living_room)"},
                },
                "required": ["entity_id"],
            },
        )
        self.register_capability(
            name="turn_on",
            description="Turn on a light, switch, or other entity",
            parameters={
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                    "brightness": {"type": "integer", "description": "0-255"},
                },
                "required": ["entity_id"],
            },
        )
        self.register_capability(
            name="turn_off",
            description="Turn off a light or switch",
            parameters={
                "type": "object",
                "properties": {"entity_id": {"type": "string"}},
                "required": ["entity_id"],
            },
        )
        self.register_capability(
            name="run_scene",
            description="Activate a Home Assistant scene",
            parameters={
                "type": "object",
                "properties": {"scene_id": {"type": "string"}},
                "required": ["scene_id"],
            },
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        start = datetime.now(timezone.utc)
        if not self._available():
            return self._error_result(
                "Home Assistant not configured. Set HA_URL and HA_TOKEN.",
                start,
            )

        if capability == "list_devices":
            return await self._list_devices(start)
        elif capability == "get_state":
            return await self._get_state(kwargs.get("entity_id", ""), start)
        elif capability == "turn_on":
            return await self._turn_on(kwargs, start)
        elif capability == "turn_off":
            return await self._turn_off(kwargs.get("entity_id", ""), start)
        elif capability == "run_scene":
            return await self._run_scene(kwargs.get("scene_id", ""), start)
        return self._error_result(f"Unknown capability: {capability}", start)

    async def _list_devices(self, start: datetime) -> SkillResult:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self._base_url}/api/states",
                    headers=self._headers(),
                )
                resp.raise_for_status()
                states = resp.json()
                lights = [s for s in states if s.get("entity_id", "").startswith("light.")]
                switches = [s for s in states if s.get("entity_id", "").startswith("switch.")]
                lines = [f"Lights: {', '.join(s['entity_id'] for s in lights[:10])}"]
                lines.append(f"Switches: {', '.join(s['entity_id'] for s in switches[:10])}")
                return self._success_result("\n".join(lines), start)
        except Exception as e:
            return self._error_result(f"Home Assistant error: {e}", start)

    async def _get_state(self, entity_id: str, start: datetime) -> SkillResult:
        if not entity_id:
            return self._error_result("entity_id required", start)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self._base_url}/api/states/{entity_id}",
                    headers=self._headers(),
                )
                resp.raise_for_status()
                state = resp.json()
                return self._success_result(
                    f"{entity_id}: {state.get('state', 'unknown')}",
                    start,
                )
        except Exception as e:
            return self._error_result(f"Home Assistant error: {e}", start)

    async def _call_service(self, domain: str, service: str, data: dict) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{self._base_url}/api/services/{domain}/{service}",
                    headers=self._headers(),
                    json=data,
                )
                resp.raise_for_status()
                return True
        except Exception:
            return False

    async def _turn_on(self, kwargs: dict, start: datetime) -> SkillResult:
        entity_id = kwargs.get("entity_id", "")
        if not entity_id:
            return self._error_result("entity_id required", start)
        domain = entity_id.split(".")[0]
        data = {"entity_id": entity_id}
        if kwargs.get("brightness"):
            data["brightness"] = kwargs["brightness"]
        ok = await self._call_service(domain, "turn_on", data)
        return self._success_result(f"Turned on {entity_id}", start) if ok else self._error_result(f"Failed to turn on {entity_id}", start)

    async def _turn_off(self, entity_id: str, start: datetime) -> SkillResult:
        if not entity_id:
            return self._error_result("entity_id required", start)
        domain = entity_id.split(".")[0]
        ok = await self._call_service(domain, "turn_off", {"entity_id": entity_id})
        return self._success_result(f"Turned off {entity_id}", start) if ok else self._error_result(f"Failed to turn off {entity_id}", start)

    async def _run_scene(self, scene_id: str, start: datetime) -> SkillResult:
        if not scene_id:
            return self._error_result("scene_id required", start)
        ok = await self._call_service("scene", "turn_on", {"entity_id": scene_id})
        return self._success_result(f"Activated scene {scene_id}", start) if ok else self._error_result(f"Failed to activate {scene_id}", start)
