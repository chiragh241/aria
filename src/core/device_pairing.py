"""Device pairing — secure pairing via short codes and allowlists.

Allows new devices (phones, tablets, other machines) to connect to Aria
by entering a short numeric pairing code. Once paired, the device gets
a token for ongoing access.

Pairing flow:
  1. User requests a pairing code from the web UI or CLI
  2. System generates a 6-digit code valid for 5 minutes
  3. User enters the code on the new device
  4. Device receives an access token and is added to the paired list
"""

import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Pairing code expiry (seconds)
CODE_EXPIRY = 300  # 5 minutes
# Token length
TOKEN_LENGTH = 48


@dataclass
class PairedDevice:
    """A paired device."""

    device_id: str
    name: str
    token_hash: str  # SHA-256 of the access token (never store raw tokens)
    paired_at: float
    last_seen: float = 0.0
    user_agent: str = ""
    ip_address: str = ""
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "paired_at": self.paired_at,
            "last_seen": self.last_seen,
            "user_agent": self.user_agent,
            "ip_address": self.ip_address,
            "enabled": self.enabled,
        }


@dataclass
class PairingCode:
    """A temporary pairing code."""

    code: str
    created_at: float
    expires_at: float
    device_name: str = ""

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at


class DevicePairingManager:
    """Manages device pairing and access tokens."""

    def __init__(self) -> None:
        self.settings = get_settings()
        data_dir = Path(self.settings.aria.data_dir).expanduser()
        self._store_path = data_dir / "paired_devices.json"
        self._devices: dict[str, PairedDevice] = {}
        self._pending_codes: dict[str, PairingCode] = {}  # code → PairingCode

    async def initialize(self) -> None:
        """Load paired devices from disk."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        if self._store_path.exists():
            try:
                data = json.loads(self._store_path.read_text())
                for d in data.get("devices", []):
                    device = PairedDevice(
                        device_id=d["device_id"],
                        name=d["name"],
                        token_hash=d["token_hash"],
                        paired_at=d["paired_at"],
                        last_seen=d.get("last_seen", 0),
                        user_agent=d.get("user_agent", ""),
                        ip_address=d.get("ip_address", ""),
                        enabled=d.get("enabled", True),
                    )
                    self._devices[device.device_id] = device
                logger.info("Loaded paired devices", count=len(self._devices))
            except Exception as e:
                logger.error("Failed to load paired devices", error=str(e))

    def _save(self) -> None:
        """Persist devices to disk."""
        try:
            data = {
                "devices": [
                    {
                        "device_id": d.device_id,
                        "name": d.name,
                        "token_hash": d.token_hash,
                        "paired_at": d.paired_at,
                        "last_seen": d.last_seen,
                        "user_agent": d.user_agent,
                        "ip_address": d.ip_address,
                        "enabled": d.enabled,
                    }
                    for d in self._devices.values()
                ]
            }
            self._store_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error("Failed to save paired devices", error=str(e))

    # --- Pairing flow ---

    def generate_code(self, device_name: str = "") -> str:
        """Generate a 6-digit pairing code.

        Returns the code string. Valid for CODE_EXPIRY seconds.
        """
        # Clean up expired codes
        self._cleanup_expired()

        code = f"{secrets.randbelow(1000000):06d}"
        # Ensure uniqueness
        while code in self._pending_codes:
            code = f"{secrets.randbelow(1000000):06d}"

        now = time.time()
        self._pending_codes[code] = PairingCode(
            code=code,
            created_at=now,
            expires_at=now + CODE_EXPIRY,
            device_name=device_name,
        )

        logger.info("Pairing code generated", device_name=device_name)
        return code

    def redeem_code(
        self,
        code: str,
        device_name: str = "",
        user_agent: str = "",
        ip_address: str = "",
    ) -> dict[str, Any]:
        """Redeem a pairing code and get an access token.

        Returns {"success": True, "token": "...", "device_id": "..."} on success,
        or {"success": False, "error": "..."} on failure.
        """
        self._cleanup_expired()

        pairing = self._pending_codes.pop(code, None)
        if pairing is None:
            return {"success": False, "error": "Invalid or expired pairing code."}

        if pairing.expired:
            return {"success": False, "error": "Pairing code has expired."}

        # Generate access token and device ID (separate from token hash)
        token = secrets.token_urlsafe(TOKEN_LENGTH)
        device_id = secrets.token_hex(8)  # 16 hex chars, independent of token
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        now = time.time()
        device = PairedDevice(
            device_id=device_id,
            name=device_name or pairing.device_name or f"Device {len(self._devices) + 1}",
            token_hash=token_hash,
            paired_at=now,
            last_seen=now,
            user_agent=user_agent,
            ip_address=ip_address,
        )

        self._devices[device_id] = device
        self._save()

        logger.info("Device paired", device_id=device_id, name=device.name)

        return {
            "success": True,
            "token": token,
            "device_id": device_id,
            "name": device.name,
        }

    # --- Token validation ---

    def validate_token(self, token: str) -> PairedDevice | None:
        """Validate an access token and return the associated device.

        Also updates last_seen timestamp (saves at most once per minute).
        """
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        for device in self._devices.values():
            if device.token_hash == token_hash and device.enabled:
                now = time.time()
                device.last_seen = now
                # Only persist every 60s to avoid excessive disk I/O
                if not hasattr(self, "_last_save_time") or now - self._last_save_time > 60:
                    self._save()
                    self._last_save_time = now
                return device

        return None

    # --- Device management ---

    def list_devices(self) -> list[dict[str, Any]]:
        """List all paired devices (without sensitive data)."""
        return [d.to_dict() for d in self._devices.values()]

    def remove_device(self, device_id: str) -> bool:
        """Unpair a device."""
        if device_id in self._devices:
            del self._devices[device_id]
            self._save()
            logger.info("Device removed", device_id=device_id)
            return True
        return False

    def toggle_device(self, device_id: str, enabled: bool) -> bool:
        """Enable or disable a paired device."""
        device = self._devices.get(device_id)
        if device:
            device.enabled = enabled
            self._save()
            return True
        return False

    def rename_device(self, device_id: str, name: str) -> bool:
        """Rename a paired device."""
        device = self._devices.get(device_id)
        if device:
            device.name = name
            self._save()
            return True
        return False

    def get_pending_codes(self) -> list[dict[str, Any]]:
        """List active pairing codes (for admin UI)."""
        self._cleanup_expired()
        return [
            {
                "code": c.code,
                "device_name": c.device_name,
                "expires_at": c.expires_at,
                "remaining_seconds": max(0, int(c.expires_at - time.time())),
            }
            for c in self._pending_codes.values()
        ]

    def _cleanup_expired(self) -> None:
        """Remove expired pairing codes."""
        expired = [c for c, p in self._pending_codes.items() if p.expired]
        for code in expired:
            del self._pending_codes[code]
