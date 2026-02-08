"""Cryptographic utilities for Aria."""

import base64
import secrets
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class CredentialManager:
    """
    Manages encryption and decryption of sensitive credentials.

    Uses Fernet symmetric encryption with a key derived from a master password.
    """

    def __init__(self, key_file: str | Path | None = None) -> None:
        """
        Initialize the credential manager.

        Args:
            key_file: Path to the encryption key file. If None, uses default location.
        """
        self._key_file = Path(key_file) if key_file else Path.home() / ".config/aria/secret.key"
        self._fernet: Fernet | None = None

    def _ensure_key(self) -> Fernet:
        """Ensure encryption key exists and return Fernet instance."""
        if self._fernet is not None:
            return self._fernet

        if self._key_file.exists():
            key = self._key_file.read_bytes()
        else:
            # Generate new key
            key = Fernet.generate_key()
            self._key_file.parent.mkdir(parents=True, exist_ok=True)
            self._key_file.write_bytes(key)
            self._key_file.chmod(0o600)  # Read/write only for owner

        self._fernet = Fernet(key)
        return self._fernet

    def encrypt(self, data: str) -> str:
        """
        Encrypt a string value.

        Args:
            data: The plaintext string to encrypt

        Returns:
            Base64-encoded encrypted string
        """
        fernet = self._ensure_key()
        encrypted = fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt an encrypted string.

        Args:
            encrypted_data: Base64-encoded encrypted string

        Returns:
            Decrypted plaintext string
        """
        fernet = self._ensure_key()
        encrypted = base64.urlsafe_b64decode(encrypted_data.encode())
        return fernet.decrypt(encrypted).decode()

    def encrypt_dict(self, data: dict[str, str]) -> dict[str, str]:
        """Encrypt all values in a dictionary."""
        return {k: self.encrypt(v) for k, v in data.items()}

    def decrypt_dict(self, data: dict[str, str]) -> dict[str, str]:
        """Decrypt all values in a dictionary."""
        return {k: self.decrypt(v) for k, v in data.items()}


def derive_key(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    """
    Derive an encryption key from a password.

    Args:
        password: The password to derive the key from
        salt: Optional salt (generated if not provided)

    Returns:
        Tuple of (derived_key, salt)
    """
    if salt is None:
        salt = secrets.token_bytes(16)

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key, salt


def generate_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token."""
    return secrets.token_urlsafe(length)


def generate_api_key() -> str:
    """Generate an API key in the format 'aria_xxx...'."""
    return f"aria_{secrets.token_urlsafe(32)}"


def hash_password(password: str) -> str:
    """
    Hash a password using Argon2.

    Args:
        password: The password to hash

    Returns:
        The hashed password string
    """
    from argon2 import PasswordHasher

    ph = PasswordHasher()
    return ph.hash(password)


def verify_password(password: str, hash_value: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        password: The password to verify
        hash_value: The hash to verify against

    Returns:
        True if the password matches, False otherwise
    """
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError

    ph = PasswordHasher()
    try:
        ph.verify(hash_value, password)
        return True
    except VerifyMismatchError:
        return False
