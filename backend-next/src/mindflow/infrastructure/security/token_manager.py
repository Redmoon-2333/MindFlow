"""Token file management for local API authentication.

Token is a 64-byte (128 hex char) random string stored in the platform
data directory. Uses secrets module for cryptographically secure random
bytes and constant-time comparison.

File permissions:
  - POSIX: chmod 0600 (owner read/write only)
  - Windows: no equivalent file permission; rely on platformdirs' user-local
    directory which is already ACL'd to the current user (NTFS default).

Security model (ADR-004):
  - No network involved — shared via filesystem between backend and frontend
  - File system permissions as security boundary
  - Constant-time comparison via secrets.compare_digest prevents timing attacks
"""

from __future__ import annotations

import hashlib
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


def load_or_create_token(path: Path) -> str:
    """Load existing token from file or generate and persist a new one.

    Args:
        path: Path to the token file (typically under platformdirs user data dir).

    Returns:
        128-character hex string (64 random bytes).
    """
    path = Path(path)
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            logger.debug("Loaded existing token from {}", path)
            return token
        logger.warning("Token file exists but is empty, regenerating")

    token = secrets.token_hex(64)  # 64 bytes → 128 hex chars
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((token + "\n").encode("utf-8"))

    _set_file_permissions(path)

    logger.info("Generated new token at {}", path)
    return token


def verify_token(provided: str, expected: str) -> bool:
    """Verify a provided token against the expected value.

    Uses constant-time comparison to prevent timing side-channel attacks.

    Args:
        provided: Token value from the Authorization header.
        expected: Token value loaded from the token file.

    Returns:
        True if tokens match, False otherwise.
    """
    if not provided or not expected:
        return False
    return secrets.compare_digest(provided.strip(), expected.strip())


def _set_file_permissions(path: Path) -> None:
    """Set restrictive file permissions on the token file.

    POSIX: chmod 0600 (owner rw, no group/other access).
    Windows: Best-effort only — NTFS ACLs on the user's data directory
    already restrict access to the current user. The chmod won't error
    but has no effect on Windows.
    """
    try:
        path.chmod(0o600)
    except NotImplementedError:
        # Windows: chmod is a no-op on some file systems; the token is
        # protected by the platformdirs directory ACL instead.
        logger.debug("chmod not supported on this platform (Windows); relying on directory ACL")
    except PermissionError:
        logger.warning("Could not set 0600 on token file — ownership issue")


@dataclass(frozen=True, slots=True)
class _BootstrapTicket:
    digest: bytes
    expires_at: float


class BootstrapTicketStore:
    """Bounded in-memory store for short-lived one-time bootstrap tickets."""

    def __init__(
        self,
        *,
        ttl_s: int = 60,
        max_entries: int = 16,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_s <= 0 or max_entries <= 0:
            raise ValueError("ttl_s and max_entries must be positive")
        self._ttl_s = ttl_s
        self._max_entries = max_entries
        self._clock = clock
        self._entries: list[_BootstrapTicket] = []

    def issue(self) -> str:
        """Create a random one-time ticket and evict expired/old entries."""
        now = self._clock()
        self._entries = [entry for entry in self._entries if entry.expires_at > now]
        token = secrets.token_urlsafe(32)
        self._entries.append(
            _BootstrapTicket(self._digest(token), now + self._ttl_s)
        )
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]
        return token

    def consume(self, provided: str) -> bool:
        """Atomically validate and remove a ticket."""
        if not provided:
            return False
        now = self._clock()
        candidate = self._digest(provided)
        retained: list[_BootstrapTicket] = []
        matched = False
        for entry in self._entries:
            if entry.expires_at <= now:
                continue
            if not matched and secrets.compare_digest(candidate, entry.digest):
                matched = True
                continue
            retained.append(entry)
        self._entries = retained
        return matched

    @staticmethod
    def _digest(token: str) -> bytes:
        return hashlib.sha256(token.encode("utf-8")).digest()


@dataclass(frozen=True, slots=True)
class _SessionToken:
    digest: bytes
    expires_at: float


class SessionTokenStore:
    """Bounded in-memory store for revocable browser session tokens."""

    def __init__(
        self,
        *,
        ttl_s: int = 24 * 60 * 60,
        max_entries: int = 16,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_s <= 0 or max_entries <= 0:
            raise ValueError("ttl_s and max_entries must be positive")
        self._ttl_s = ttl_s
        self._max_entries = max_entries
        self._clock = clock
        self._entries: list[_SessionToken] = []

    def issue(self) -> str:
        """Create an independent browser session without exposing the root token."""
        now = self._clock()
        self._entries = [entry for entry in self._entries if entry.expires_at > now]
        token = secrets.token_urlsafe(32)
        self._entries.append(_SessionToken(self._digest(token), now + self._ttl_s))
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]
        return token

    def verify(self, provided: str) -> bool:
        """Validate an unexpired browser session token without consuming it."""
        if not provided:
            return False
        now = self._clock()
        candidate = self._digest(provided)
        retained: list[_SessionToken] = []
        matched = False
        for entry in self._entries:
            if entry.expires_at <= now:
                continue
            retained.append(entry)
            matched = matched or secrets.compare_digest(candidate, entry.digest)
        self._entries = retained
        return matched

    @staticmethod
    def _digest(token: str) -> bytes:
        return hashlib.sha256(token.encode("utf-8")).digest()
