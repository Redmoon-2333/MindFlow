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

import secrets
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
