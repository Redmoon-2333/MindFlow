"""HMAC-SHA256 signing/verification for pickled model artifacts.

``ModelManager`` persists models via ``joblib.dump``/``joblib.load``, which
is pickle under the hood. ``models_dir`` is a ``platformdirs`` user-writable
directory, so any local process (or malware) can drop a crafted ``.pkl``
there; ``joblib.load`` on that file achieves arbitrary code execution the
next time the app loads models. This module closes that gap by signing every
saved model file with HMAC-SHA256 and refusing to load a file whose sibling
``.hmac`` is missing or does not match.

This is NOT encryption — model contents stay in plaintext. HMAC only proves
"this file was written by a process holding the signing key", which is
sufficient for the local-trust-boundary threat model (a compromised sibling
process/tab should not be able to plant code that gets executed on load).

Key storage mirrors ``infrastructure/security/token_manager.py``: a random
key generated on first use, persisted to disk, chmod 0600 on POSIX (no-op on
Windows, where the platformdirs directory ACL is the actual boundary). The
key is intentionally NOT reused from ``token_manager`` — the auth token is a
64-byte hex string meant for constant-time string comparison over HTTP; this
key is raw bytes for ``hmac.new()``. Sharing one file would leak the token's
byte length into the signing key material for no benefit, so each gets its
own file.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from pathlib import Path

from loguru import logger

_KEY_FILENAME = "model_signing.key"
_KEY_BYTES = 32  # HMAC-SHA256 key size


def _load_or_create_signing_key(data_dir: Path) -> bytes:
    """Load the HMAC signing key from ``data_dir``, generating it if absent.

    Args:
        data_dir: Directory to store ``model_signing.key`` in (typically
            ``ModelManager.models_dir``, so the key travels with the models
            it signs).

    Returns:
        32 raw key bytes for use with ``hmac.new(key, ..., hashlib.sha256)``.
    """
    key_path = data_dir / _KEY_FILENAME
    if key_path.exists():
        key = key_path.read_bytes()
        if len(key) == _KEY_BYTES:
            return key
        logger.warning("Model signing key at {} has unexpected length, regenerating", key_path)

    key = secrets.token_bytes(_KEY_BYTES)
    data_dir.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(key)
    _set_file_permissions(key_path)

    logger.info("Generated new model signing key at {}", key_path)
    return key


def _set_file_permissions(path: Path) -> None:
    """Restrict the signing key file to owner read/write.

    POSIX: chmod 0600. Windows: no-op — chmod has no effect there, so the
    platformdirs user data directory's NTFS ACL is the actual boundary
    (mirrors ``token_manager._set_file_permissions``).
    """
    try:
        path.chmod(0o600)
    except NotImplementedError:
        logger.debug("chmod not supported on this platform (Windows); relying on directory ACL")
    except PermissionError:
        logger.warning("Could not set 0600 on model signing key — ownership issue")


def _hmac_path(model_path: Path) -> Path:
    """Return the sibling ``.hmac`` path for a model file, e.g. ``foo.pkl.hmac``."""
    return model_path.with_name(model_path.name + ".hmac")


def sign_model_file(path: Path, key: bytes) -> None:
    """Compute the HMAC-SHA256 of ``path`` and write it to a sibling ``.hmac`` file.

    Args:
        path: Path to the model file that was just written (e.g. a ``.pkl``).
        key: Signing key from ``_load_or_create_signing_key``.
    """
    digest = hmac.new(key, path.read_bytes(), hashlib.sha256).hexdigest()
    _hmac_path(path).write_text(digest, encoding="utf-8")


def verify_model_file(path: Path, key: bytes) -> bool:
    """Verify a model file's HMAC-SHA256 against its sibling ``.hmac`` file.

    Args:
        path: Path to the model file to verify.
        key: Signing key from ``_load_or_create_signing_key``.

    Returns:
        True only if the ``.hmac`` sibling exists and matches. False for a
        missing sibling (unsigned file) or any mismatch (tampered file) —
        callers must treat both as "do not load", not just log a warning.
    """
    hmac_path = _hmac_path(path)
    if not hmac_path.exists():
        return False

    expected = hmac_path.read_text(encoding="utf-8").strip()
    actual = hmac.new(key, path.read_bytes(), hashlib.sha256).hexdigest()
    # Constant-time comparison — signature verification is a MAC check, not a
    # UI diff; timing leaks on hex digest comparison are a classic side
    # channel (mirrors token_manager.verify_token's use of compare_digest).
    return hmac.compare_digest(actual, expected)
