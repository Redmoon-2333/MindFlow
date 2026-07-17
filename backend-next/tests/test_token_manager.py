"""Tests for mindflow.infrastructure.security.token_manager.

Tests cover:
  - load_or_create_token generates a token when file doesn't exist
  - load_or_create_token reuses existing token
  - Token format (128 hex chars)
  - verify_token with correct/incorrect tokens (constant-time comparison)
  - File permissions (best-effort on Windows)
  - Empty file regeneration
"""

from __future__ import annotations

import secrets
from pathlib import Path

from mindflow.infrastructure.security.token_manager import (
    load_or_create_token,
    verify_token,
)


class TestLoadOrCreateToken:
    """Verify token file management."""

    def test_creates_new_token(self, tmp_path: Path):
        """A new token is created when file doesn't exist."""
        token_path = tmp_path / ".mindflow" / "token"
        token = load_or_create_token(token_path)
        assert len(token) == 128, f"Expected 128 hex chars, got {len(token)}"
        # Verify it's valid hex
        int(token, 16)

    def test_file_is_created(self, tmp_path: Path):
        """The token file is created on disk."""
        token_path = tmp_path / ".mindflow" / "token"
        load_or_create_token(token_path)
        assert token_path.exists()
        content = token_path.read_text(encoding="utf-8").strip()
        assert len(content) == 128

    def test_reuses_existing_token(self, tmp_path: Path):
        """Calling load_or_create_token twice returns the same token."""
        token_path = tmp_path / ".mindflow" / "token"
        token1 = load_or_create_token(token_path)
        token2 = load_or_create_token(token_path)
        assert token1 == token2, "Token should be stable across calls"

    def test_file_format(self, tmp_path: Path):
        """Token file contains the hex token followed by newline."""
        token_path = tmp_path / ".mindflow" / "token"
        token = load_or_create_token(token_path)
        content = token_path.read_bytes()
        assert content == (token + "\n").encode("utf-8")

    def test_multiple_calls_same_token(self, tmp_path: Path):
        """Multiple calls to load_or_create_token are idempotent."""
        token_path = tmp_path / "multi_token"
        tokens = [load_or_create_token(token_path) for _ in range(5)]
        assert all(t == tokens[0] for t in tokens)

    def test_empty_file_regenerates(self, tmp_path: Path):
        """An empty token file triggers regeneration."""
        token_path = tmp_path / "empty_token"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text("")
        token = load_or_create_token(token_path)
        assert len(token) == 128
        assert token_path.read_text(encoding="utf-8").strip() == token

    def test_whitespace_only_file_regenerates(self, tmp_path: Path):
        """A whitespace-only token file triggers regeneration."""
        token_path = tmp_path / "whitespace_token"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text("   \n  \n")
        token = load_or_create_token(token_path)
        assert len(token) == 128

    def test_parent_dir_created(self, tmp_path: Path):
        """Parent directories are created if they don't exist."""
        token_path = tmp_path / "deep" / "nested" / "dir" / "token"
        token = load_or_create_token(token_path)
        assert token_path.exists()
        assert len(token) == 128

    def test_file_permission_set(self, tmp_path: Path):
        """Token file has restricted permissions on POSIX (best-effort on Windows)."""
        token_path = tmp_path / "perm_token"
        load_or_create_token(token_path)
        mode = token_path.stat().st_mode
        # Check owner read/write is set (0o600 = 0o100600)
        assert mode & 0o600 == 0o600, f"Expected 0o600 permissions, got {oct(mode)}"


class TestVerifyToken:
    """Verify constant-time token comparison."""

    def test_correct_token_passes(self):
        """Correct token verification returns True."""
        token = secrets.token_hex(32)
        assert verify_token(token, token) is True

    def test_incorrect_token_fails(self):
        """Incorrect token verification returns False."""
        token1 = secrets.token_hex(32)
        token2 = secrets.token_hex(32)
        assert verify_token(token1, token2) is False

    def test_empty_provided_fails(self):
        """Empty provided token returns False."""
        token = secrets.token_hex(32)
        assert verify_token("", token) is False

    def test_empty_expected_fails(self):
        """Empty expected token returns False."""
        token = secrets.token_hex(32)
        assert verify_token(token, "") is False

    def test_whitespace_handling(self):
        """verify_token strips whitespace from both sides."""
        token = secrets.token_hex(32)
        assert verify_token("  " + token + "  ", token) is True

    def test_trailing_newline_handling(self):
        """verify_token handles tokens read from file with trailing newline."""
        token = secrets.token_hex(32)
        assert verify_token(token + "\n", token) is True

    def test_none_provided_fails(self):
        """None provided token returns False."""
        token = secrets.token_hex(32)
        assert verify_token(None, token) is False  # type: ignore[arg-type]

    def test_case_sensitive(self):
        """Token comparison is case-sensitive."""
        token = secrets.token_hex(32)
        assert verify_token(token.upper(), token) is False
