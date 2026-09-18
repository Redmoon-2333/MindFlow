"""Allowlisted provider error metadata; never stringify upstream exceptions."""

from __future__ import annotations


def safe_error_metadata(exc: BaseException) -> str:
    """Exclude messages, bodies, headers, request URLs and validation inputs."""
    status = getattr(exc, "status_code", None)
    suffix = f" status={status}" if type(status) is int else ""
    return f"type={type(exc).__name__}{suffix}"
