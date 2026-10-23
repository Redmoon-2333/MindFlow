"""Local provider selection and credentials. Never return or log secrets."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from threading import Lock
from typing import Literal, cast

from dotenv import dotenv_values
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, SecretStr

from mindflow.config import Settings, _get_data_dir

router = APIRouter(tags=["settings"])
_lock = Lock()
_KEYS = {"deepseek": "MINDFLOW_LLM__DEEPSEEK_API_KEY", "ecnu": "MINDFLOW_LLM__ECNU_API_KEY"}
_CONFIG_NAMES = {
    "MINDFLOW_LLM__SERVICE",
    "MINDFLOW_LLM__ECNU_MODEL",
    "MINDFLOW_LLM__ECNU_COMPAT_ENABLED",
}


class CredentialInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: SecretStr
    service: Literal["deepseek", "ecnu"] = "deepseek"
    ecnu_model: Literal["ecnu-max", "ecnu-plus"] = "ecnu-max"


class CredentialStatus(BaseModel):
    saved: bool
    active_configured: bool
    environment_override: bool
    restart_required: bool
    provider: str
    model: str
    service: Literal["deepseek", "ecnu"]
    active_provider: str


def _env_path() -> Path:
    return _get_data_dir() / ".env"


def _status(request: Request) -> CredentialStatus:
    settings = cast(Settings, request.app.state.settings)
    path = _env_path()
    values = dotenv_values(path) if path.exists() else {}
    # Same loader and precedence as a fresh backend process (without caching).
    pending = Settings(_env_file=path).llm  # type: ignore[call-arg]
    target = pending.l1_target()
    active = settings.llm.l1_target()
    saved_key = values.get(_KEYS[pending.service]) or ""
    if pending.service == "deepseek" and not saved_key:
        saved_key = values.get("DEEPSEEK_API_KEY") or ""
    relevant_keys = {_KEYS[pending.service]}
    if pending.service == "deepseek":
        relevant_keys.add("DEEPSEEK_API_KEY")
    overridden = any(bool(os.environ.get(name)) for name in relevant_keys | _CONFIG_NAMES)
    return CredentialStatus(
        saved=bool(saved_key),
        active_configured=bool(active.api_key),
        environment_override=overridden,
        restart_required=(target.provider, target.model, target.base_url, target.api_key)
        != (active.provider, active.model, active.base_url, active.api_key),
        provider="ChatECNU" if target.is_ecnu else "DeepSeek",
        model=target.model,
        service=pending.service,
        active_provider="ChatECNU" if active.is_ecnu else "DeepSeek",
    )


def _persist(key: str | None, service: str | None = None, model: str = "ecnu-max") -> None:
    path = _env_path()
    with _lock:
        original = path.read_text(encoding="utf-8") if path.exists() else ""
        values = dotenv_values(path) if path.exists() else {}
        selected = service or values.get("MINDFLOW_LLM__SERVICE") or "deepseek"
        if selected not in _KEYS:
            selected = "deepseek"
        names = {_KEYS[selected]}
        if selected == "deepseek":
            names.add("DEEPSEEK_API_KEY")
        if key is not None:
            names |= _CONFIG_NAMES
        pattern = r"^\s*(?:export\s+)?(?:" + "|".join(re.escape(n) for n in names) + r")\s*="
        lines = [line for line in original.splitlines() if not re.match(pattern, line)]
        if key is not None:
            lines += [
                f"{_KEYS[selected]}='{key}'",
                f"MINDFLOW_LLM__SERVICE='{selected}'",
                "MINDFLOW_LLM__ECNU_COMPAT_ENABLED='false'",
            ]
            if selected == "ecnu":
                lines.append(f"MINDFLOW_LLM__ECNU_MODEL='{model}'")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".env-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write("\n".join(lines) + "\n")
            os.chmod(name, 0o600)
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)


@router.get("/settings/ai", response_model=CredentialStatus)
def get_ai_settings(request: Request) -> CredentialStatus:
    return _status(request)


@router.put("/settings/ai", response_model=CredentialStatus)
def save_ai_settings(body: CredentialInput, request: Request) -> CredentialStatus:
    key = body.api_key.get_secret_value().strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{8,512}", key):
        raise HTTPException(400, "Key 格式无效，请填写所选服务的有效 API Key")
    _persist(key, body.service, body.ecnu_model)
    return _status(request)


@router.delete("/settings/ai", response_model=CredentialStatus)
def delete_ai_settings(request: Request) -> CredentialStatus:
    # Delete the effective selection shown to the user, even under env override.
    status = _status(request)
    _persist(None, status.service)
    return _status(request)
