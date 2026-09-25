"""Application configuration via Pydantic BaseSettings.

Configuration source priority (highest to lowest):
  1. Environment variables
  2. .env file (in platformdirs user data dir)
  3. Default values

All datetime values are timezone-aware UTC throughout the application.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import platformdirs
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogSettings(BaseSettings):
    """Structured logging configuration via loguru."""

    level: str = Field(default="DEBUG", description="Log level: DEBUG|INFO|WARNING|ERROR|CRITICAL")
    json_format: bool = Field(default=False, description="Emit JSON-structured logs (production)")
    rotation: str = Field(default="10 MB", description="Log file rotation threshold")
    retention: str = Field(default="30 days", description="Log file retention period")
    compression: str = Field(default="gz", description="Log file compression format")


@dataclass(frozen=True, slots=True)
class L1Target:
    """The effective L1 (primary LLM) endpoint for this process.

    Produced by :meth:`LLMSettings.l1_target`.  Carries no secret material in
    its ``provenance``/``provider``/``model`` fields, so it is safe to log and
    to embed in diagnostics — the ``api_key`` field must never be serialised.
    """

    provider: Literal["generic", "ecnu"]
    api_key: str | None
    base_url: str
    model: str
    reasoning_effort: str
    thinking_enabled: bool
    max_output_tokens: int
    #: Why this target was chosen: ``"deepseek-direct"`` (production pin) or
    #: ``"ecnu-compat"`` (legacy triple, explicitly opted in).
    provenance: str

    @property
    def is_ecnu(self) -> bool:
        return self.provider == "ecnu"

    def describe(self) -> dict[str, object]:
        """Secret-free description for logs, reports and diagnostics."""
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "thinking_enabled": self.thinking_enabled,
            "max_output_tokens": self.max_output_tokens,
            "provenance": self.provenance,
            "credential_present": bool(self.api_key),
        }


class LLMSettings(BaseSettings):
    """LLM API configuration for attribution pipeline (Wave 6).

    Three-tier degradation chain (Architecture §3.3):
      L1: **DeepSeek direct** (`provider=generic`,
          ``base_url=https://api.deepseek.com``, ``model=deepseek-flash``,
          credential from ``DEEPSEEK_API_KEY``).  The campus (ECNU) gateway is
          *not* used in production and there is no ECNU→DeepSeek failover: a
          missing DeepSeek credential makes L1 unavailable and the existing
          L2 (Ollama, when enabled) / L3 (RuleEngine) tiers take over.
      L2: Ollama local (ollama_enabled + ollama_base_url + ollama_model)
      L3: RuleEngine (always available, zero config)

    Legacy ECNU values (``PROVIDER=ecnu``, an ``ecnu.edu.cn`` base URL, an
    ``ecnu-*`` model or an ECNU-issued key) are deliberately **overridden** by
    the pin so a lingering ``.env`` cannot mix an ECNU endpoint with a DeepSeek
    model or key.  The ECNU adapter code is retained, and the legacy triple can
    still be exercised by opting in explicitly (``ecnu_compat_enabled=True``,
    env ``MINDFLOW_LLM__ECNU_COMPAT_ENABLED``) — used by the adapter's own
    compatibility tests and diagnostics, never by the production assembly.
    """

    timeout_s: int = Field(default=180, ge=1, le=600, description="LLM request timeout in seconds")
    max_retries: int = Field(default=1, ge=0, le=10, description="LLM retry budget")
    api_key: str | None = Field(default=None, description="LLM API key (legacy ECNU triple)")
    base_url: str | None = Field(default=None, description="LLM API base URL (legacy ECNU triple)")
    model: str | None = Field(default=None, description="LLM model identifier (legacy ECNU triple)")
    # ── Provider identity / thinking-mode controls ──────────────────────
    provider: str = Field(
        default="auto",
        description=(
            "Primary provider: 'auto' (infer from base_url/model), 'ecnu', "
            "or 'generic' (plain OpenAI-compatible). Only consulted when "
            "`ecnu_compat_enabled` is true; production always pins DeepSeek."
        ),
    )
    # ── Production L1 pin (DeepSeek direct) ─────────────────────────────
    deepseek_api_key: str | None = Field(
        default=None,
        description=(
            "Production L1 credential. Read from DEEPSEEK_API_KEY or "
            "MINDFLOW_LLM__DEEPSEEK_API_KEY. When absent, L1 is unavailable and "
            "the chain degrades to L2/L3 — it never falls back to the ECNU key."
        ),
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        description="DeepSeek API base URL used by the production pin",
    )
    deepseek_model: str = Field(
        default="deepseek-flash",
        description=(
            "Model id used for every L1 request. Panel 'chat'/'reasoner' are "
            "output policies (JSON vs prose), not two different models."
        ),
    )
    ecnu_compat_enabled: bool = Field(
        default=False,
        description=(
            "Honour the legacy ECNU triple (provider/base_url/model/api_key) "
            "instead of the DeepSeek pin. For the ECNU adapter's compatibility "
            "tests and diagnostics only — must stay False in production."
        ),
    )
    thinking_enabled: bool = Field(
        default=True,
        description="Request thinking mode on every generation (DeepSeek `thinking`)",
    )
    reasoning_effort: str = Field(
        default="high",
        description=(
            "Thinking intensity sent per request. DeepSeek maps minimal→low, "
            "medium/xhigh→high, ultra→max; 'none' disables thinking."
        ),
    )
    max_output_tokens: int = Field(
        default=16384,
        ge=256,
        le=131072,
        description="Explicit output cap per generation; recorded in the audit trail.",
    )
    max_concurrent_requests: int = Field(
        default=1,
        ge=1,
        le=8,
        description=(
            "In-flight generation cap for this process. The campus gateway "
            "allows 3 concurrent requests per user per model, so the default "
            "stays at 1 until measured."
        ),
    )
    panel_fanout_concurrency: int = Field(
        default=3,
        ge=1,
        le=8,
        description=(
            "Burst permit count for the panel's parallel attribution/rebuttal "
            "fan-out only. 1 keeps the strict global serialization; a higher "
            "value lets the parallel expert batch overlap through a dedicated "
            "gateway with its own concurrency gate while every other L1 path "
            "(chat, intervention, structured attribution) keeps "
            "``max_concurrent_requests``. Measured 2026-09-25 against real "
            "DeepSeek (2 scenarios, 9 calls each): burst 3 cut panel wall time "
            "114.5s→60.5s (-47%) and serial queueing 82.5s→0ms (-100%) with "
            "identical verdict types and zero errors. Set 1 to revert."
        ),
    )
    ollama_enabled: bool = Field(default=False, description="Enable Ollama local fallback (L2)")
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama API base URL"
    )
    ollama_model: str = Field(default="qwen3:8b", description="Ollama model name")

    @model_validator(mode="before")
    @classmethod
    def _deepseek_key_from_environment(cls, values: Any) -> Any:
        """Resolve the DeepSeek credential from either environment variable.

        ``MINDFLOW_LLM__DEEPSEEK_API_KEY`` always wins over the bare
        ``DEEPSEEK_API_KEY``:

        * nested under ``Settings`` the prefixed name maps straight onto this
          field, so nothing to do;
        * a *standalone* ``LLMSettings`` would only see the bare name (its field
          name uppercased), which is why the bare value is replaced here.

        A credential passed explicitly by the caller is never overridden — only
        the direct mapping of the bare variable is.
        """
        if not isinstance(values, dict):
            return values
        prefixed = os.environ.get("MINDFLOW_LLM__DEEPSEEK_API_KEY", "").strip()
        current = values.get("deepseek_api_key")
        if prefixed and current != prefixed:
            bare = os.environ.get("DEEPSEEK_API_KEY", "").strip()
            if current is None or current == bare:
                return {**values, "deepseek_api_key": prefixed}
        if current:
            return values
        bare = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if bare:
            return {**values, "deepseek_api_key": bare}
        return values

    @property
    def is_ecnu(self) -> bool:
        """True when this configuration *describes* the campus gateway.

        This is a property of the legacy triple (used by the ECNU adapter and
        its compatibility tests). Production does not consult it directly: the
        assembly resolves :meth:`l1_target`, which pins DeepSeek unless
        ``ecnu_compat_enabled`` opts back into the legacy behaviour.
        """
        if self.provider.strip().lower() == "ecnu":
            return True
        if self.provider.strip().lower() == "generic":
            return False
        return "ecnu.edu.cn" in (self.base_url or "").lower() or (
            (self.model or "").lower().startswith("ecnu-")
        )

    @property
    def deepseek_credential(self) -> str | None:
        """The production L1 credential, or None when L1 is unavailable.

        Resolution order:

        1. the dedicated DeepSeek settings/env var (``DEEPSEEK_API_KEY`` /
           ``MINDFLOW_LLM__DEEPSEEK_API_KEY``);
        2. the legacy ``api_key`` **only when the legacy triple is not
           ECNU-shaped** — i.e. a configuration that already pointed at
           DeepSeek (as the repository's own ``.env`` does);
        3. otherwise None: an ECNU-issued key is never sent to DeepSeek, and a
           missing credential degrades to L2/L3 rather than falling back to
           ECNU.
        """
        explicit = (self.deepseek_api_key or "").strip()
        if explicit:
            return explicit
        if self.is_ecnu:
            return None
        legacy = (self.api_key or "").strip()
        return legacy or None

    def l1_target(self) -> L1Target:
        """Resolve the effective L1 provider/endpoint/model for this process.

        Production (``ecnu_compat_enabled`` false, the default) always returns
        the DeepSeek pin, overriding any legacy ECNU URL, model or key that is
        still present in the environment.  Setting ``ecnu_compat_enabled``
        returns the legacy triple unchanged so the ECNU adapter remains usable
        in its own tests and in diagnostics.
        """
        if self.ecnu_compat_enabled:
            legacy_ecnu = self.is_ecnu
            return L1Target(
                provider="ecnu" if legacy_ecnu else "generic",
                api_key=self.api_key,
                base_url=(self.base_url or self.deepseek_base_url).rstrip("/"),
                model=self.model or self.deepseek_model,
                reasoning_effort=self.reasoning_effort,
                thinking_enabled=self.thinking_enabled,
                max_output_tokens=self.max_output_tokens,
                provenance="ecnu-compat",
            )
        return L1Target(
            provider="generic",
            api_key=self.deepseek_credential,
            base_url=self.deepseek_base_url.rstrip("/"),
            model=self.deepseek_model,
            reasoning_effort=self.reasoning_effort,
            thinking_enabled=self.thinking_enabled,
            max_output_tokens=self.max_output_tokens,
            provenance="deepseek-direct",
        )


_cached_data_dir: Path | None = None


def _get_data_dir() -> Path:
    """Return platform-appropriate user data directory (cached)."""
    global _cached_data_dir
    if _cached_data_dir is None:
        _cached_data_dir = Path(platformdirs.user_data_dir("mindflow", ensure_exists=True))
    return _cached_data_dir


class Settings(BaseSettings):
    """Application-wide settings.

    Priority: env vars > .env file > defaults.
    The .env file is searched in platformdirs user data directory.
    """

    model_config = SettingsConfigDict(
        env_prefix="MINDFLOW_",
        env_file=None,
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )

    # --- Runtime paths ---
    data_dir: Path = Field(
        default_factory=_get_data_dir,
        description="Application data directory; relative paths are anchored to platform data",
    )
    models_dir: Path = Field(
        default=Path("models"),
        description="ML model directory; relative paths are anchored to data_dir",
    )

    @property
    def backup_dir(self) -> Path:
        """Directory used for database backups."""
        return self.data_dir / "backups"

    @property
    def token_path(self) -> Path:
        """Path to the local API authentication token."""
        return self.data_dir / "token"

    @property
    def otel_db_path(self) -> Path:
        """Path to the local OpenTelemetry span database (ADR-003)."""
        return self.data_dir / "otel_traces.db"

    # --- Database ---
    db_url: str = Field(
        default="sqlite+aiosqlite:///{data_dir}/mindflow.db",
        description="SQLAlchemy async database URL",
    )

    # --- Server ---
    host: str = Field(default="127.0.0.1", description="Bind address")
    port: int = Field(default=8765, description="Bind port")
    timezone: str = Field(
        default="local",
        description="Local business timezone: 'local' or an IANA timezone name",
    )

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        if value == "local":
            return value
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            msg = f"Unknown timezone: {value}"
            raise ValueError(msg) from exc
        return value

    # --- Runtime roles ---
    run_scheduler: bool = Field(default=True, description="Run scheduled background jobs")
    run_collectors: bool = Field(default=True, description="Run activity and input collectors")

    # --- Collector ---
    collect_interval_s: int = Field(
        default=5, ge=1, le=60, description="Collector tick interval in seconds"
    )
    idle_collect_interval_s: int = Field(
        default=30, ge=5, le=600,
        description=(
            "Widened collector tick interval while the machine is idle "
            "(architecture plan H / adaptive frequency; saves battery)"
        ),
    )
    idle_threshold_s: int = Field(
        default=120, ge=30, le=1800,
        description=(
            "Seconds of no input before a snapshot is marked idle. Larger "
            "values keep focused-but-quiet reading/thinking from being "
            "classified as away from the keyboard."
        ),
    )
    heartbeat_pulsetime_s: int = Field(
        default=10, ge=1, le=300, description="Heartbeat merge window in seconds"
    )

    # --- Data Retention ---
    event_retention_days: int = Field(
        default=30, description="Raw event retention in days (7-90)"
    )
    workflow_retention_days: int = Field(
        default=30,
        description=(
            "Workflow run retention in days (7-90). Completed/failed/cancelled "
            "runs older than this are cleaned up. Analyses and chat messages "
            "are preserved."
        ),
    )
    stale_run_timeout_minutes: int = Field(
        default=60,
        description="Minutes before a run stuck in 'running' status is marked 'failed'",
    )

    # --- OpenTelemetry (local only, ADR-003) ---
    otel_exporter: str = Field(
        default="sqlite",
        description="OTel span exporter: 'console' | 'in_memory' | 'sqlite'",
    )
    otel_retention_days: int = Field(
        default=30, ge=7, le=365,
        description="OTel span retention in days (cleaned by daily maintenance)",
    )

    @field_validator("event_retention_days")
    @classmethod
    def _validate_retention(cls, v: int) -> int:
        if not 7 <= v <= 90:
            msg = f"event_retention_days must be between 7 and 90, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("workflow_retention_days")
    @classmethod
    def _validate_workflow_retention(cls, v: int) -> int:
        if not 7 <= v <= 90:
            msg = f"workflow_retention_days must be between 7 and 90, got {v}"
            raise ValueError(msg)
        return v

    # --- Logging ---
    log: LogSettings = Field(default_factory=LogSettings)

    # --- Chat ---
    max_history_rounds: int = Field(
        default=10,
        ge=1,
        le=100,
        description=(
            "Max conversation rounds loaded for chat; the verbatim window is the "
            "last 6 turns, older loaded turns are folded into one summary message"
        ),
    )

    # --- Auto-intervention ---
    auto_intervention_min_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Min confidence to trigger auto-intervention"
    )
    auto_intervention_panel_confidence: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Confidence threshold for panel escalation"
    )
    intervention_start_hour: int = Field(
        default=8, ge=0, le=23, description="Auto-intervention window start hour (local 24h)"
    )
    intervention_end_hour: int = Field(
        default=23, ge=0, le=24,
        description="Auto-intervention window end hour (exclusive, local 24h)",
    )

    # --- Intervention throttle ---
    throttle_daily_limit: int = Field(
        default=3, ge=1, le=20, description="Max interventions per user per day"
    )
    throttle_type_limit: int = Field(
        default=2, ge=1, le=10, description="Max interventions of same type per day"
    )
    throttle_cooldown_hours: float = Field(
        default=2.0, ge=0.5, le=24.0, description="Min hours between interventions"
    )
    throttle_ignore_rate_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Ignore rate above which fatigue kicks in"
    )
    throttle_fatigue_daily_limit: int = Field(
        default=1, ge=1, le=10, description="Reduced daily cap when fatigued"
    )
    throttle_annoying_threshold: int = Field(
        default=3, ge=1, le=20, description="Annoying feedback count that reduces type limit"
    )

    # --- Human Review Interrupt (Todo 10) ---
    human_review_enabled: bool = Field(
        default=False,
        description="Enable human review interrupt on low-confidence/high-disagreement verdicts",
    )
    human_review_confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Confidence below which human review is triggered",
    )
    human_review_disagreement_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description=(
            "Disagreement strength above which human review is triggered "
            "(1.0 - agreement_strength)"
        ),
    )

    # --- Intervention work suppression ---
    intervention_work_suppress_enabled: bool = Field(
        default=True,
        description="Skip automated interventions while a work-state signal is active",
    )
    intervention_work_suppress_focus_threshold: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Deep-work focus_score threshold that implies a work state",
    )
    intervention_work_suppress_keypress_rate: float = Field(
        default=60.0,
        ge=0.0,
        le=1000.0,
        description="Sustained keypress rate per minute that implies a work state",
    )
    intervention_work_suppress_browser_work: bool = Field(
        default=False,
        description="Treat current browser work browsing as a work-state signal",
    )

    # --- Graph orchestration (ADR-005 — v2 graphs are the only paths) ---
    checkpointing_enabled: bool = Field(
        default=False, description="Enable LangGraph checkpoint persistence"
    )
    training_use_window_labels: bool = Field(
        default=True,
        description=(
            "Use user-calibrated behavior_feature_windows.label values as an "
            "additional training signal. Measured on real data (2026-08-20) to "
            "lift balanced accuracy 0.46->0.64, Brier 0.40->0.23 and make "
            "date-fold stability pass, activating the ML model. Quality-gate "
            "counts remain feedback-only. Set 0/False to disable."
        ),
    )
    new_analysis_graph: bool = Field(
        default=True,
        description=(
            "Deprecated compatibility flag; v2 AnalysisGraph is always active "
            "and this value no longer changes routing"
        ),
    )
    new_chat_graph: bool = Field(
        default=True,
        description=(
            "Deprecated compatibility flag; v2 ChatGraph is always active "
            "and this value no longer changes routing"
        ),
    )
    panel_fast_path_enabled: bool = Field(
        default=False,
        description=(
            "Enable the panel fast path (optimisation plan 2.2): when evidence "
            "coverage and quality are high, the rule engine is confident and "
            "nothing conflicts, skip the LLM panel and answer from the rule "
            "engine / a single expert; when evidence coverage is insufficient, "
            "return insufficient_data without spending LLM calls. Default OFF — "
            "the plan requires an offline replay showing >=95% key-conclusion "
            "agreement with the full panel and no increase in safety violations "
            "before this is switched on. Fast-path results still traverse the "
            "same safety word, schema and evidence validation as the panel path."
        ),
    )

    # --- LLM placeholder ---
    llm: LLMSettings = Field(default_factory=LLMSettings)

    @model_validator(mode="after")
    def _resolve_runtime_paths(self) -> Settings:
        """Anchor runtime paths to the platform data directory, never cwd."""
        if not self.data_dir.is_absolute():
            self.data_dir = _get_data_dir() / self.data_dir
        self.data_dir = self.data_dir.expanduser()

        if not self.models_dir.is_absolute():
            self.models_dir = self.data_dir / self.models_dir
        self.models_dir = self.models_dir.expanduser()

        if "{data_dir}" in self.db_url:
            self.db_url = self.db_url.format(data_dir=self.data_dir.as_posix())
        return self


SETTINGS: Settings | None = None


def get_settings() -> Settings:
    """Return cached application settings (global singleton).

    The .env file is loaded from the platform data directory (platformdirs).
    Environment variables with MINDFLOW_ prefix override .env values,
    which in turn override default values.
    """
    global SETTINGS

    if SETTINGS is not None:
        return SETTINGS

    data_dir = _get_data_dir()
    env_path = data_dir / ".env"

    SETTINGS = Settings(_env_file=env_path) if env_path.exists() else Settings()  # type: ignore[call-arg]

    return SETTINGS
