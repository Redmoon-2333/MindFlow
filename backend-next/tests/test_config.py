"""Tests for mindflow.config — Settings and validation.

Tests cover:
  - Default values
  - Environment variable overrides (via env vars or constructor kwargs)
  - event_retention_days boundary validation (7-90)
  - Collect interval bounds
"""

from __future__ import annotations

import os
from unittest import mock

import pytest
from pydantic_core import ValidationError

from mindflow.config import LLMSettings, LogSettings, Settings


class TestSettingsDefaults:
    """Verify default values match architecture document."""

    def test_db_url_resolved(self):
        """Default db_url is resolved via model_validator with platformdirs path."""
        settings = Settings()
        assert "{data_dir}" not in settings.db_url
        assert settings.db_url.startswith("sqlite+aiosqlite:///")
        assert "mindflow" in settings.db_url
        assert settings.db_url.endswith("/mindflow.db")

    def test_host_and_port_defaults(self):
        """Default host is 127.0.0.1 and port is 8765."""
        settings = Settings()
        assert settings.host == "127.0.0.1"
        assert settings.port == 8765

    def test_collect_interval_defaults(self):
        """Default collect interval is 5 seconds."""
        settings = Settings()
        assert settings.collect_interval_s == 5
        assert 1 <= settings.collect_interval_s <= 60

    def test_heartbeat_pulsetime_defaults(self):
        """Default heartbeat pulsetime is 10 seconds."""
        settings = Settings()
        assert settings.heartbeat_pulsetime_s == 10
        assert 1 <= settings.heartbeat_pulsetime_s <= 300

    def test_event_retention_defaults(self):
        """Default event retention is 30 days."""
        settings = Settings()
        assert settings.event_retention_days == 30

    def test_log_settings_defaults(self):
        """Default log settings exist with expected defaults."""
        settings = Settings()
        assert settings.log.level == "DEBUG"
        assert settings.log.rotation == "10 MB"
        assert settings.log.retention == "30 days"
        assert settings.log.compression == "gz"
        assert settings.log.json_format is False

    def test_llm_settings_defaults(self):
        """Default LLM settings are all None."""
        settings = Settings()
        assert settings.llm.api_key is None
        assert settings.llm.base_url is None
        assert settings.llm.model is None


class TestSettingsFromEnv:
    """Verify environment variable overrides work."""

    def test_override_db_url(self):
        """DB URL can be overridden via constructor."""
        settings = Settings(db_url="sqlite+aiosqlite:///custom.db")
        assert settings.db_url == "sqlite+aiosqlite:///custom.db"

    def test_override_port(self):
        """Port can be overridden."""
        settings = Settings(port=9999)
        assert settings.port == 9999

    def test_override_collect_interval(self):
        """Collect interval can be overridden within bounds."""
        settings = Settings(collect_interval_s=30)
        assert settings.collect_interval_s == 30

    def test_env_prefix(self):
        """Settings respect MINDFLOW_ env prefix."""
        with mock.patch.dict(os.environ, {"MINDFLOW_HOST": "0.0.0.0"}):
            settings = Settings()
            assert settings.host == "0.0.0.0"

    def test_env_override_retention(self):
        """Event retention can be set via env var."""
        with mock.patch.dict(os.environ, {"MINDFLOW_EVENT_RETENTION_DAYS": "45"}):
            settings = Settings()
            assert settings.event_retention_days == 45


class TestSettingsValidation:
    """Verify field validators reject invalid values."""

    @pytest.mark.parametrize("bad_days", [3, 6, 91, 100, -1, 0])
    def test_retention_below_minimum(self, bad_days):
        """Event retention below 7 is rejected."""
        with pytest.raises(ValidationError):
            Settings(event_retention_days=bad_days)

    @pytest.mark.parametrize("good_days", [7, 30, 60, 90])
    def test_retention_valid_boundaries(self, good_days):
        """Event retention at boundaries (7, 90) and valid values are accepted."""
        settings = Settings(event_retention_days=good_days)
        assert settings.event_retention_days == good_days

    @pytest.mark.parametrize("bad_interval", [0, -1, 61, 100])
    def test_collect_interval_bounds(self, bad_interval):
        """Collect interval is validated to be 1-60."""
        with pytest.raises(ValidationError):
            Settings(collect_interval_s=bad_interval)

    @pytest.mark.parametrize("bad_pulsetime", [0, -5, 301, 500])
    def test_heartbeat_pulsetime_bounds(self, bad_pulsetime):
        """Heartbeat pulsetime is validated to be 1-300."""
        with pytest.raises(ValidationError):
            Settings(heartbeat_pulsetime_s=bad_pulsetime)


class TestLogSettings:
    """Test LogSettings standalone."""

    def test_log_settings_defaults(self):
        """LogSettings has sensible defaults."""
        log = LogSettings()
        assert log.level == "DEBUG"
        assert log.rotation == "10 MB"
        assert log.retention == "30 days"
        assert log.compression == "gz"

    def test_log_settings_override(self):
        """LogSettings fields can be overridden."""
        log = LogSettings(level="INFO", json_format=True)
        assert log.level == "INFO"
        assert log.json_format is True


class TestLLMSettings:
    """Test LLMSettings placeholder."""

    def test_llm_settings_defaults(self):
        """LLMSettings defaults to None."""
        llm = LLMSettings()
        assert llm.api_key is None
        assert llm.base_url is None
        assert llm.model is None

    def test_llm_settings_override(self):
        """LLMSettings can be configured."""
        llm = LLMSettings(api_key="sk-test", model="deepseek-chat")
        assert llm.api_key == "sk-test"
        assert llm.model == "deepseek-chat"
