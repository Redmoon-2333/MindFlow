"""OpenAPI generation does not initialise private runtime state."""

import runpy
from pathlib import Path
from unittest.mock import Mock


def test_export_does_not_load_private_settings_or_configure_logging(monkeypatch) -> None:
    import mindflow.app
    import mindflow.config

    private_settings = Mock(side_effect=AssertionError("private settings must not be loaded"))
    logging = Mock(side_effect=AssertionError("logging must not be configured"))
    monkeypatch.setattr(mindflow.config, "get_settings", private_settings)
    monkeypatch.setattr(mindflow.app, "setup_logging", logging)
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_openapi.py"
    export = runpy.run_path(str(script))["export_openapi"]
    schema = export()
    assert "/api/v1/analytics/training-jobs/{job_id}" in schema["paths"]
    private_settings.assert_not_called()
    logging.assert_not_called()
