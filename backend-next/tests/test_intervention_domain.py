"""Tests for domain/intervention.py — pure data, no I/O.

Covers:
  - Intervention dataclass construction and immutability
  - InterventionIntensity enum values
  - InterventionResponse enum values
  - INTENSITY_TEMPLATES completeness
  - INTERVENTION_TYPE_LABELS completeness
  - Forbidden term compliance (NF-S7)
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mindflow.domain.intervention import (
    INTENSITY_TEMPLATES,
    INTERVENTION_TYPE_LABELS,
    Intervention,
    InterventionIntensity,
    InterventionResponse,
)


class TestInterventionDataclass:
    """Intervention frozen dataclass behavior."""

    def test_construct(self) -> None:
        """Basic construction with all fields."""
        now = datetime.now(UTC)
        intervention = Intervention(
            id="test-id-001",
            user_id=1,
            intervention_type="nudge",
            cbt_technique="behavioral_experiment",
            title="测试标题",
            message="测试消息",
            dismissible=True,
            created_at=now,
        )
        assert intervention.id == "test-id-001"
        assert intervention.user_id == 1
        assert intervention.intervention_type == "nudge"
        assert intervention.cbt_technique == "behavioral_experiment"
        assert intervention.title == "测试标题"
        assert intervention.message == "测试消息"
        assert intervention.dismissible is True
        assert intervention.created_at == now

    def test_frozen(self) -> None:
        """Cannot modify a frozen dataclass."""
        now = datetime.now(UTC)
        intervention = Intervention(
            id="test-id-002",
            user_id=1,
            intervention_type="task_breakdown",
            cbt_technique=None,
            title="标题",
            message="消息",
            dismissible=True,
            created_at=now,
        )
        with pytest.raises(AttributeError):
            intervention.title = "新标题"  # type: ignore[misc]

    def test_with_cbt_technique_none(self) -> None:
        """CBT technique may be None for generic interventions."""
        now = datetime.now(UTC)
        intervention = Intervention(
            id="test-id-003",
            user_id=1,
            intervention_type="nudge",
            cbt_technique=None,
            title="标题",
            message="消息",
            dismissible=True,
            created_at=now,
        )
        assert intervention.cbt_technique is None

    def test_all_intervention_types(self) -> None:
        """All four intervention types should be constructable."""
        now = datetime.now(UTC)
        types = [
            "task_breakdown",
            "nudge",
            "environment_optimization",
            "smart_prioritization",
        ]
        for t in types:
            intervention = Intervention(
                id=f"test-{t}",
                user_id=1,
                intervention_type=t,  # type: ignore[arg-type]
                cbt_technique=None,
                title="标题",
                message="消息",
                dismissible=True,
                created_at=now,
            )
            assert intervention.intervention_type == t


class TestInterventionIntensity:
    """InterventionIntensity enum."""

    def test_members(self) -> None:
        """Three intensity levels."""
        assert InterventionIntensity.GENTLE == "gentle"
        assert InterventionIntensity.STANDARD == "standard"
        assert InterventionIntensity.STRICT == "strict"

    def test_all_accounted_for(self) -> None:
        """All intensities have templates."""
        for intensity in InterventionIntensity:
            assert intensity in INTENSITY_TEMPLATES
            title_tmpl, body_tmpl = INTENSITY_TEMPLATES[intensity]
            assert "{detail}" in body_tmpl
            # GENTLE: title has type_label, body is short (no suggestion)
            if intensity == InterventionIntensity.GENTLE:
                assert "{type_label}" in title_tmpl
            # STANDARD: fixed title, body has suggestion
            if intensity == InterventionIntensity.STANDARD:
                assert "{suggestion}" in body_tmpl
            # STRICT: fixed title, body has suggestion
            if intensity == InterventionIntensity.STRICT:
                assert "{suggestion}" in body_tmpl


class TestInterventionResponse:
    """InterventionResponse enum."""

    def test_members(self) -> None:
        """Three response values."""
        assert InterventionResponse.ACCEPTED == "accepted"
        assert InterventionResponse.IGNORED == "ignored"
        assert InterventionResponse.DISMISSED == "dismissed"


class TestTemplateCompliance:
    """NF-S7 forbidden term compliance."""

    FORBIDDEN = {"诊断", "治疗", "患者", "处方"}

    def test_templates_no_forbidden_terms(self) -> None:
        """All message templates are NF-S7 compliant."""
        for intensity in InterventionIntensity:
            title_tmpl, body_tmpl = INTENSITY_TEMPLATES[intensity]
            full_text = title_tmpl + " " + body_tmpl
            for term in self.FORBIDDEN:
                assert term not in full_text, f"Forbidden term '{term}' in {intensity} template"

    def test_type_labels_no_forbidden_terms(self) -> None:
        """Chinese type labels are clean."""
        for label in INTERVENTION_TYPE_LABELS.values():
            for term in self.FORBIDDEN:
                assert term not in label, f"Forbidden term '{term}' in label '{label}'"


class TestTypeLabels:
    """INTERVENTION_TYPE_LABELS completeness."""

    VALID_TYPES = frozenset({
        "task_breakdown",
        "nudge",
        "environment_optimization",
        "smart_prioritization",
    })

    def test_all_types_have_labels(self) -> None:
        """Every valid intervention type has a Chinese label."""
        for t in self.VALID_TYPES:
            assert t in INTERVENTION_TYPE_LABELS, f"Missing label for {t}"

    def test_no_extra_keys(self) -> None:
        """No unexpected keys in labels dict."""
        for key in INTERVENTION_TYPE_LABELS:
            assert key in self.VALID_TYPES, f"Unexpected key: {key}"
