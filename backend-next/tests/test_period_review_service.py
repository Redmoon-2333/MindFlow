"""Tests for PeriodReviewService — digest assembly, privacy, and AI fallbacks.

Covers the binding spec's §6 list: AI success strips markers into
``evidence``/``paper_ids``; unknown evidence IDs, forbidden medical vocabulary
and malformed JSON all fall back to the template; a gateway exception, a
timeout and an unconfigured gateway each report their own reason; ``use_ai``
false never touches the gateway. The privacy test builds a reminder row full of
private strings and asserts none of them survives into the payload.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from mindflow.domain.period_review import PeriodDigest
from mindflow.services.period_review_service import (
    PeriodReviewResult,
    PeriodReviewService,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
TODAY = date(2026, 11, 16)
LEAKY_TITLE = "秘密项目.docx 修改"
LEAKY_URL = "https://example.com/private"
LEAKY_COMMENT = "私密"


# ══════════════════════════════════════════════════════════════════════════════
# Fakes
# ══════════════════════════════════════════════════════════════════════════════


class _FakeClassifier:
    """Deterministic 2-class classifier: every window is the same probability."""

    def __init__(self, p_distraction: float = 0.9) -> None:
        self._is_fitted = True
        self._p = p_distraction
        self.feature_names_: list[str] | None = None

    def predict_proba(self, matrix: Any) -> Any:
        import numpy as np

        rows = matrix.shape[0]
        return np.column_stack(
            [np.full(rows, self._p), np.full(rows, 1.0 - self._p)]
        )

    def get_feature_importance(self) -> dict[str, float]:
        return {}


class _FakeModelManager:
    def __init__(
        self,
        p_distraction: float = 0.9,
        version: str = "base_v4_20261111_r2",
    ) -> None:
        self.classifier = _FakeClassifier(p_distraction)
        self.current_version_tag = version


class _FakeTelemetryRepo:
    """Serves feature-window rows for the request's range."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.calls: list[tuple[Any, Any]] = []

    async def list_feature_windows_in_range(
        self,
        user_id: int,
        start: datetime,
        end: datetime,
        feature_schema_version: int | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append((start, end))
        return list(self._rows)


class _FakeInterventionRepo:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.calls: list[tuple[Any, Any]] = []

    async def query_range(
        self, user_id: int, start: datetime, end: datetime
    ) -> list[dict[str, Any]]:
        self.calls.append((start, end))
        return list(self._rows)


class _FakeGateway:
    """Stands in for ``PanelLLMGateway``."""

    def __init__(
        self,
        *,
        reply: str | None = None,
        exc: BaseException | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self.reply = reply
        self.exc = exc
        self.delay_s = delay_s
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        system: str,
        user: str,
        model: str = "chat",
        policy: Any = None,
    ) -> str:
        self.calls.append(
            {"system": system, "user": user, "model": model, "policy": policy}
        )
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.exc is not None:
            raise self.exc
        assert self.reply is not None
        return self.reply


def _feature_row(start_local: datetime, *, task_type_code: float = 1.0) -> dict[str, Any]:
    """A feature-window row with the V4 vocabulary filled in."""
    from mindflow.domain.feature_schema import V2_FEATURE_NAMES

    features = {name: 0.5 for name in V2_FEATURE_NAMES}
    features["task_type_code"] = task_type_code
    start_utc = start_local.astimezone(UTC)
    return {
        "window_start_utc": start_utc.isoformat(),
        "window_end_utc": (start_utc + timedelta(minutes=5)).isoformat(),
        "features_json": json.dumps(features),
        "feature_schema_version": 4,
        "quality_json": None,
    }


def _ready_rows(count: int = 24) -> list[dict[str, Any]]:
    """Enough windows for ``data_state == "ready"`` on the last two days."""
    base = datetime(2026, 11, 15, 9, 0, tzinfo=SHANGHAI)
    return [_feature_row(base + timedelta(minutes=5 * index)) for index in range(count)]


def _service(
    *,
    rows: list[dict[str, Any]] | None = None,
    interventions: list[dict[str, Any]] | None = None,
    gateway: Any = None,
    gateway_provider: Any = None,
    ai_timeout_s: float = 45.0,
    p_distraction: float = 0.9,
    version: str = "base_v4_20261111_r2",
) -> PeriodReviewService:
    from mindflow.services.prediction_service import FocusPredictionService

    prediction_service = FocusPredictionService(
        telemetry_repository=_FakeTelemetryRepo(
            _ready_rows() if rows is None else rows
        ),
        model_manager=_FakeModelManager(p_distraction, version),
    )
    provider = gateway_provider
    if provider is None:
        provider = lambda: gateway  # noqa: E731
    return PeriodReviewService(
        prediction_service=prediction_service,
        intervention_repo=_FakeInterventionRepo(interventions or []),
        gateway_provider=provider,
        timezone="Asia/Shanghai",
        ai_timeout_s=ai_timeout_s,
    )


def _ai_reply(
    findings: list[str],
    suggestions: list[str] | None = None,
    headline: str = "分心集中在娱乐上",
) -> str:
    default_suggestions = ["把娱乐应用留到任务结束后再打开。"]
    return json.dumps(
        {
            "headline": headline,
            "findings": [{"text": text} for text in findings],
            "suggestions": [{"text": text} for text in (suggestions or default_suggestions)],
        },
        ensure_ascii=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# build_digest
# ══════════════════════════════════════════════════════════════════════════════


async def test_build_digest_rejects_unsupported_span() -> None:
    with pytest.raises(ValueError, match="days must be one of"):
        await _service().build_digest(1, 5)


async def test_build_digest_spans_the_requested_local_days() -> None:
    service = _service()
    digest = await service.build_digest(1, 7, today=TODAY)
    assert digest.period.end_date == TODAY
    assert digest.period.start_date == date(2026, 11, 10)
    assert digest.period.days == 7
    assert len(digest.daily) == 7


async def test_build_digest_detects_base_model_from_version_prefix() -> None:
    digest = await _service(version="base_v4_20261111_r2").build_digest(1, 7, today=TODAY)
    assert digest.model.is_base_model is True
    assert digest.model.version == "base_v4_20261111_r2"


async def test_build_digest_personal_model_is_not_base() -> None:
    digest = await _service(version="20260726_v2").build_digest(1, 7, today=TODAY)
    assert digest.model.is_base_model is False


async def test_build_digest_without_model_reports_no_model() -> None:
    from mindflow.services.prediction_service import FocusPredictionService

    service = PeriodReviewService(
        prediction_service=FocusPredictionService(
            telemetry_repository=_FakeTelemetryRepo([])
        ),
        intervention_repo=_FakeInterventionRepo([]),
        gateway_provider=lambda: None,
        timezone="Asia/Shanghai",
    )
    digest = await service.build_digest(1, 3, today=TODAY)
    assert digest.data_state == "no_model"
    assert digest.totals.active_minutes == 0
    assert len(digest.daily) == 3


async def test_build_digest_without_windows_is_insufficient() -> None:
    digest = await _service(rows=[]).build_digest(1, 3, today=TODAY)
    assert digest.data_state == "insufficient"


async def test_build_digest_scores_every_window() -> None:
    digest = await _service().build_digest(1, 3, today=TODAY)
    assert digest.totals.active_minutes == 120  # 24 windows × 5 min


async def test_build_digest_reads_reasons_and_responses() -> None:
    interventions = [
        {
            "triggered_at": datetime(2026, 11, 15, 2, 0, tzinfo=UTC).isoformat(),
            "context_json": {"procrastination_types": ["impulsivity", "not_a_type"]},
            "user_response": "accepted",
        },
        {
            "triggered_at": datetime(2026, 11, 15, 3, 0, tzinfo=UTC).isoformat(),
            "context_json": None,
            "user_response": None,
        },
    ]
    digest = await _service(interventions=interventions).build_digest(1, 3, today=TODAY)
    assert digest.interventions.total == 2
    assert digest.interventions.accepted == 1
    assert digest.interventions.pending == 1
    assert [item.type for item in digest.interventions.by_reason] == ["impulsivity"]


# ══════════════════════════════════════════════════════════════════════════════
# Privacy
# ══════════════════════════════════════════════════════════════════════════════


async def test_payload_never_contains_window_or_reminder_text() -> None:
    """The digest is built from numbers only; private text cannot ride along."""
    interventions = [
        {
            "triggered_at": datetime(2026, 11, 15, 2, 0, tzinfo=UTC).isoformat(),
            "context_json": {"procrastination_types": ["impulsivity"]},
            "user_response": "accepted",
            "title": LEAKY_TITLE,
            "message": LEAKY_URL,
            "feedback_comment": LEAKY_COMMENT,
        }
    ]
    # Feature rows carry the same private strings in fields the service must
    # never copy into the digest.
    rows = _ready_rows()
    for row in rows:
        row["window_title"] = LEAKY_TITLE
        row["url"] = LEAKY_URL
        row["app_name"] = LEAKY_TITLE

    service = _service(rows=rows, interventions=interventions)
    result = await service.review(1, 7, use_ai=False)
    payload = result.llm_payload
    for secret in (LEAKY_TITLE, LEAKY_URL, LEAKY_COMMENT, "example.com", ".docx"):
        assert secret not in payload, secret
    # …and the same holds for the serialised digest the UI receives.
    assert LEAKY_URL not in result.digest.model_dump_json()


async def test_digest_models_forbid_extra_keys() -> None:
    digest = await _service().build_digest(1, 3, today=TODAY)
    assert PeriodDigest.model_config["extra"] == "forbid"
    for model in (
        digest.period,
        digest.model,
        digest.totals,
        digest.episodes,
        digest.interventions,
        *digest.daily,
        *digest.distraction_by_daypart,
    ):
        assert type(model).model_config["extra"] == "forbid", type(model).__name__


# ══════════════════════════════════════════════════════════════════════════════
# AI path
# ══════════════════════════════════════════════════════════════════════════════


async def test_valid_ai_output_is_used_and_markers_are_stripped() -> None:
    gateway = _FakeGateway(
        reply=_ai_reply(
            findings=[
                "有效记录 2 小时，分心约占 100%。[证据: totals.active_minutes]",
                "最近 2 天分心 1 次。[证据: episodes.count][证据: episodes.longest_minutes]",
            ],
            suggestions=["下午的任务拆小一点。[论文: lyngs_2019.p002.c01]"],
        )
    )
    result = await _service(gateway=gateway).review(1, 3, use_ai=True)
    assert isinstance(result, PeriodReviewResult)
    assert result.narrative.source == "ai"
    assert result.narrative.fallback_reason is None
    assert "[证据:" not in result.narrative.findings[0].text
    assert "[论文:" not in result.narrative.suggestions[0].text
    assert result.narrative.findings[0].evidence[0].id == "totals.active_minutes"
    assert result.narrative.findings[0].evidence[0].label.startswith("有效记录时长：")
    assert result.narrative.suggestions[0].paper_ids == ["lyngs_2019.p002.c01"]
    assert result.narrative.papers[0].chunk_id == "lyngs_2019.p002.c01"
    assert result.ai_requested is True
    assert len(gateway.calls) == 1
    # The request must carry the digest and the catalog, and use our policy.
    assert "## period_digest" in gateway.calls[0]["user"]
    assert "## evidence_catalog" in gateway.calls[0]["user"]
    assert gateway.calls[0]["model"] == "chat"
    assert gateway.calls[0]["policy"].role == "period_review"


async def test_ai_payload_is_compact_json_of_the_digest() -> None:
    gateway = _FakeGateway(reply=_ai_reply(["a。[证据: episodes.count]"]))
    result = await _service(gateway=gateway).review(1, 3, use_ai=True)
    parsed = json.loads(result.llm_payload)
    assert parsed["period"]["days"] == 3
    assert ", " not in result.llm_payload  # model_dump_json is compact


async def test_unknown_evidence_id_falls_back() -> None:
    gateway = _FakeGateway(reply=_ai_reply(["分心很多。[证据: totals.secret_metric]"]))
    result = await _service(gateway=gateway).review(1, 3, use_ai=True)
    assert result.narrative.source == "template"
    assert result.narrative.fallback_reason == "AI 输出未通过校验"


async def test_finding_without_evidence_falls_back() -> None:
    gateway = _FakeGateway(reply=_ai_reply(["分心很多，但没给证据。"]))
    result = await _service(gateway=gateway).review(1, 3, use_ai=True)
    assert result.narrative.fallback_reason == "AI 输出未通过校验"


async def test_forbidden_medical_word_falls_back() -> None:
    gateway = _FakeGateway(
        reply=_ai_reply(["建议诊断一下。[证据: episodes.count]"])
    )
    result = await _service(gateway=gateway).review(1, 3, use_ai=True)
    assert result.narrative.source == "template"
    assert result.narrative.fallback_reason == "AI 输出未通过校验"


async def test_unknown_paper_id_falls_back() -> None:
    gateway = _FakeGateway(
        reply=_ai_reply(
            ["分心很多。[证据: episodes.count]"],
            suggestions=["试试番茄钟。[论文: nonexistent_paper_2099]"],
        )
    )
    result = await _service(gateway=gateway).review(1, 3, use_ai=True)
    assert result.narrative.fallback_reason == "AI 输出未通过校验"


async def test_malformed_json_falls_back() -> None:
    gateway = _FakeGateway(reply="{not json")
    result = await _service(gateway=gateway).review(1, 3, use_ai=True)
    assert result.narrative.fallback_reason == "AI 输出未通过校验"


async def test_schema_violation_falls_back() -> None:
    reply = json.dumps(
        {"headline": "x", "findings": [{"text": "a。[证据: episodes.count]", "extra": 1}],
         "suggestions": [{"text": "b。"}]},
        ensure_ascii=False,
    )
    result = await _service(gateway=_FakeGateway(reply=reply)).review(1, 3, use_ai=True)
    assert result.narrative.fallback_reason == "AI 输出未通过校验"


async def test_code_fenced_json_is_accepted() -> None:
    inner = _ai_reply(["分心很多。[证据: episodes.count]"])
    result = await _service(gateway=_FakeGateway(reply=f"```json\n{inner}\n```")).review(
        1, 3, use_ai=True
    )
    assert result.narrative.source == "ai"


async def test_gateway_exception_falls_back_to_call_failed() -> None:
    gateway = _FakeGateway(exc=RuntimeError("upstream exploded"))
    result = await _service(gateway=gateway).review(1, 3, use_ai=True)
    assert result.narrative.source == "template"
    assert result.narrative.fallback_reason == "AI 调用失败"


async def test_timeout_falls_back_to_timeout_reason() -> None:
    gateway = _FakeGateway(reply=_ai_reply(["x。[证据: episodes.count]"]), delay_s=0.5)
    result = await _service(gateway=gateway, ai_timeout_s=0.01).review(1, 3, use_ai=True)
    assert result.narrative.fallback_reason == "AI 响应超时"


async def test_missing_gateway_reports_not_configured() -> None:
    result = await _service(gateway=None).review(1, 3, use_ai=True)
    assert result.narrative.source == "template"
    assert result.narrative.fallback_reason == "AI 未配置"


async def test_gateway_not_configured_error_reports_not_configured() -> None:
    from mindflow.agents.llm_gateway import GatewayNotConfiguredError

    gateway = _FakeGateway(exc=GatewayNotConfiguredError("no key"))
    result = await _service(gateway=gateway).review(1, 3, use_ai=True)
    assert result.narrative.fallback_reason == "AI 未配置"


async def test_use_ai_false_never_calls_the_gateway() -> None:
    gateway = _FakeGateway(reply=_ai_reply(["x。[证据: episodes.count]"]))
    result = await _service(gateway=gateway).review(1, 3, use_ai=False)
    assert gateway.calls == []
    assert result.ai_requested is False
    assert result.narrative.source == "template"
    assert result.narrative.fallback_reason is None


async def test_non_ready_data_skips_the_ai_call() -> None:
    gateway = _FakeGateway(reply=_ai_reply(["x。[证据: episodes.count]"]))
    result = await _service(rows=[], gateway=gateway).review(1, 3, use_ai=True)
    assert gateway.calls == []
    assert result.narrative.source == "template"
    assert result.narrative.fallback_reason is None
    assert result.narrative.headline == "记录还不够，暂时总结不出规律"


async def test_review_is_called_at_most_once() -> None:
    gateway = _FakeGateway(exc=RuntimeError("boom"))
    await _service(gateway=gateway).review(1, 3, use_ai=True)
    assert len(gateway.calls) == 1


async def test_base_model_caveat_can_cite_model_version() -> None:
    reply = _ai_reply(
        [
            "分心主要在娱乐上。[证据: episodes.count]",
            "基础模型的判断可能有偏差，可以在提醒里纠正。[证据: model.version]",
        ]
    )
    result = await _service(gateway=_FakeGateway(reply=reply)).review(1, 7, use_ai=True)
    assert result.narrative.source == "ai"
    assert result.narrative.fallback_reason is None
    caveat = result.narrative.findings[1]
    assert [ref.id for ref in caveat.evidence] == ["model.version"]
    assert caveat.evidence[0].label.startswith("当前模型：")


async def test_headline_markers_are_stripped() -> None:
    reply = _ai_reply(
        ["分心主要在娱乐上。[证据: episodes.count]"],
        headline="这周分心偏多 [证据: episodes.count]",
    )
    result = await _service(gateway=_FakeGateway(reply=reply)).review(1, 7, use_ai=True)
    assert result.narrative.source == "ai"
    assert "[证据" not in result.narrative.headline


async def test_citation_markers_do_not_count_towards_item_length() -> None:
    markers = "".join(
        f"[证据: {item}]"
        for item in (
            "totals.active_minutes",
            "totals.focused_minutes",
            "totals.distracted_minutes",
            "totals.distracted_share",
            "episodes.count",
            "episodes.longest_minutes",
        )
    )
    sentence = "这几天分心偏多，主要集中在娱乐上，最长一次持续很久。"
    assert len(sentence + markers) > 160
    result = await _service(gateway=_FakeGateway(reply=_ai_reply([sentence + markers]))).review(
        1, 7, use_ai=True
    )
    assert result.narrative.source == "ai"
    assert result.narrative.findings[0].text == sentence


async def test_overlong_clean_sentence_still_falls_back() -> None:
    sentence = "分" * 161 + "[证据: episodes.count]"
    result = await _service(gateway=_FakeGateway(reply=_ai_reply([sentence]))).review(
        1, 7, use_ai=True
    )
    assert result.narrative.source == "template"
    assert result.narrative.fallback_reason == "AI 输出未通过校验"


async def test_extra_findings_are_trimmed_and_caveat_is_kept() -> None:
    reply = _ai_reply(
        [
            "第一条。[证据: episodes.count]",
            "第二条。[证据: episodes.count]",
            "第三条。[证据: episodes.count]",
            "第四条。[证据: episodes.count]",
            "第五条。[证据: episodes.count]",
            "基础模型的判断可能有偏差，可以在提醒里纠正。[证据: model.version]",
        ],
        suggestions=["一。", "二。", "三。", "四。"],
    )
    result = await _service(gateway=_FakeGateway(reply=reply)).review(1, 7, use_ai=True)
    assert result.narrative.source == "ai"
    assert [f.text for f in result.narrative.findings][:3] == ["第一条。", "第二条。", "第三条。"]
    assert result.narrative.findings[3].evidence[0].id == "model.version"
    assert len(result.narrative.suggestions) == 3


async def test_json_wrapped_in_prose_is_accepted() -> None:
    inner = _ai_reply(["分心很多。[证据: episodes.count]"])
    reply = f"下面是总结：\n{inner}\n以上。"
    result = await _service(gateway=_FakeGateway(reply=reply)).review(1, 3, use_ai=True)
    assert result.narrative.source == "ai"


async def test_no_space_left_before_punctuation_after_marker_removal() -> None:
    reply = _ai_reply(["专注 1799 分钟 [证据: episodes.count]。"])
    result = await _service(gateway=_FakeGateway(reply=reply)).review(1, 3, use_ai=True)
    assert result.narrative.findings[0].text == "专注 1799 分钟。"


async def test_extra_top_level_keys_are_ignored() -> None:
    data = json.loads(_ai_reply(["分心很多。[证据: episodes.count]"]))
    data["disclaimer_placeholder"] = "不展示"
    reply = json.dumps(data, ensure_ascii=False)
    result = await _service(gateway=_FakeGateway(reply=reply)).review(1, 3, use_ai=True)
    assert result.narrative.source == "ai"
    assert "不展示" not in result.model_dump_json()
