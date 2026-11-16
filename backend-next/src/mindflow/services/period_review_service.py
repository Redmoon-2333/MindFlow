"""Period review service — deterministic digest + optional AI narration.

Two layers, deliberately separated:

1. **Facts** (:meth:`PeriodReviewService.build_digest`): the loaded behaviour
   classifier re-scores every 5-minute window in the span, and
   ``domain.period_review.build_period_digest`` aggregates those scores plus the
   enumeration fields of the reminder log. The result is a closed whitelist of
   numbers, categories and dates — no window title, URL, file name, application
   name or reminder body can enter it, because no such field exists.

2. **Words** (:meth:`PeriodReviewService.review`): the digest JSON is sent to
   the LLM as-is, together with an evidence catalog and the retrieved
   literature. Every citation ID is checked against the catalog, every paper ID
   against the retrieved passages, and forbidden medical vocabulary is
   rejected; anything that fails falls back to the local template. The AI never
   sees a number it is not also allowed to cite.

The digest is also what the chat tool ``query_period_review`` returns, so the
assistant and the report page quote the same facts.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mindflow.agents.llm_gateway import GatewayNotConfiguredError, PanelLLMGateway
from mindflow.agents.policies import PERIOD_REVIEW_POLICY
from mindflow.agents.types import _contains_forbidden_words
from mindflow.domain.period_review import (
    PERIOD_DAYS_ALLOWED,
    REASON_LABELS_ZH,
    RESPONSE_KEYS,
    DistractionDriver,
    EvidenceRef,
    FallbackReason,
    InterventionFact,
    Narrative,
    NarrativeFinding,
    NarrativePaper,
    NarrativeSuggestion,
    PeriodDigest,
    build_period_digest,
    evidence_catalog,
    template_narrative,
)
from mindflow.services.literature_service import (
    LITERATURE_RULES,
    LiteratureContext,
    invalid_literature_citations,
    retrieve_literature,
)
from mindflow.services.llm_observability import llm_call_context
from mindflow.services.prediction_service import FocusPredictionService
from mindflow.time_utils import (
    TimezoneLike,
    business_day_bounds_utc,
    business_today,
    resolve_timezone,
)

#: Literature query for the review role. The role's own query expansion lives
#: in ``literature_service._ROLE_QUERIES``; this is the request-specific part.
_LITERATURE_QUERY = "self-monitoring reflection distraction interruption review"

#: Fixed system prompt (binding spec §4.3, verbatim). Split across adjacent
#: literals purely to satisfy the 100-column limit — the concatenated value is
#: the spec's text unchanged.
PERIOD_REVIEW_SYSTEM_PROMPT = (
    "你是 MindFlow 的“阶段复盘”助手。用户想回顾最近几天的专注情况。"
    "你只能使用 period_digest 里的统计数字：它们来自本地行为分类模型对每 5 分钟窗口的判断，"
    "以及系统提醒的记录。\n"
    "规则：\n"
    "1. 只写数字里有的事实。数据里没有网页、文件、聊天内容，"
    "不要猜测用户具体在看什么、写什么。\n"
    "2. findings 写 2–4 条发现，每条一句话，必须带至少一个 [证据: ID]，"
    "ID 只能取自 evidence_catalog。\n"
    "3. suggestions 写 1–3 条下一步建议，具体、能马上做、语气平和；"
    "引用 research_context 里的论文时用 [论文: chunk_id] 标注，不引用也可以。\n"
    "4. 如果 model.is_base_model 为 true，在一条 finding 里说明"
    "“基础模型的判断可能有偏差，可以在提醒里纠正”，并引用 [证据: model.version]。\n"
    "5. 不使用“诊断”“治疗”“患者”“处方”等医疗用语，不给用户贴性格标签。\n"
    '只输出 JSON，不要代码块：{"headline": "不超过30字的一句话总结", '
    '"findings": [{"text": "..."}], "suggestions": [{"text": "..."}]}'
)

#: Same pattern as ``agents/orchestrator.validate_citations`` — one definition
#: of what a well-formed evidence reference looks like.
_EVIDENCE_CITATION = re.compile(r"\[证据[:：]\s*([A-Za-z0-9_.]+)\s*\]")
#: Same pattern family as ``literature_service._CITATION``.
_PAPER_CITATION = re.compile(r"\[论文[:：]\s*([^\]\s]+)\s*\]")

_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


# ══════════════════════════════════════════════════════════════════════════════
# AI output schema (untrusted — validated before anything reaches the user)
# ══════════════════════════════════════════════════════════════════════════════


_RAW_ITEM_MAX_CHARS = 400
_CLEAN_ITEM_MAX_CHARS = 160


class _AINarrativeItem(BaseModel):
    """One finding or suggestion as the model returns it (text only)."""

    model_config = ConfigDict(extra="forbid")

    # The raw text still carries ``[证据: …]`` / ``[论文: …]`` markers, which can
    # easily double its length; the 160-character limit applies to the cleaned
    # sentence in ``_validate_narrative``.
    text: str = Field(min_length=1, max_length=_RAW_ITEM_MAX_CHARS)


class _AINarrative(BaseModel):
    """The JSON object the model is asked to produce."""

    model_config = ConfigDict(extra="forbid")

    headline: str = Field(min_length=1, max_length=40)
    findings: list[_AINarrativeItem] = Field(min_length=1, max_length=4)
    suggestions: list[_AINarrativeItem] = Field(min_length=1, max_length=3)


class PeriodReviewResult(BaseModel):
    """Everything one review returns: facts, the exact AI payload, and words.

    ``llm_payload`` is the compact JSON string that *was* (or, when AI was off,
    *would have been*) sent to the model. The UI shows it verbatim under
    "查看发给 AI 的内容" so the privacy claim is inspectable rather than
    asserted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    digest: PeriodDigest
    llm_payload: str
    ai_requested: bool
    narrative: Narrative
    generated_at: str


# ══════════════════════════════════════════════════════════════════════════════
# Service
# ══════════════════════════════════════════════════════════════════════════════


class PeriodReviewService:
    """Build period digests and, optionally, narrate them.

    Args:
        prediction_service: Scores the feature windows for the span.
        intervention_repo: Reminder log repository (``query_range``).
        gateway_provider: Returns the shared LLM gateway, or ``None`` when the
            application was assembled without one. Called per request so a
            gateway attached after construction is still picked up.
        timezone: Business timezone used for the local day and daypart buckets.
        ai_timeout_s: Deadline for the single narration call.
    """

    def __init__(
        self,
        *,
        prediction_service: FocusPredictionService,
        intervention_repo: Any,
        gateway_provider: Callable[[], PanelLLMGateway | None],
        timezone: TimezoneLike,
        ai_timeout_s: float = 90.0,
    ) -> None:
        self._prediction_service = prediction_service
        self._intervention_repo = intervention_repo
        self._gateway_provider = gateway_provider
        self._timezone = timezone
        self._ai_timeout_s = ai_timeout_s

    # ── Facts ──────────────────────────────────────────────────────────────

    async def build_digest(
        self,
        user_id: int,
        days: int,
        today: date | None = None,
    ) -> PeriodDigest:
        """Aggregate the last *days* local calendar days into a digest.

        Args:
            user_id: The user to review.
            days: One of :data:`PERIOD_DAYS_ALLOWED`.
            today: Override the last day (tests); defaults to the business
                today in the configured timezone.

        Returns:
            A frozen :class:`PeriodDigest`.

        Raises:
            ValueError: If *days* is not an allowed span.
        """
        if days not in PERIOD_DAYS_ALLOWED:
            msg = f"days must be one of {PERIOD_DAYS_ALLOWED}, got {days!r}"
            raise ValueError(msg)

        end_date = today or business_today(self._timezone)
        start_date = end_date - timedelta(days=days - 1)
        start_utc = business_day_bounds_utc(start_date, self._timezone)[0]
        end_utc = business_day_bounds_utc(end_date, self._timezone)[1]

        scoring = await self._prediction_service.score_windows(
            user_id, start_utc, end_utc
        )
        model_available = True
        if scoring.status == "no_model":
            model_available = False
        elif scoring.status not in {"ready", "no_data"}:
            # schema_mismatch / inference_error: keep the reminder facts and
            # let the window count (zero) land on "insufficient" rather than
            # claiming no model is loaded.
            logger.warning(
                "PeriodReviewService: window scoring degraded ({}): {}",
                scoring.status,
                scoring.reason,
            )

        model_version = scoring.model_version
        is_base_model = bool(model_version and model_version.startswith("base_"))

        facts = await self._intervention_facts(user_id, start_utc, end_utc)
        drivers: tuple[DistractionDriver, ...] = scoring.drivers

        return build_period_digest(
            scoring.windows,
            facts,
            drivers,
            start_date=start_date,
            end_date=end_date,
            tz=resolve_timezone(self._timezone),
            model_version=model_version,
            is_base_model=is_base_model,
            model_available=model_available,
        )

    async def _intervention_facts(
        self,
        user_id: int,
        start_utc: datetime,
        end_utc: datetime,
    ) -> list[InterventionFact]:
        """Read only the three allowed fields from the reminder log.

        ``title``, ``message`` and ``feedback_comment`` are present in the row
        dicts and are deliberately dropped here: this is the boundary where the
        privacy guarantee is enforced, not just documented.
        """
        rows = await self._intervention_repo.query_range(user_id, start_utc, end_utc)
        facts: list[InterventionFact] = []
        for row in rows:
            triggered_at = _parse_iso_utc(row.get("triggered_at"))
            if triggered_at is None:
                continue
            context = row.get("context_json")
            raw_reasons = (
                context.get("procrastination_types")
                if isinstance(context, dict)
                else None
            )
            reasons: tuple[str, ...] = ()
            if isinstance(raw_reasons, list):
                reasons = tuple(
                    dict.fromkeys(
                        str(reason)
                        for reason in raw_reasons
                        if str(reason) in REASON_LABELS_ZH
                    )
                )
            response = str(row.get("user_response") or "")
            facts.append(
                InterventionFact(
                    triggered_at_utc=triggered_at,
                    reasons=reasons,
                    response=response if response in RESPONSE_KEYS else "pending",
                )
            )
        return facts

    # ── Facts + words ──────────────────────────────────────────────────────

    async def review(
        self,
        user_id: int,
        days: int,
        use_ai: bool,
    ) -> PeriodReviewResult:
        """Build the digest and write the summary (AI when possible).

        Args:
            user_id: The user to review.
            days: One of :data:`PERIOD_DAYS_ALLOWED`.
            use_ai: Whether to request the AI narration. ``False`` never
                touches the gateway.

        Returns:
            A :class:`PeriodReviewResult`. The narrative always exists; when the
            AI path is not used or fails, it comes from the local template and
            ``fallback_reason`` says why.
        """
        digest = await self.build_digest(user_id, days)
        llm_payload = digest.model_dump_json()
        generated_at = datetime.now(UTC).isoformat()

        if not use_ai or digest.data_state != "ready":
            return PeriodReviewResult(
                digest=digest,
                llm_payload=llm_payload,
                ai_requested=use_ai,
                narrative=template_narrative(digest),
                generated_at=generated_at,
            )

        narrative, fallback_reason = await self._narrate(digest, llm_payload)
        if narrative is None:
            # Validation failed (or the call did): the local template carries
            # the reason so the UI can say why the words are not from the AI.
            narrative = template_narrative(digest).model_copy(
                update={"fallback_reason": fallback_reason}
            )
        return PeriodReviewResult(
            digest=digest,
            llm_payload=llm_payload,
            ai_requested=use_ai,
            narrative=narrative,
            generated_at=generated_at,
        )

    async def _narrate(
        self,
        digest: PeriodDigest,
        llm_payload: str,
    ) -> tuple[Narrative | None, FallbackReason | None]:
        """One AI call. Returns ``(narrative, None)`` or ``(None, reason)``."""
        gateway = self._gateway_provider()
        if gateway is None:
            return None, "AI 未配置"

        catalog = evidence_catalog(digest)
        literature = retrieve_literature(_LITERATURE_QUERY, role="period_review")
        system = PERIOD_REVIEW_SYSTEM_PROMPT + "\n\n" + LITERATURE_RULES
        user = literature.augment(
            "## period_digest\n"
            + llm_payload
            + "\n\n## evidence_catalog\n"
            + json.dumps(dict(catalog), ensure_ascii=False)
        )

        try:
            with llm_call_context(graph="period_review", node="narrate", role="period_review"):
                raw = await asyncio.wait_for(
                    gateway.complete(
                        system=system,
                        user=user,
                        model="chat",
                        policy=PERIOD_REVIEW_POLICY,
                    ),
                    timeout=self._ai_timeout_s,
                )
        except GatewayNotConfiguredError:
            return None, "AI 未配置"
        except TimeoutError:
            return None, "AI 响应超时"
        except Exception as exc:
            logger.opt(exception=True).warning(
                "PeriodReviewService: AI narration failed: {}", exc
            )
            return None, "AI 调用失败"

        return self._validate_narrative(raw, catalog, literature)

    def _validate_narrative(
        self,
        raw: str,
        catalog: Mapping[str, str],
        literature: LiteratureContext,
    ) -> tuple[Narrative | None, FallbackReason | None]:
        """Parse and check the model output; strip markers into structured refs.

        Every failure returns ``(None, "AI 输出未通过校验")`` — the caller then
        falls back to the template, so an unchecked sentence never reaches the
        user.
        """
        parsed = _parse_ai_narrative(raw)
        if parsed is None:
            return None, "AI 输出未通过校验"

        all_text = "\n".join(
            [parsed.headline, *(item.text for item in parsed.findings),
             *(item.text for item in parsed.suggestions)]
        )

        # ── Evidence IDs must exist in this digest's catalog ────────────────
        cited = _EVIDENCE_CITATION.findall(all_text)
        unknown = [citation for citation in cited if citation not in catalog]
        if unknown:
            logger.warning(
                "PeriodReviewService: unknown evidence IDs {} — rejecting output",
                unknown,
            )
            return None, "AI 输出未通过校验"
        for finding in parsed.findings:
            if not _EVIDENCE_CITATION.search(finding.text):
                logger.warning(
                    "PeriodReviewService: finding without evidence — rejecting output"
                )
                return None, "AI 输出未通过校验"

        # ── Paper IDs must exist in the retrieved context ───────────────────
        if invalid_literature_citations(all_text, literature.citation_ids):
            return None, "AI 输出未通过校验"

        # ── No medical vocabulary ───────────────────────────────────────────
        if _contains_forbidden_words(all_text) is not None:
            return None, "AI 输出未通过校验"

        hits = {hit.chunk_id: hit for hit in literature.hits}
        referenced: list[str] = []

        def _clean(text: str) -> str:
            stripped = _PAPER_CITATION.sub("", _EVIDENCE_CITATION.sub("", text))
            collapsed = re.sub(r"[ \t]{2,}", " ", stripped)
            # A marker removed just before punctuation leaves "分钟 。".
            return re.sub(r"[ \t]+([。，；：！？、）])", r"\1", collapsed).strip()

        cleaned_items = [
            _clean(item.text) for item in (*parsed.findings, *parsed.suggestions)
        ]
        if any(not text or len(text) > _CLEAN_ITEM_MAX_CHARS for text in cleaned_items):
            logger.warning(
                "PeriodReviewService: an item is empty or longer than {} characters "
                "after removing citation markers — rejecting output",
                _CLEAN_ITEM_MAX_CHARS,
            )
            return None, "AI 输出未通过校验"

        findings: list[NarrativeFinding] = []
        for finding in parsed.findings:
            ids = list(dict.fromkeys(_EVIDENCE_CITATION.findall(finding.text)))
            findings.append(
                NarrativeFinding(
                    text=_clean(finding.text),
                    evidence=[
                        EvidenceRef(id=item, label=catalog[item]) for item in ids
                    ],
                )
            )

        suggestions: list[NarrativeSuggestion] = []
        for suggestion in parsed.suggestions:
            ids = list(dict.fromkeys(_PAPER_CITATION.findall(suggestion.text)))
            referenced.extend(paper_id for paper_id in ids if paper_id not in referenced)
            suggestions.append(
                NarrativeSuggestion(
                    text=_clean(suggestion.text),
                    paper_ids=ids,
                )
            )

        papers = [
            NarrativePaper(
                chunk_id=paper_id,
                title=hits[paper_id].title,
                pages=list(hits[paper_id].pages),
            )
            for paper_id in referenced
            if paper_id in hits
        ]

        return (
            Narrative(
                source="ai",
                headline=_clean(parsed.headline),
                findings=findings,
                suggestions=suggestions,
                papers=papers,
            ),
            None,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _parse_iso_utc(value: object) -> datetime | None:
    """Parse a stored ISO timestamp into an aware UTC datetime."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


_MAX_FINDINGS = 4
_MAX_SUGGESTIONS = 3
_TOP_LEVEL_KEYS = frozenset({"headline", "findings", "suggestions"})


def _trim_items(data: dict[str, Any]) -> dict[str, Any]:
    """Keep at most 4 findings / 3 suggestions instead of rejecting a long list.

    Extra items are not wrong, only more than the page shows; every kept item is
    still validated. The base-model caveat (cites ``model.version``) is kept even
    when the model wrote it after the fourth finding.

    Unknown top-level keys (e.g. an invented ``disclaimer``) are dropped, not
    displayed: only the three whitelisted fields are ever read. Items keep
    ``extra="forbid"``.
    """
    dropped = sorted(str(key) for key in data if key not in _TOP_LEVEL_KEYS)
    if dropped:
        logger.info("PeriodReviewService: ignoring extra AI keys {}", dropped)
    trimmed = {key: value for key, value in data.items() if key in _TOP_LEVEL_KEYS}
    findings = data.get("findings")
    if isinstance(findings, list) and len(findings) > _MAX_FINDINGS:
        kept = findings[:_MAX_FINDINGS]
        caveat = next(
            (
                item
                for item in findings[_MAX_FINDINGS:]
                if isinstance(item, dict) and "model.version" in str(item.get("text", ""))
            ),
            None,
        )
        if caveat is not None:
            kept[-1] = caveat
        trimmed["findings"] = kept
    suggestions = data.get("suggestions")
    if isinstance(suggestions, list) and len(suggestions) > _MAX_SUGGESTIONS:
        trimmed["suggestions"] = suggestions[:_MAX_SUGGESTIONS]
    return trimmed


def _parse_ai_narrative(raw: str) -> _AINarrative | None:
    """Strip an optional code fence, parse JSON, and validate the schema."""
    text = raw.strip()
    fenced = _CODE_FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # Models sometimes wrap the object in a sentence; try the outermost {...}.
        start, end = text.find("{"), text.rfind("}")
        try:
            data = json.loads(text[start : end + 1]) if 0 <= start < end else None
        except json.JSONDecodeError:
            data = None
        if data is None:
            logger.warning(
                "PeriodReviewService: AI output was not valid JSON "
                "(length {}, starts with brace: {})",
                len(text),
                text.startswith("{"),
            )
            return None
    if isinstance(data, dict):
        data = _trim_items(data)
    try:
        return _AINarrative.model_validate(data)
    except ValidationError as exc:
        logger.warning(
            "PeriodReviewService: AI output failed schema validation: {}",
            [(".".join(str(part) for part in err["loc"]), err["type"]) for err in exc.errors()],
        )
        return None


__all__ = [
    "PERIOD_REVIEW_SYSTEM_PROMPT",
    "PeriodReviewResult",
    "PeriodReviewService",
]
