"""Tests for the explicit v4 feature-window backfill CLI (plan item 3).

Covers ``python -m mindflow.telemetry rebuild-features`` end to end against real
SQLite databases: preview writes nothing, ``--apply`` rebuilds from the retained
raw events, a day block without raw data is reported instead of skipped, a
failed block is retryable with the same parameters, the requested range is
clamped to the 180-day rebuild boundary, v3 labels are inherited while an
existing v4 label wins, and ``features_json`` never disagrees with the explicit
``f01..f28`` columns of the same row.

Two shapes are used on purpose:

* the CLI-level tests drive ``main()`` exactly as an operator would (their own
  temp database file, a fresh event loop per call) and assert on the JSON that
  reaches stdout;
* the core-level tests drive ``cli.run_rebuild_features()`` with the shared
  ``session_factory``/``create_tables`` fixtures, which is where a fixed
  ``now_utc`` makes the retention clamp deterministic.
"""

from __future__ import annotations

import ast
import asyncio
import json
import subprocess
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, TypeVar

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.events import make_event
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION, V2_FEATURE_NAMES
from mindflow.infrastructure.database import create_engine, create_session_factory
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
    activity_events,
)
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.infrastructure.schema import behavior_feature_windows, metadata
from mindflow.services.telemetry_service import _FEATURE_REBUILD_MAX_DAYS
from mindflow.telemetry import __main__ as cli

_LEGACY_VERSION = FEATURE_SCHEMA_VERSION - 1
_FEATURE_COLUMNS = tuple(f"f{index:02d}" for index in range(1, len(V2_FEATURE_NAMES) + 1))
_T = TypeVar("_T")

# One event of this length starting on a 5-minute boundary covers exactly two
# feature windows, which keeps the expected counts readable.
_EVENT_SECONDS = 600


def _day(days_ago: int) -> date:
    """A whole UTC calendar day comfortably inside the 180-day rebuild cut."""
    return (datetime.now(UTC) - timedelta(days=days_ago)).date()


def _at(day: date, hour: int = 8, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=UTC)


def _midnight(day: date) -> datetime:
    """UTC midnight — the boundary every rebuild block starts on."""
    return datetime.combine(day, time.min, tzinfo=UTC)


def _features(**overrides: float) -> str:
    payload: dict[str, float] = dict.fromkeys(V2_FEATURE_NAMES, 0.0)
    payload.update(overrides)
    return json.dumps(payload)


def _feature_row(
    start: datetime,
    *,
    user_id: int = 1,
    feature_schema_version: int = FEATURE_SCHEMA_VERSION,
    label: str | None = None,
    features_json: str | None = None,
    **overrides: float,
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "window_start_utc": start,
        "window_end_utc": start + timedelta(minutes=5),
        "feature_schema_version": feature_schema_version,
        "features_json": features_json
        if features_json is not None
        else _features(**overrides),
        "label": label,
        "quality_json": None,
    }


async def _seed_event(
    activity: SQLAlchemyActivityRepository,
    day: date,
    *,
    hour: int = 8,
    user_id: int = 1,
    duration_s: float = _EVENT_SECONDS,
    process_name: str = "code.exe",
) -> None:
    await activity.append_event(
        make_event(
            user_id=user_id,
            timestamp_utc=_at(day, hour),
            duration_s=duration_s,
            process_name=process_name,
            is_idle=False,
        )
    )


@dataclass(frozen=True)
class _Handles:
    """Repository handles for one call against the temp database."""

    session_factory: async_sessionmaker[AsyncSession]
    activity: SQLAlchemyActivityRepository
    telemetry: TelemetryRepository

    async def v4_rows(self, user_id: int = 1) -> list[dict[str, Any]]:
        return await self.telemetry.list_feature_windows(
            user_id, feature_schema_version=FEATURE_SCHEMA_VERSION
        )

    async def legacy_rows(self, user_id: int = 1) -> list[dict[str, Any]]:
        return await self.telemetry.list_feature_windows(
            user_id, feature_schema_version=_LEGACY_VERSION
        )


@dataclass(frozen=True)
class _Db:
    """A temp SQLite database, reached through a fresh event loop per call.

    The CLI owns its event loop in production (``asyncio.run`` inside
    ``main()``), so the tests that drive ``main()`` must not hold a loop open.
    """

    path: Path

    def run(self, body: Callable[[_Handles], Awaitable[_T]]) -> _T:
        return asyncio.run(self._run(body))

    async def _run(self, body: Callable[[_Handles], Awaitable[_T]]) -> _T:
        engine = create_engine(f"sqlite+aiosqlite:///{self.path.as_posix()}")
        try:
            async with engine.begin() as connection:
                await connection.run_sync(metadata.create_all)
                await connection.run_sync(activity_events.metadata.create_all)
            session_factory = create_session_factory(engine)
            handles = _Handles(
                session_factory=session_factory,
                activity=SQLAlchemyActivityRepository(session_factory),
                telemetry=TelemetryRepository(session_factory),
            )
            return await body(handles)
        finally:
            await engine.dispose()


@pytest.fixture
def db(tmp_path: Path) -> _Db:
    return _Db(tmp_path / "rebuild_cli.db")


def _argv(db: _Db, start: date, end: date, *extra: str) -> list[str]:
    return [
        "rebuild-features",
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
        "--user-id",
        "1",
        "--database-path",
        str(db.path),
        *extra,
    ]


def _run_cli(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> tuple[int, dict[str, Any], str]:
    """Run ``main()`` and return (exit code, parsed JSON summary, stderr)."""
    code = cli.main(argv)
    captured = capsys.readouterr()
    payload = json.loads(captured.out) if captured.out.strip() else {}
    return code, payload, captured.err


# ── Preview / apply ───────────────────────────────────────────────────────


def test_preview_writes_nothing_and_reports_the_blocks_it_would_rebuild(
    db: _Db, capsys: pytest.CaptureFixture[str]
) -> None:
    with_day, empty_day = _day(3), _day(2)

    def seed(handles: _Handles) -> Awaitable[None]:
        return _seed_event(handles.activity, with_day)

    db.run(seed)
    code, summary, stderr = _run_cli(_argv(db, with_day, empty_day), capsys)

    assert code == 0
    assert summary["mode"] == "preview"
    assert summary["applied"] is False
    assert summary["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert summary["feature_column_count"] == len(V2_FEATURE_NAMES) == 28
    assert summary["chunk_days"] == 1
    assert summary["backup_required"] is True
    assert summary["clamped"] is False
    assert summary["effective_start"] == _midnight(with_day).isoformat()
    assert summary["effective_end"] == _midnight(empty_day + timedelta(days=1)).isoformat()
    assert [block["block_start"] for block in summary["blocks"]] == [
        _midnight(with_day).isoformat(),
        _midnight(empty_day).isoformat(),
    ]
    assert [block["windows_rebuilt"] for block in summary["blocks"]] == [2, 0]
    assert [block["windows_written"] for block in summary["blocks"]] == [0, 0]
    assert [block["status"] for block in summary["blocks"]] == ["preview", "missing_raw_data"]
    assert summary["totals"]["windows_written"] == 0

    # Nothing was written, and the report says what applying would do.
    assert db.run(lambda handles: handles.v4_rows()) == []
    assert "Preview only" in stderr
    assert "[1/2]" in stderr and "[2/2]" in stderr


def test_apply_rebuilds_windows_for_blocks_with_raw_events(
    db: _Db, capsys: pytest.CaptureFixture[str]
) -> None:
    first_day, second_day = _day(4), _day(3)

    async def seed(handles: _Handles) -> None:
        await _seed_event(handles.activity, first_day)
        await _seed_event(handles.activity, second_day, hour=14)

    db.run(seed)
    code, summary, stderr = _run_cli(_argv(db, first_day, second_day, "--apply"), capsys)

    assert code == 0
    assert summary["mode"] == "apply"
    assert summary["totals"] == {
        "blocks": 2,
        "blocks_succeeded": 2,
        "blocks_failed": 0,
        "blocks_missing_raw_data": 0,
        "windows_rebuilt": 4,
        "windows_written": 4,
        "labels_inherited": 0,
        "labels_preserved": 0,
    }
    for block in summary["blocks"]:
        assert block["status"] == "rebuilt"
        assert block["succeeded"] is True
        assert block["failed"] is False
        assert block["windows_rebuilt"] == block["windows_written"] == 2

    rows = db.run(lambda handles: handles.v4_rows())
    assert [row["window_start_utc"] for row in rows] == [
        _at(first_day).isoformat(),
        _at(first_day, 8, 5).isoformat(),
        _at(second_day, 14).isoformat(),
        _at(second_day, 14, 5).isoformat(),
    ]
    assert all(row["feature_schema_version"] == FEATURE_SCHEMA_VERSION for row in rows)
    assert all(row["quality_json"] for row in rows), "per-window quality is recorded"
    # WARNING about the operator-owned backup is printed before any write.
    assert "WARNING: --apply writes v4 feature windows" in stderr
    assert "rebuilt=2 written=2" in stderr


def test_block_without_raw_data_is_reported_and_exit_code_stays_zero(
    db: _Db, capsys: pytest.CaptureFixture[str]
) -> None:
    event_day, empty_day = _day(5), _day(4)

    def seed(handles: _Handles) -> Awaitable[None]:
        return _seed_event(handles.activity, event_day)

    db.run(seed)
    code, summary, stderr = _run_cli(_argv(db, event_day, empty_day, "--apply"), capsys)

    assert code == 0
    assert summary["totals"]["blocks_missing_raw_data"] == 1
    assert summary["totals"]["blocks_failed"] == 0
    rebuilt, missing = summary["blocks"]
    assert rebuilt["missing_raw_data"] is False and rebuilt["succeeded"] is True
    assert missing["missing_raw_data"] is True
    assert missing["succeeded"] is True and missing["failed"] is False
    assert missing["status"] == "missing_raw_data"
    assert "activity events" in missing["reason"]
    assert missing["windows_rebuilt"] == 0 and missing["windows_written"] == 0
    assert "missing_raw_data" in stderr, "the empty block is reported, not skipped"
    assert db.run(lambda handles: handles.v4_rows()) != []


def test_rerun_with_same_parameters_is_idempotent(
    db: _Db, capsys: pytest.CaptureFixture[str]
) -> None:
    day = _day(6)

    async def seed(handles: _Handles) -> None:
        await _seed_event(handles.activity, day)
        await handles.telemetry.upsert_feature_windows([
            _feature_row(_at(day), feature_schema_version=_LEGACY_VERSION, label="focus"),
        ])

    db.run(seed)
    first_code, first, _ = _run_cli(_argv(db, day, day, "--apply"), capsys)
    rows_after_first = db.run(lambda handles: handles.v4_rows())

    second_code, second, _ = _run_cli(_argv(db, day, day, "--apply"), capsys)
    rows_after_second = db.run(lambda handles: handles.v4_rows())

    assert first_code == second_code == 0
    assert first["totals"]["blocks"] == second["totals"]["blocks"] == 1
    assert first["totals"]["windows_rebuilt"] == second["totals"]["windows_rebuilt"] == 2
    assert first["totals"]["windows_written"] == second["totals"]["windows_written"] == 2
    assert first["totals"]["blocks_failed"] == second["totals"]["blocks_failed"] == 0
    # The label is inherited on the first pass and then simply kept, so exactly
    # one row is labelled on every run (never two, never zero).
    assert first["totals"]["labels_inherited"] == 1
    assert second["totals"]["labels_preserved"] == 1
    for totals in (first["totals"], second["totals"]):
        assert totals["labels_inherited"] + totals["labels_preserved"] == 1
    assert len(rows_after_first) == len(rows_after_second) == 2
    for before, after in zip(rows_after_first, rows_after_second, strict=True):
        assert before["window_start_utc"] == after["window_start_utc"]
        assert before["features_json"] == after["features_json"]
        assert before["label"] == after["label"], "the label never drifts"
    assert [row["label"] for row in rows_after_second] == ["focus", None]


def test_failed_block_is_retryable_with_the_same_parameters(
    db: _Db, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    first_day, second_day = _day(8), _day(7)

    async def seed(handles: _Handles) -> None:
        await _seed_event(handles.activity, first_day)
        await _seed_event(handles.activity, second_day, hour=14)

    db.run(seed)
    real_write_block = cli._write_block
    attempts = {"count": 0}

    async def flaky_write_block(
        service: Any,
        block_start: datetime,
        block_end: datetime,
        user_id: int,
        *,
        apply: bool,
    ) -> Any:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("simulated block failure")
        return await real_write_block(
            service, block_start, block_end, user_id, apply=apply
        )

    monkeypatch.setattr(cli, "_write_block", flaky_write_block)
    code, summary, stderr = _run_cli(_argv(db, first_day, second_day, "--apply"), capsys)

    assert code != 0, "an applied block failed"
    assert summary["exit_code"] == code
    assert summary["totals"]["blocks_failed"] == 1
    assert summary["totals"]["windows_written"] == 2, "the healthy block still ran"
    failed = summary["failed_blocks"][0]
    assert failed["block_start"] == _midnight(first_day).isoformat()
    assert "simulated block failure" in failed["reason"]
    assert "FAILED: RuntimeError: simulated block failure" in stderr
    # The failed block wrote nothing at all: a retry starts from a clean slate.
    assert [
        row["window_start_utc"] for row in db.run(lambda handles: handles.v4_rows())
    ] == [_at(second_day, 14).isoformat(), _at(second_day, 14, 5).isoformat()]

    monkeypatch.setattr(cli, "_write_block", real_write_block)
    retry_code, retry, _ = _run_cli(_argv(db, first_day, second_day, "--apply"), capsys)

    assert retry_code == 0
    assert retry["totals"]["blocks_failed"] == 0
    assert retry["totals"]["windows_written"] == 4
    assert len(db.run(lambda handles: handles.v4_rows())) == 4, "no duplicate rows"


def test_preview_with_a_failing_block_still_exits_zero(
    db: _Db, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preview never writes, so it can never leave a half-applied range behind."""
    day = _day(9)

    def seed(handles: _Handles) -> Awaitable[None]:
        return _seed_event(handles.activity, day)

    db.run(seed)

    async def failing_write_block(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated block failure")

    monkeypatch.setattr(cli, "_write_block", failing_write_block)
    code, summary, _ = _run_cli(_argv(db, day, day), capsys)

    assert code == 0
    assert summary["mode"] == "preview"
    assert summary["totals"]["blocks_failed"] == 1
    assert summary["exit_code"] == 0


# ── Clamping ──────────────────────────────────────────────────────────────


async def test_start_older_than_the_rebuild_boundary_is_clamped_and_reported(
    session_factory: async_sessionmaker[AsyncSession], create_tables: None
) -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    summary = await cli.run_rebuild_features(
        session_factory=session_factory,
        user_id=1,
        requested_start=date(2025, 1, 1),
        requested_end=date(2026, 2, 2),
        apply=False,
        now_utc=now,
    )

    assert summary.clamped is True
    assert str(_FEATURE_REBUILD_MAX_DAYS) == "180"
    assert "180-day v4 rebuild boundary" in summary.clamp_reason
    assert "2025-01-01" in summary.clamp_reason
    # Clamped to the UTC day of the boundary (2026-02-01T12:00Z) so that every
    # block stays a whole UTC calendar day; the exact cut is reported too.
    assert summary.cutoff_utc == datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
    assert summary.effective_start == datetime(2026, 2, 1, tzinfo=UTC)
    assert summary.effective_end == datetime(2026, 2, 3, tzinfo=UTC)
    assert [block.block_start for block in summary.blocks] == [
        datetime(2026, 2, 1, tzinfo=UTC),
        datetime(2026, 2, 2, tzinfo=UTC),
    ]
    assert summary.empty_range is False
    assert all(block.missing_raw_data for block in summary.blocks)

    payload = summary.to_dict()
    assert payload["clamped"] is True
    assert payload["requested_start"] == "2025-01-01"
    assert payload["retention_days"] == _FEATURE_REBUILD_MAX_DAYS
    assert payload["cutoff_utc"] == "2026-02-01T12:00:00+00:00"


async def test_range_entirely_outside_retention_reports_no_applicable_blocks(
    session_factory: async_sessionmaker[AsyncSession], create_tables: None
) -> None:
    summary = await cli.run_rebuild_features(
        session_factory=session_factory,
        user_id=1,
        requested_start=date(2020, 1, 1),
        requested_end=date(2020, 1, 3),
        apply=True,
        now_utc=datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    )

    assert summary.clamped is True
    assert summary.blocks == ()
    assert summary.empty_range is True
    assert summary.exit_code == 0
    assert summary.to_dict()["empty_range"] is True


# ── Label inheritance and v4 priority ─────────────────────────────────────


def test_v3_label_is_inherited_by_the_rebuilt_v4_row(
    db: _Db, capsys: pytest.CaptureFixture[str]
) -> None:
    day = _day(10)

    async def seed(handles: _Handles) -> None:
        await _seed_event(handles.activity, day)
        await handles.telemetry.upsert_feature_windows([
            _feature_row(_at(day), feature_schema_version=_LEGACY_VERSION, label="focus"),
        ])

    db.run(seed)
    code, summary, _ = _run_cli(_argv(db, day, day, "--apply"), capsys)

    assert code == 0
    assert summary["totals"]["labels_inherited"] == 1
    assert summary["totals"]["labels_preserved"] == 0
    rebuilt = {row["window_start_utc"]: row for row in db.run(lambda h: h.v4_rows())}
    assert rebuilt[_at(day).isoformat()]["label"] == "focus"
    assert rebuilt[_at(day, 8, 5).isoformat()]["label"] is None
    assert db.run(lambda h: h.legacy_rows())[0]["label"] == "focus", "v3 rows are kept"


def test_existing_v4_label_wins_over_the_inherited_v3_label(
    db: _Db, capsys: pytest.CaptureFixture[str]
) -> None:
    day = _day(11)

    async def seed(handles: _Handles) -> None:
        await _seed_event(handles.activity, day)
        await handles.telemetry.upsert_feature_windows([
            _feature_row(_at(day), feature_schema_version=_LEGACY_VERSION, label="focus"),
            _feature_row(_at(day), label="distracted"),
        ])

    db.run(seed)
    code, summary, _ = _run_cli(_argv(db, day, day, "--apply"), capsys)

    assert code == 0
    assert summary["totals"]["labels_preserved"] == 1
    assert summary["totals"]["labels_inherited"] == 0
    rows = db.run(lambda handles: handles.v4_rows())
    labelled = next(row for row in rows if row["window_start_utc"] == _at(day).isoformat())
    assert labelled["label"] == "distracted", "a rebuild never overwrites a v4 label"


def test_backfill_keeps_the_v3_rows_it_inherited_from(
    db: _Db, capsys: pytest.CaptureFixture[str]
) -> None:
    day = _day(12)

    async def seed(handles: _Handles) -> None:
        await _seed_event(handles.activity, day)
        await handles.telemetry.upsert_feature_windows([
            _feature_row(
                _at(day),
                feature_schema_version=_LEGACY_VERSION,
                label="focus",
                features_json=_features(idle_ratio=0.42),
            ),
        ])

    db.run(seed)
    code, _, _ = _run_cli(_argv(db, day, day, "--apply"), capsys)

    assert code == 0
    legacy = db.run(lambda handles: handles.legacy_rows())
    assert len(legacy) == 1
    assert legacy[0]["label"] == "focus"
    # The v3 blob is untouched evidence of what the model was trained on; the
    # v4 features were regenerated from raw events, not copied from it.
    assert json.loads(legacy[0]["features_json"])["idle_ratio"] == pytest.approx(0.42)
    v4_idle = json.loads(db.run(lambda handles: handles.v4_rows())[0]["features_json"])
    assert v4_idle["idle_ratio"] == pytest.approx(0.0)


# ── Explicit column consistency ───────────────────────────────────────────


def test_rebuilt_features_json_agrees_with_every_explicit_column(
    db: _Db, capsys: pytest.CaptureFixture[str]
) -> None:
    day = _day(13)

    def seed(handles: _Handles) -> Awaitable[None]:
        return _seed_event(handles.activity, day)

    db.run(seed)
    code, _, _ = _run_cli(_argv(db, day, day, "--apply"), capsys)

    assert code == 0
    rows = db.run(lambda handles: handles.v4_rows())
    assert rows, "the block was rebuilt"
    for row in rows:
        features = json.loads(row["features_json"])
        assert set(V2_FEATURE_NAMES) <= set(features)
        for name, column in zip(V2_FEATURE_NAMES, _FEATURE_COLUMNS, strict=True):
            assert row[column] == pytest.approx(features[name]), f"{name} != {column}"
    sampled = rows[0]
    assert sampled["f04"] == pytest.approx(
        json.loads(sampled["features_json"])["idle_ratio"]
    )


async def test_same_key_upsert_refreshes_json_and_explicit_columns_together(
    session_factory: async_sessionmaker[AsyncSession], create_tables: None
) -> None:
    """A same-key re-roll must not refresh only one representation of a row."""
    repository = TelemetryRepository(session_factory)
    start = datetime(2026, 7, 24, 8, 0, tzinfo=UTC)
    await repository.upsert_feature_windows([
        _feature_row(start, idle_ratio=0.1, task_type_entropy=0.2, app_switch_count=3.0),
    ])
    await repository.upsert_feature_windows([
        _feature_row(
            start,
            idle_ratio=0.9,
            task_type_entropy=0.8,
            app_switch_count=17.0,
            label="focus",
        ),
    ])

    rows = await repository.list_feature_windows(1)
    assert len(rows) == 1, "the same key stays one row"
    stored = rows[0]
    features = json.loads(stored["features_json"])
    assert features["idle_ratio"] == pytest.approx(0.9)
    assert features["app_switch_count"] == pytest.approx(17.0)
    for name, column in zip(V2_FEATURE_NAMES, _FEATURE_COLUMNS, strict=True):
        assert stored[column] == pytest.approx(features[name]), f"{name} != {column}"
    # Every fNN column really moved, not just the two sampled values.
    assert stored["f04"] == pytest.approx(0.9)
    assert stored["f25"] == pytest.approx(0.8)
    assert stored["f01"] == pytest.approx(17.0)


# ── CLI surface ───────────────────────────────────────────────────────────


def test_cli_is_runnable_as_a_module_and_help_does_not_raise() -> None:
    for extra in (["--help"], ["rebuild-features", "--help"]):
        result = subprocess.run(
            [sys.executable, "-m", "mindflow.telemetry", *extra],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
        assert "--start" in result.stdout or extra == ["--help"]


def test_unknown_dates_produce_a_clear_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for bad in ("not-a-date", "2026-13-45", "2026/07/01", "20260701"):
        with pytest.raises(SystemExit) as excinfo:
            cli.main([
                "rebuild-features",
                "--start",
                bad,
                "--end",
                "2026-07-02",
                "--user-id",
                "1",
            ])
        assert excinfo.value.code != 0
        stderr = capsys.readouterr().err
        assert "YYYY-MM-DD" in stderr
        assert bad in stderr


def test_start_after_end_is_rejected(
    db: _Db, capsys: pytest.CaptureFixture[str]
) -> None:
    code, summary, stderr = _run_cli(_argv(db, _day(2), _day(3)), capsys)

    assert code != 0
    assert summary == {}
    assert "--start" in stderr and "must not be after" in stderr


def test_missing_or_unmigrated_database_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = _Db(tmp_path / "not_there.db")
    code, summary, stderr = _run_cli(_argv(missing, _day(2), _day(2)), capsys)
    assert code != 0
    assert summary == {}
    assert "database not found" in stderr

    empty = _Db(tmp_path / "empty.db")
    empty.path.write_bytes(b"")
    code, summary, stderr = _run_cli(_argv(empty, _day(2), _day(2)), capsys)
    assert code != 0
    assert summary == {}
    assert "alembic upgrade head" in stderr
    assert "behavior_feature_windows" in stderr


def test_json_out_writes_the_summary_to_a_file_instead_of_stdout(
    db: _Db, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    day = _day(14)

    def seed(handles: _Handles) -> Awaitable[None]:
        return _seed_event(handles.activity, day)

    db.run(seed)
    out_path = tmp_path / "reports" / "summary.json"
    code = cli.main(_argv(db, day, day, "--json-out", str(out_path)))
    captured = capsys.readouterr()

    assert code == 0
    assert captured.out.strip() == ""
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["command"] == "rebuild-features"
    assert written["mode"] == "preview"
    assert written["totals"]["windows_rebuilt"] == 2
    assert str(out_path) in captured.err


def test_cli_registers_no_http_route() -> None:
    """The backfill stays a local CLI: no FastAPI router, no new endpoint."""
    source = Path(str(cli.__file__)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not [name for name in imported if name.startswith("mindflow.api")]
    assert "include_router" not in source
    assert "APIRouter" not in source

    # Importing the CLI in a clean interpreter must not pull the API package in.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, mindflow.telemetry.__main__; "
            "print(any(m.startswith('mindflow.api') for m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=180,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "False"


def test_raw_activity_events_are_the_only_feature_source(
    db: _Db, capsys: pytest.CaptureFixture[str]
) -> None:
    """A v3 row without retained raw events cannot become a v4 row."""
    day = _day(15)

    async def seed(handles: _Handles) -> None:
        await handles.telemetry.upsert_feature_windows([
            _feature_row(
                _at(day),
                feature_schema_version=_LEGACY_VERSION,
                label="focus",
                features_json=_features(idle_ratio=0.77),
            ),
        ])

    db.run(seed)
    code, summary, _ = _run_cli(_argv(db, day, day, "--apply"), capsys)

    assert code == 0
    assert summary["totals"]["windows_written"] == 0
    assert summary["blocks"][0]["missing_raw_data"] is True
    assert db.run(lambda handles: handles.v4_rows()) == []
    assert len(db.run(lambda handles: handles.legacy_rows())) == 1


def test_backfill_never_touches_another_user_or_version(
    db: _Db, capsys: pytest.CaptureFixture[str]
) -> None:
    day = _day(16)

    async def seed(handles: _Handles) -> None:
        await _seed_event(handles.activity, day)
        await _seed_event(handles.activity, day, hour=14, user_id=2)
        async with handles.session_factory() as session:
            await session.execute(
                sa.insert(behavior_feature_windows).values(
                    id="other-user",
                    user_id=2,
                    window_start_utc=_at(day).isoformat(),
                    window_end_utc=_at(day, 8, 5).isoformat(),
                    feature_schema_version=FEATURE_SCHEMA_VERSION,
                    features_json=_features(),
                    label="distracted",
                    created_at=datetime.now(UTC).isoformat(),
                )
            )
            await session.commit()

    db.run(seed)
    code, _, _ = _run_cli(_argv(db, day, day, "--apply"), capsys)

    assert code == 0
    other = db.run(lambda handles: handles.v4_rows(user_id=2))
    assert [row["id"] for row in other] == ["other-user"]
    assert other[0]["label"] == "distracted"
    assert len(db.run(lambda handles: handles.v4_rows(user_id=1))) == 2
