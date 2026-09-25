"""Operator CLI for local telemetry maintenance — explicit v4 feature backfill.

Usage (run from ``mindflow-app/backend-next``)::

    # Preview: builds and counts the windows, writes nothing at all
    python -m mindflow.telemetry rebuild-features \
        --start 2026-07-01 --end 2026-07-07 --user-id 1

    # Apply: same range, real writes (one transaction per day block)
    python -m mindflow.telemetry rebuild-features \
        --start 2026-07-01 --end 2026-07-07 --user-id 1 --apply

Contract of ``rebuild-features``:

* **v4 only, from raw events.** Windows are rebuilt through the same code path
  as the incremental rollup (retained ``activity_events`` plus interaction
  buckets and browser segments). Features are never synthesised from the
  previous version's ``features_json``.
* **Labels are inherited, never stolen.** A v3 row's non-null label is written
  onto the rebuilt v4 row for the same ``window_start_utc``; an existing
  non-null **v4** label always wins and is never overwritten.
* **v3 rows are kept.** A backfill never deletes the previous schema version:
  it stays the evidence of what the model was trained on.
* **Bounded by the 180-day rebuild cut.** A requested start older than the
  feature-window retention boundary is clamped and the clamp is reported
  explicitly (never a silent truncation).
* **Day-blocked.** The range is processed in ``_FEATURE_REBUILD_CHUNK_DAYS``
  blocks, and every block is reported that starts/progresses/fails on its own,
  so a long run is observable and a failure is attributable to one block.
  Each applied block is written in a single transaction, so re-running the same
  command with the same parameters is an idempotent retry.
* **No HTTP surface.** Nothing here registers a route; this is a local CLI.
* **Backups are an operator step.** The CLI never backs the database up itself;
  it prints the warning instead.

Output: human-readable progress goes to **stderr** (one flushed line per
block), the machine-readable JSON summary goes to **stdout** — or to the file
named by ``--json-out``. Exit codes: ``0`` when every applicable block succeeded
(or the run was a preview), ``1`` when an applied block failed, ``2`` for a
usage/pre-flight error (bad date, missing database, missing tables).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Final, TextIO

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.config import get_settings
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION, V2_FEATURE_NAMES
from mindflow.infrastructure.database import create_engine, create_session_factory
from mindflow.infrastructure.repositories.activity import SQLAlchemyActivityRepository
from mindflow.infrastructure.repositories.preferences import PreferencesRepository
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.services.collector_interval_lifecycle import safe_error_text
from mindflow.services.telemetry_service import (
    _FEATURE_REBUILD_CHUNK_DAYS,
    _FEATURE_REBUILD_MAX_DAYS,
    FeatureWindowRebuildResult,
    TelemetryService,
    feature_rebuild_cutoff,
)

_COMMAND: Final = "rebuild-features"
_DATE_FORMAT: Final = "YYYY-MM-DD"
_DATE_PATTERN: Final = re.compile(r"\d{4}-\d{2}-\d{2}")

_EXIT_OK: Final = 0
_EXIT_BLOCK_FAILED: Final = 1
_EXIT_USAGE: Final = 2
_EXIT_INTERRUPTED: Final = 130

# Tables a rebuild must be able to read; a database without them has not been
# migrated yet and a backfill would fail with a confusing per-block error.
_REQUIRED_TABLES: Final[tuple[str, ...]] = (
    "activity_events",
    "behavior_feature_windows",
)

_EXIT_CODE_MEANING: Final = (
    "0 = every applicable block succeeded (or the run was a preview), "
    "1 = at least one applied block failed, "
    "2 = usage or pre-flight error, 130 = interrupted"
)

ProgressCallback = Callable[[int, int, "BlockReport"], None]
"""Called as ``progress(index, total, report)`` after every block."""


@dataclass(frozen=True, slots=True)
class BlockReport:
    """Outcome of one day block — the smallest retryable unit.

    ``windows_rebuilt`` counts the windows this block built: with ``--apply``
    they are the rows written (``windows_written``), without it they are the
    rows that *would* be written. A block with no retained raw evidence reports
    ``missing_raw_data`` instead of vanishing silently, and a block that raised
    reports ``failed`` with ``reason`` while the other blocks still run.
    """

    block_start: datetime
    block_end: datetime
    windows_rebuilt: int
    windows_written: int
    missing_raw_data: bool
    succeeded: bool
    failed: bool
    status: str
    reason: str
    labels_inherited: int = 0
    labels_preserved: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_start": _iso(self.block_start),
            "block_end": _iso(self.block_end),
            "windows_rebuilt": self.windows_rebuilt,
            "windows_written": self.windows_written,
            "missing_raw_data": self.missing_raw_data,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "status": self.status,
            "reason": self.reason,
            "labels_inherited": self.labels_inherited,
            "labels_preserved": self.labels_preserved,
        }


@dataclass(frozen=True, slots=True)
class RebuildRunSummary:
    """Machine-readable result of one ``rebuild-features`` invocation."""

    applied: bool
    user_id: int
    database_path: str
    requested_start: date
    requested_end: date
    effective_start: datetime | None
    effective_end: datetime | None
    clamped: bool
    clamp_reason: str
    cutoff_utc: datetime
    blocks: tuple[BlockReport, ...]

    @property
    def failed_blocks(self) -> tuple[BlockReport, ...]:
        return tuple(block for block in self.blocks if block.failed)

    @property
    def empty_range(self) -> bool:
        """True when the whole request fell outside the retained range."""
        if self.effective_start is None or self.effective_end is None:
            return True
        return self.effective_start >= self.effective_end

    @property
    def exit_code(self) -> int:
        """0 unless an applied block failed (a preview must never write)."""
        if self.applied and self.failed_blocks:
            return _EXIT_BLOCK_FAILED
        return _EXIT_OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": _COMMAND,
            "mode": "apply" if self.applied else "preview",
            "applied": self.applied,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_column_count": len(V2_FEATURE_NAMES),
            "user_id": self.user_id,
            "database_path": self.database_path,
            "requested_start": self.requested_start.isoformat(),
            "requested_end": self.requested_end.isoformat(),
            "effective_start": _iso_or_none(self.effective_start),
            "effective_end": _iso_or_none(self.effective_end),
            "empty_range": self.empty_range,
            "retention_days": _FEATURE_REBUILD_MAX_DAYS,
            "chunk_days": _FEATURE_REBUILD_CHUNK_DAYS,
            "cutoff_utc": _iso(self.cutoff_utc),
            "clamped": self.clamped,
            "clamp_reason": self.clamp_reason,
            "backup_required": True,
            "blocks": [block.to_dict() for block in self.blocks],
            "totals": {
                "blocks": len(self.blocks),
                "blocks_succeeded": sum(1 for b in self.blocks if b.succeeded),
                "blocks_failed": len(self.failed_blocks),
                "blocks_missing_raw_data": sum(
                    1 for b in self.blocks if b.missing_raw_data
                ),
                "windows_rebuilt": sum(b.windows_rebuilt for b in self.blocks),
                "windows_written": sum(b.windows_written for b in self.blocks),
                "labels_inherited": sum(b.labels_inherited for b in self.blocks),
                "labels_preserved": sum(b.labels_preserved for b in self.blocks),
            },
            "failed_blocks": [
                {
                    "block_start": _iso(block.block_start),
                    "block_end": _iso(block.block_end),
                    "reason": block.reason,
                }
                for block in self.failed_blocks
            ],
            "exit_code": self.exit_code,
        }


class _PreflightError(RuntimeError):
    """The database cannot host a rebuild (missing file, unmigrated schema)."""


# ── Async core ────────────────────────────────────────────────────────────


async def run_rebuild_features(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    user_id: int,
    requested_start: date,
    requested_end: date,
    apply: bool,
    now_utc: datetime | None = None,
    database_path: Path | None = None,
    data_dir: Path | None = None,
    pulsetime_s: int | None = None,
    progress: ProgressCallback | None = None,
) -> RebuildRunSummary:
    """Rebuild v4 windows day block by day block over the clamped range.

    Both requested dates are UTC calendar dates and ``--end`` is inclusive as a
    date: the internal ranges stay half-open (``[00:00, next 00:00)``). The
    request is clamped to the ``_FEATURE_REBUILD_MAX_DAYS`` boundary at UTC-day
    granularity so every block is a whole UTC day; the exact boundary instant is
    reported as ``cutoff_utc`` alongside ``clamped`` and ``clamp_reason``.

    The service is wired without a baseline repository on purpose: a backfill
    repairs feature windows only and never silently rewrites the Welford
    baseline that prediction reads.
    """
    now = now_utc if now_utc is not None else datetime.now(UTC)
    cutoff = feature_rebuild_cutoff(now)
    cutoff_date = cutoff.date()
    clamped = requested_start < cutoff_date
    effective_start_date = max(requested_start, cutoff_date)
    effective_start = _utc_midnight(effective_start_date)
    effective_end = _utc_midnight(requested_end) + timedelta(days=1)
    if clamped:
        clamp_reason = (
            f"--start {requested_start.isoformat()} is older than the "
            f"{_FEATURE_REBUILD_MAX_DAYS}-day v4 rebuild boundary "
            f"(retention cutoff {_iso(cutoff)}); clamped to "
            f"{effective_start_date.isoformat()} so every block stays a whole "
            "UTC day"
        )
    else:
        clamp_reason = ""

    activity_repository = SQLAlchemyActivityRepository(
        session_factory, pulsetime_s=pulsetime_s,
    )
    service = TelemetryService(
        TelemetryRepository(session_factory),
        PreferencesRepository(session_factory),
        data_dir=data_dir if data_dir is not None else Path.cwd(),
        activity_repository=activity_repository,
    )

    chunk = timedelta(days=_FEATURE_REBUILD_CHUNK_DAYS)
    block_bounds: list[tuple[datetime, datetime]] = []
    block_start = effective_start
    while block_start < effective_end:
        block_end = min(block_start + chunk, effective_end)
        block_bounds.append((block_start, block_end))
        block_start = block_end

    reports: list[BlockReport] = []
    for index, (start, end) in enumerate(block_bounds, start=1):
        report = await _rebuild_one_block(
            service, start, end, user_id, apply=apply,
        )
        reports.append(report)
        if progress is not None:
            progress(index, len(block_bounds), report)

    return RebuildRunSummary(
        applied=apply,
        user_id=user_id,
        database_path=str(database_path) if database_path is not None else "",
        requested_start=requested_start,
        requested_end=requested_end,
        effective_start=effective_start,
        effective_end=effective_end,
        clamped=clamped,
        clamp_reason=clamp_reason,
        cutoff_utc=cutoff,
        blocks=tuple(reports),
    )


async def _write_block(
    service: TelemetryService,
    block_start: datetime,
    block_end: datetime,
    user_id: int,
    *,
    apply: bool,
) -> FeatureWindowRebuildResult:
    """Per-block writer seam: exactly one service call per day block.

    Kept as a module-level function so a failure can be injected for one block
    (tests, post-mortems) and so the CLI's retry granularity is explicit: one
    call means one block, nothing more.
    """
    return await service.rebuild_feature_windows(
        block_start, block_end, user_id, apply=apply,
    )


async def _rebuild_one_block(
    service: TelemetryService,
    block_start: datetime,
    block_end: datetime,
    user_id: int,
    *,
    apply: bool,
) -> BlockReport:
    """Run one block, converting a failure into a report instead of a crash.

    A failing block never aborts the run: every remaining block still gets its
    own report, and the failure stays attributable to exactly one block. The
    applied block is written in a single transaction, so nothing partial is
    left behind and the same command can simply be re-run.
    """
    try:
        result = await _write_block(
            service, block_start, block_end, user_id, apply=apply,
        )
    except Exception as exc:  # noqa: BLE001 — one block must not kill the run
        return BlockReport(
            block_start=block_start,
            block_end=block_end,
            windows_rebuilt=0,
            windows_written=0,
            missing_raw_data=False,
            succeeded=False,
            failed=True,
            status="failed",
            reason=safe_error_text(exc),
        )
    return BlockReport(
        block_start=block_start,
        block_end=block_end,
        windows_rebuilt=result.windows_rolled,
        windows_written=result.windows_written,
        missing_raw_data=result.missing_raw_data,
        succeeded=True,
        failed=False,
        status=result.reason,
        reason=(
            "no retained activity events, interaction buckets or browser "
            "segments in this block"
            if result.missing_raw_data
            else ""
        ),
        labels_inherited=result.labels_inherited,
        labels_preserved=result.labels_preserved,
    )


async def _missing_tables(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[str, ...]:
    """Required tables absent from the database (empty when it is ready)."""
    async with session_factory() as session:
        rows = await session.execute(
            sa.text("SELECT name FROM sqlite_master WHERE type = 'table'")
        )
    present = {str(row[0]) for row in rows.fetchall()}
    return tuple(name for name in _REQUIRED_TABLES if name not in present)


# ── CLI plumbing ──────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mindflow.telemetry",
        description=(
            "MindFlow local telemetry maintenance commands (operator CLIs; "
            "nothing here is exposed over HTTP)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    rebuild = subparsers.add_parser(
        _COMMAND,
        help="Rebuild v4 feature windows from retained raw activity events",
        description=(
            "Rebuild v4 feature windows from retained raw activity events. "
            "Preview by default (nothing is written); --apply performs the "
            "writes, one transaction per day block. v3 windows are kept and "
            "only their labels are inherited; an existing v4 label is never "
            "overwritten. The requested range is clamped to the "
            f"{_FEATURE_REBUILD_MAX_DAYS}-day rebuild boundary and any clamp is "
            "reported."
        ),
        epilog=(
            "Output: one human-readable progress line per block on stderr; the "
            "JSON summary on stdout (or in --json-out). Exit codes: "
            f"{_EXIT_CODE_MEANING}.\n\n"
            "WARNING: back the database up yourself before --apply - this CLI "
            "never creates a backup."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    rebuild.add_argument(
        "--start",
        required=True,
        type=_parse_utc_date,
        metavar=_DATE_FORMAT,
        help=f"Inclusive UTC start date ({_DATE_FORMAT})",
    )
    rebuild.add_argument(
        "--end",
        required=True,
        type=_parse_utc_date,
        metavar=_DATE_FORMAT,
        help=(
            f"Inclusive UTC end date ({_DATE_FORMAT}); internal ranges stay "
            "half-open"
        ),
    )
    rebuild.add_argument(
        "--user-id",
        type=int,
        default=1,
        dest="user_id",
        help="User id whose windows are rebuilt (default: 1)",
    )
    rebuild.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Write the rebuilt windows. Without it the run is a preview: "
            "windows are built and reported but nothing is written."
        ),
    )
    rebuild.add_argument(
        "--json-out",
        type=str,
        default="",
        dest="json_out",
        help="Write the JSON summary to this path instead of stdout",
    )
    rebuild.add_argument(
        "--database-path",
        type=str,
        default="",
        dest="database_path",
        help=(
            "SQLite database file to rebuild (default: the configured "
            "application database)"
        ),
    )
    return parser


def _parse_utc_date(value: str) -> date:
    """Parse an argparse date argument, naming the expected format on error."""
    error = argparse.ArgumentTypeError(
        f"expected a UTC calendar date in {_DATE_FORMAT} format, got {value!r}"
    )
    if not _DATE_PATTERN.fullmatch(value):
        raise error
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise error from None


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the requested command; returns the exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != _COMMAND:  # pragma: no cover - argparse restricts this
        parser.error(f"unknown command: {args.command}")
    return _run_rebuild_command(args, sys.stdout, sys.stderr)


def _run_rebuild_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    database_path = _resolve_database_path(args.database_path)
    if args.start > args.end:
        stderr.write(
            f"error: --start {args.start.isoformat()} must not be after "
            f"--end {args.end.isoformat()}\n"
        )
        return _EXIT_USAGE
    _write_backup_notice(stderr, apply=bool(args.apply), database_path=database_path)
    if not database_path.exists():
        stderr.write(
            f"error: database not found: {database_path}\n"
            "hint: start MindFlow once, or pass --database-path explicitly\n"
        )
        return _EXIT_USAGE

    try:
        summary = asyncio.run(_execute(args, database_path, stderr))
    except _PreflightError as exc:
        stderr.write(f"error: {exc}\n")
        return _EXIT_USAGE
    except KeyboardInterrupt:  # pragma: no cover - interactive abort
        stderr.write(
            "interrupted: blocks are written transactionally, so the last "
            "block either landed completely or not at all\n"
        )
        return _EXIT_INTERRUPTED

    try:
        _write_json_summary(summary, args.json_out, stdout, stderr)
    except OSError as exc:
        stderr.write(f"error: could not write --json-out {args.json_out}: {exc}\n")
        return _EXIT_USAGE
    _write_human_footer(summary, stderr)
    return summary.exit_code


async def _execute(
    args: argparse.Namespace,
    database_path: Path,
    stderr: TextIO,
) -> RebuildRunSummary:
    engine = create_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")
    try:
        session_factory = create_session_factory(engine)
        missing = await _missing_tables(session_factory)
        if missing:
            raise _PreflightError(
                f"database {database_path} is missing table(s): "
                f"{', '.join(missing)}; run `uv run alembic upgrade head` first"
            )
        return await run_rebuild_features(
            session_factory=session_factory,
            user_id=args.user_id,
            requested_start=args.start,
            requested_end=args.end,
            apply=bool(args.apply),
            database_path=database_path,
            data_dir=database_path.parent,
            progress=lambda index, total, report: _write_progress(
                index, total, report, applied=bool(args.apply), stream=stderr,
            ),
        )
    finally:
        await engine.dispose()


def _resolve_database_path(explicit: str | None) -> Path:
    """Resolve the SQLite file: explicit ``--database-path`` first, then settings."""
    if explicit:
        return Path(explicit).expanduser()
    settings = get_settings()
    prefix = "sqlite+aiosqlite:///"
    if settings.db_url.startswith(prefix):
        return Path(settings.db_url[len(prefix):]).expanduser()
    return settings.data_dir / "mindflow.db"


# ── Reporting ─────────────────────────────────────────────────────────────


def _write_backup_notice(stream: TextIO, *, apply: bool, database_path: Path) -> None:
    """State the backup requirement up front; the CLI never backs up for you."""
    if apply:
        stream.write(
            "WARNING: --apply writes v4 feature windows into "
            f"{database_path}\n"
            "         Back the database up first (operator step; this CLI never "
            "creates a backup):\n"
            f'           sqlite3 "{database_path}" '
            f'".backup "{database_path}.pre-backfill.bak"\n'
        )
    else:
        stream.write(
            "Preview only: no rows will be written. A backup is required before "
            "re-running with --apply.\n"
        )
    stream.flush()


def _write_progress(
    index: int,
    total: int,
    report: BlockReport,
    *,
    applied: bool,
    stream: TextIO,
) -> None:
    """One flushed human-readable line per block, so a long run is observable."""
    span = f"{_iso(report.block_start)} -> {_iso(report.block_end)}"
    if report.failed:
        outcome = f"FAILED: {report.reason}"
    elif report.missing_raw_data:
        outcome = f"missing_raw_data: {report.reason}"
    elif applied:
        outcome = (
            f"rebuilt={report.windows_rebuilt} written={report.windows_written} "
            f"labels_inherited={report.labels_inherited} "
            f"labels_preserved={report.labels_preserved}"
        )
    else:
        outcome = (
            f"would_rebuild={report.windows_rebuilt} written=0 "
            f"labels_inherited={report.labels_inherited} "
            f"labels_preserved={report.labels_preserved}"
        )
    stream.write(f"[{index}/{total}] {span} | {outcome}\n")
    stream.flush()


def _write_json_summary(
    summary: RebuildRunSummary,
    json_out: str | None,
    stdout: TextIO,
    stderr: TextIO,
) -> None:
    """Emit the machine-readable summary to stdout, or to ``--json-out``.

    ``json.dumps`` keeps the default ASCII escaping on purpose: stdout must
    survive a redirect on a Windows console whose code page cannot encode every
    character of a database path.
    """
    text = json.dumps(summary.to_dict(), indent=2)
    if json_out:
        out_path = Path(json_out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        stderr.write(f"JSON summary written to {out_path}\n")
    else:
        stdout.write(text + "\n")
    stdout.flush()


def _write_human_footer(summary: RebuildRunSummary, stream: TextIO) -> None:
    totals = summary.to_dict()["totals"]
    mode = "apply" if summary.applied else "preview"
    written = (
        f"{totals['windows_written']} window(s) written"
        if summary.applied
        else "nothing written"
    )
    stream.write(
        f"done ({mode}): {totals['blocks']} block(s), "
        f"{totals['windows_rebuilt']} window(s) rebuilt, {written}, "
        f"{totals['blocks_failed']} failed, "
        f"{totals['blocks_missing_raw_data']} without raw data\n"
    )
    if summary.clamped:
        stream.write(f"note: {summary.clamp_reason}\n")
    if summary.empty_range:
        stream.write(
            "note: no applicable blocks — the whole requested range is outside "
            f"the {_FEATURE_REBUILD_MAX_DAYS}-day rebuild boundary\n"
        )
    stream.flush()


# ── Small helpers ─────────────────────────────────────────────────────────


def _utc_midnight(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _iso_or_none(value: datetime | None) -> str | None:
    return None if value is None else _iso(value)


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
