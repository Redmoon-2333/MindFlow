"""Reproducible event-level simulation for a synthetic, non-personalised prior.

Labels express the simulator's planned goal versus goal deviation, never an
app blacklist, clock time, or inactivity threshold. Ambiguous intentions share
observations across classes; synthetic validation is not real-world accuracy.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from mindflow.domain.events import ActivityEvent, WindowSnapshot
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION, V2_FEATURE_NAMES
from mindflow.domain.features import count_confirmed_switches
from mindflow.domain.task_context import TASK_TYPE_CODES
from mindflow.services.telemetry_features import (
    build_v2_feature_window,
    task_context_from_feature_window,
)
from mindflow.train.base_catalog import (
    APP_CATALOG,
    ROLES,
    SCENARIO_CATALOG,
    SCENARIOS,
    TIMEZONES,
    AppSpec,
    RoleSpec,
    ScenarioSpec,
)

_RECIPE_VERSION = "event-sessions-v2"
_EPOCH = datetime(2026, 10, 1, tzinfo=UTC)
_WINDOW_SECONDS = 300
# Per 30-second bucket: keys, clicks, scroll units, distance pixels, active seconds.
_INPUT_PROFILES: dict[str, tuple[float, float, float, float, float]] = {
    "typing": (24, 4, 5, 350, 18),
    "reading": (0.8, 1.2, 22, 80, 3),
    "watching": (0.3, 0.7, 1, 60, 1.5),
    "collaboration": (12, 8, 15, 650, 15),
    "drawing": (4, 16, 10, 2100, 23),
    "gaming": (30, 20, 5, 1800, 25),
    "rest": (0.3, 0.6, 1, 50, 1),
    "away": (0, 0, 0, 0, 0),
    "missing": (15, 6, 8, 500, 15),
}
_DWELL_SECONDS: dict[str, tuple[int, int]] = {
    "typing": (45, 180),
    "reading": (100, 300),
    "watching": (120, 300),
    "collaboration": (12, 38),
    "drawing": (70, 240),
    "gaming": (45, 240),
    "rest": (100, 300),
    "away": (300, 300),
    "missing": (45, 180),
}
_UNIT_FEATURES = frozenset(
    {
        "longest_segment_ratio",
        "idle_ratio",
        "input_active_ratio",
        "browser_ratio",
        "audible_browser_ratio",
        "active_seconds_ratio",
        "top_app_ratio",
        "top_domain_ratio",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "task_type_entropy",
        "task_type_dominant_ratio",
        "task_context_transition",
        "task_unknown_ratio",
    }
)
_REQUIRED_ROW_FIELDS = frozenset(
    {
        "id",
        "window_start_utc",
        "window_end_utc",
        "feature_schema_version",
        "features",
        "label",
        "label_source",
        "persona_id",
        "role",
        "scenario",
        "session_id",
        "split",
        "local_hour",
        "timezone",
        "apps",
    }
)


@dataclass
class SyntheticCorpus:
    windows: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _Persona:
    id: str
    user_id: int
    role: RoleSpec
    index: int
    timezone_name: str
    utc_offset_hours: float
    proficiency: float
    keyboard_preference: float
    start_hour_utc: int


def _session_lengths(count: int, rng: random.Random) -> list[int]:
    """Partition observed windows into 10–60 minute sessions, without a short tail."""
    result: list[int] = []
    while count:
        size = min(count, rng.randint(2, 12))
        if count - size == 1:
            size = size - 1 if size > 2 else size + 1
        result.append(size)
        count -= size
    return result


def _session_apps(role: RoleSpec, scenario: ScenarioSpec, rng: random.Random) -> list[AppSpec]:
    keys = list(scenario.apps or role.apps)
    if scenario.key == "browser_closed":
        keys = [key for key in keys if not APP_CATALOG[key].domain]
    if scenario.key in {"collaboration", "mixed_intent"}:
        keys += ["slack", "teams", "notion"]
    if scenario.key == "mixed_intent":
        keys += ["reddit", "youtube"]
    keys = list(dict.fromkeys(keys))
    # A subset is stable for the whole session; people do not install new tools
    # independently in every five-minute window.
    chosen = rng.sample(keys, min(len(keys), rng.randint(2, 4)))
    return [APP_CATALOG[key] for key in chosen]


def _raw_window(
    persona: _Persona,
    scenario: ScenarioSpec,
    apps: list[AppSpec],
    start: datetime,
    activity_level: float,
    rng: random.Random,
) -> tuple[list[ActivityEvent], list[dict[str, Any]], list[dict[str, Any]]]:
    """Simulate non-overlapping foreground dwell segments and 30s input buckets."""
    events: list[ActivityEvent] = []
    browser: list[dict[str, Any]] = []
    offset = 0
    app = rng.choice(apps)
    while offset < _WINDOW_SECONDS:
        if rng.random() < (0.95 if scenario.mode == "collaboration" else 0.3):
            app = rng.choice(apps)
        duration = min(_WINDOW_SECONDS - offset, rng.randint(*_DWELL_SECONDS[scenario.mode]))
        stamp = start + timedelta(seconds=offset)
        idle_probability = 1.0 if scenario.mode == "away" else 0.03
        if scenario.mode in {"reading", "watching", "rest"}:
            idle_probability = 0.15
        idle = rng.random() < idle_probability
        snapshot = WindowSnapshot(app.name, "Synthetic observation", app.process, idle, stamp)
        events.append(
            ActivityEvent(
                id=f"{persona.id}:{start.isoformat()}:{offset}",
                user_id=persona.user_id,
                timestamp_utc=stamp,
                duration_s=float(duration),
                event_type="window_snapshot",
                data=snapshot,
            )
        )
        if app.domain and not idle:
            browser.append(
                {
                    "timestamp": stamp.isoformat(),
                    "duration_s": duration,
                    "domain": app.domain,
                    "audible": scenario.mode in {"watching", "rest"} and rng.random() < 0.85,
                }
            )
        offset += duration

    buckets: list[dict[str, Any]] = []
    if scenario.mode == "missing":
        return events, buckets, browser
    profile = _INPUT_PROFILES[scenario.mode]
    pause_probability = {
        "reading": 0.65,
        "watching": 0.85,
        "rest": 0.9,
        "away": 1.0,
    }.get(scenario.mode, 0.08)
    for bucket_index in range(10):
        stamp = start + timedelta(seconds=bucket_index * 30)
        # Input and desktop idle derive from the same timeline, rather than
        # producing impossible keys while the simulated user is away.
        event = next(
            event
            for event in events
            if (
                event.timestamp_utc
                <= stamp
                < event.timestamp_utc + timedelta(seconds=event.duration_s)
            )
        )
        active_overlap = sum(
            max(
                0.0,
                (
                    min(
                        stamp + timedelta(seconds=30),
                        item.timestamp_utc + timedelta(seconds=item.duration_s),
                    )
                    - max(stamp, item.timestamp_utc)
                ).total_seconds(),
            )
            for item in events
            if not item.data.is_idle
        )
        engaged = not event.data.is_idle and rng.random() >= pause_probability
        scale = activity_level * rng.lognormvariate(-0.08, 0.35) if engaged else 0.0
        keys, clicks, scroll, distance, active = profile
        keyboard = persona.keyboard_preference
        buckets.append(
            {
                "window_start_utc": stamp.isoformat(),
                "duration_s": 30,
                "keypress_count": int(keys * scale * keyboard * persona.proficiency),
                "mouse_click_count": int(clicks * scale / keyboard),
                "scroll_delta": int(scroll * scale),
                "mouse_distance_px": round(distance * scale / keyboard, 3),
                "input_active_s": round(min(active_overlap, active * scale), 4),
                "interaction_burst_count": int(rng.randint(1, 4) * min(scale, 2.0)),
            }
        )
    return events, buckets, browser


def _simulate_day(
    persona: _Persona,
    day_index: int,
    windows_per_day: int,
    split: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    eligible = [s for s in SCENARIOS if not s.specialty_only or s.key == persona.role.specialty]
    weights = [4.0 if s.label == 1 else 2.5 if s.label == 0 else 1.3 for s in eligible]
    lengths = _session_lengths(windows_per_day, rng)
    # Every UTC calendar day is self-contained, including long custom runs.
    latest_start = (1440 - windows_per_day * 5) // 60
    start_hour = min(latest_start, max(0, persona.start_hour_utc + rng.randint(-1, 1)))
    start = _EPOCH + timedelta(days=day_index, hours=start_hour)
    previous_context: str | None = None  # Also reset at split/date boundaries.
    rows: list[dict[str, Any]] = []
    for session_index, length in enumerate(lengths):
        # Stratified exploration plus probabilistic sessions: small tests need
        # not cover the complete catalogue; default generation does.
        if session_index % 3 == 0:
            recipe_index = day_index * 7 + persona.index * 11 + session_index
            scenario = eligible[recipe_index % len(eligible)]
        else:
            scenario = rng.choices(eligible, weights=weights, k=1)[0]
        session_id = f"{persona.id}:d{day_index:03d}:s{session_index:02d}"
        apps = _session_apps(persona.role, scenario, rng)
        baseline_level = rng.uniform(0.6, 1.45)
        level = baseline_level
        process_contexts = {app.process: ("unknown" if app.domain else app.context) for app in apps}
        domain_contexts = {app.domain: app.context for app in apps if app.domain}
        for _ in range(length):
            level = min(2.2, max(0.2, 0.85 * level + 0.15 * baseline_level + rng.gauss(0, 0.07)))
            end = start + timedelta(seconds=_WINDOW_SECONDS)
            events, buckets, browser = _raw_window(persona, scenario, apps, start, level, rng)
            built = build_v2_feature_window(
                events,
                buckets,
                browser,
                start,
                end,
                process_task_contexts=process_contexts,
                domain_task_contexts=domain_contexts,
                previous_task_context=previous_context,
            )
            previous_context = task_context_from_feature_window(built)
            observed_names = sorted({event.data.app_name for event in events})
            local = start.astimezone(timezone(timedelta(hours=persona.utc_offset_hours)))
            rows.append(
                {
                    "id": f"base:{persona.id}:{day_index:03d}:{len(rows):03d}",
                    "window_start_utc": start.isoformat(),
                    "window_end_utc": end.isoformat(),
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "features": {name: built[name] for name in V2_FEATURE_NAMES},
                    "label": scenario.label,
                    "label_source": "synthetic",
                    "persona_id": persona.id,
                    "role": persona.role.key,
                    "scenario": scenario.key,
                    "session_id": session_id,
                    "split": split,
                    "local_hour": local.hour,
                    "timezone": persona.timezone_name,
                    "apps": observed_names,
                    "simulation": {
                        "goal": scenario.goal,
                        "day_index": day_index,
                        "event_count": len(events),
                        "input_bucket_count": len(buckets),
                        "browser_segment_count": len(browser),
                        "confirmed_switch_count": count_confirmed_switches(events),
                    },
                }
            )
            start = end
    return rows


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    apps = sorted({str(app) for row in rows for app in row.get("apps", [])})
    roles = sorted({str(row.get("role")) for row in rows})
    scenarios = sorted({str(row.get("scenario")) for row in rows})
    labels = Counter(
        "unlabelled" if row.get("label") is None else str(row["label"]) for row in rows
    )
    app_labels: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if row.get("label") in (0, 1):
            for app in row.get("apps", []):
                app_labels[app].add(row["label"])
    return {
        "windows": len(rows),
        "roles": roles,
        "role_count": len(roles),
        "personas": len({row.get("persona_id") for row in rows}),
        "apps": apps,
        "app_count": len(apps),
        "scenarios": scenarios,
        "scenario_count": len(scenarios),
        "label_counts": dict(labels),
        "label_proportions": {label: count / max(len(rows), 1) for label, count in labels.items()},
        "split_counts": dict(Counter(str(row.get("split")) for row in rows)),
        "local_hours": sorted({row.get("local_hour") for row in rows}),
        "utc_weekdays": sorted(
            {
                datetime.fromisoformat(row["window_start_utc"]).weekday()
                for row in rows
                if isinstance(row.get("window_start_utc"), str)
            }
        ),
        "timezones": sorted({str(row.get("timezone")) for row in rows}),
        "dual_label_apps": sorted(app for app, values in app_labels.items() if values == {0, 1}),
        "sessions": len({row.get("session_id") for row in rows}),
    }


def generate_base_corpus(
    *,
    seed: int = 20261110,
    personas_per_role: int = 6,
    days: int = 28,
    windows_per_day: int = 48,
) -> SyntheticCorpus:
    """Generate a corpus using only local RNGs and the production v4 rollup.

    The last persona of each role is held out; the penultimate is calibration
    only. Other personas contribute the first 3/4 of UTC dates to training and
    the remaining dates to future testing. Calibration retains its split for
    all dates: consumers MUST filter it strictly before the calendar cutoff.
    Observed activity only is generated (default four hours/person/day), with
    no fabricated sleep windows. Different shifts/zones cover the full clock.
    """
    for name, value in (
        ("seed", seed),
        ("personas_per_role", personas_per_role),
        ("days", days),
        ("windows_per_day", windows_per_day),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if personas_per_role < 3 or days < 1 or not 2 <= windows_per_day <= 288:
        raise ValueError(
            "Require personas_per_role >= 3, days >= 1 and 2 <= windows_per_day <= 288"
        )
    rng = random.Random(seed)
    train_days = max(1, days * 3 // 4)
    cutoff = _EPOCH + timedelta(days=train_days)
    rows: list[dict[str, Any]] = []
    personas: dict[str, dict[str, Any]] = {}
    for role_index, role in enumerate(ROLES):
        for index in range(personas_per_role):
            zone_name, offset = TIMEZONES[(role_index + index) % len(TIMEZONES)]
            persona = _Persona(
                id=f"{role.key}:p{index:02d}",
                user_id=role_index * personas_per_role + index + 1,
                role=role,
                index=index,
                timezone_name=zone_name,
                utc_offset_hours=offset,
                proficiency=rng.uniform(0.55, 1.65),
                keyboard_preference=rng.uniform(0.55, 1.7),
                start_hour_utc=(role_index * 5 + index * 3) % 20,
            )
            fixed_split = (
                "test_persona"
                if index == personas_per_role - 1
                else "calibration"
                if index == personas_per_role - 2
                else None
            )
            personas[persona.id] = {
                "role": role.key,
                "index": index,
                "fixed_split": fixed_split,
                "timezone": zone_name,
                "utc_offset_hours": offset,
                "proficiency": persona.proficiency,
                "keyboard_preference": persona.keyboard_preference,
            }
            for day_index in range(days):
                split = fixed_split or ("train" if day_index < train_days else "test_future")
                rows.extend(_simulate_day(persona, day_index, windows_per_day, split, rng))
    coverage = _coverage(rows)
    return SyntheticCorpus(
        rows,
        {
            "synthetic_only": True,
            "contains_real_user_data": False,
            "purpose": "Synthetic prior; not real-world accuracy or personalisation evidence",
            "seed": seed,
            "recipe_version": _RECIPE_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": list(V2_FEATURE_NAMES),
            "personas_per_role": personas_per_role,
            "days": days,
            "windows_per_day": windows_per_day,
            "epoch_utc": _EPOCH.isoformat(),
            "train_days": train_days,
            "train_calendar_cutoff_utc": cutoff.isoformat(),
            "personas": personas,
            "recipe": {
                "features": "ActivityEvent + 30s input buckets + browser segments -> "
                "services.telemetry_features.build_v2_feature_window",
                "sessions": "10-60min sessions; AR(1) activity fluctuations within session",
                "labels": "Goal adherence=1; deviation=0; rest/missing/mixed/unknown=None",
                "ambiguity": "Same tools and input modes deliberately overlap between both labels",
                "splits": "Role's last persona=test_persona; penultimate=calibration; remaining "
                "personas' first train_days UTC dates=train, later dates=test_future",
                "calibration": "Filter calibration rows with window_end_utc <= "
                "train_calendar_cutoff_utc before fitting; future calibration stays held out",
                "clock": "UTC feature clock; local_hour uses declared fixed UTC offset; "
                "no timezone database/DST dependency",
                "observation": "Only sampled activity is represented; unsampled time has no label",
            },
            "catalog": {
                "roles": len(ROLES),
                "real_apps": len(APP_CATALOG) - 1,
                "scenarios": len(SCENARIOS),
            },
            "coverage": coverage,
        },
    )


def _audit_row(row: dict[str, Any], checks: dict[str, bool]) -> None:
    checks["required_fields"] &= row.keys() >= _REQUIRED_ROW_FIELDS
    features = row.get("features")
    if not isinstance(features, dict):
        checks["feature_names"] = False
        return
    checks["feature_names"] &= set(features) == set(V2_FEATURE_NAMES)
    checks["schema_version"] &= row.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
    checks["synthetic_provenance"] &= row.get("label_source") == "synthetic"
    label = row.get("label")
    checks["label_domain"] &= label is None or (type(label) is int and label in (0, 1))
    scenario = SCENARIO_CATALOG.get(str(row.get("scenario")))
    checks["scenario_labels"] &= scenario is not None and label == scenario.label
    if scenario is not None and scenario.label is None:
        checks["rest_missing_unlabelled"] &= label is None
    for name, value in features.items():
        valid = (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )
        checks["finite_values"] &= valid
        if not valid:
            continue
        checks["nonnegative_values"] &= value >= 0
        if name in _UNIT_FEATURES:
            checks["unit_ranges"] &= 0 <= value <= 1
    code = features.get("task_type_code")
    checks["task_codes"] &= code in TASK_TYPE_CODES.values()
    for name in ("app_switch_count", "domain_switch_count"):
        value = features.get(name, -1)
        checks["switch_ranges"] &= (
            isinstance(value, (int, float)) and 0 <= value <= 29 and value == int(value)
        )
    active, idle = features.get("active_seconds_ratio"), features.get("idle_ratio")
    input_active = features.get("input_active_ratio")
    checks["input_within_active_time"] &= (
        isinstance(active, (int, float))
        and isinstance(input_active, (int, float))
        and input_active <= active + 1e-5
    )
    if isinstance(active, (int, float)) and isinstance(idle, (int, float)):
        checks["activity_proportions"] &= math.isclose(active + idle, 1.0, abs_tol=1e-5)
    else:
        checks["activity_proportions"] = False
    for name, upper_bound in (
        ("keypress_rate_per_min", 1500),
        ("mouse_click_rate_per_min", 1000),
        ("mouse_distance_per_min", 200000),
    ):
        value = features.get(name)
        checks["input_ranges"] &= isinstance(value, (int, float)) and 0 <= value <= upper_bound
    checks["transition_binary"] &= features.get("task_context_transition") in (0.0, 1.0)
    expected = row.get("simulation", {}).get("confirmed_switch_count")
    checks["confirmed_switches"] &= (
        expected is not None and features.get("app_switch_count") == expected
    )
    if row.get("scenario") == "browser_closed":
        checks["browser_closed"] &= features.get("browser_ratio") == 0
    if row.get("scenario") == "input_missing":
        checks["missing_input"] &= all(
            features.get(name) == 0
            for name in (
                "keypress_rate_per_min",
                "mouse_click_rate_per_min",
                "input_active_ratio",
            )
        )
    checks["apps_list"] &= isinstance(row.get("apps"), list) and all(
        isinstance(app, str) for app in row.get("apps", [])
    )


def audit_base_corpus(corpus: SyntheticCorpus) -> dict[str, Any]:
    """Return executable invariants and measured coverage, including small runs.

    Coverage is descriptive; small parameter tests do not fail just because
    they did not visit every application. The default coverage gate is enabled
    only when the complete default-sized recipe was requested.
    """
    checks = dict.fromkeys(
        (
            "nonempty",
            "required_fields",
            "feature_names",
            "schema_version",
            "synthetic_provenance",
            "label_domain",
            "scenario_labels",
            "rest_missing_unlabelled",
            "finite_values",
            "nonnegative_values",
            "unit_ranges",
            "task_codes",
            "switch_ranges",
            "confirmed_switches",
            "activity_proportions",
            "input_within_active_time",
            "input_ranges",
            "transition_binary",
            "class_proportions",
            "non_overlapping_windows",
            "browser_closed",
            "missing_input",
            "apps_list",
            "utc_five_minute_windows",
            "local_clock",
            "unique_ids",
            "persona_split_isolation",
            "session_split_isolation",
            "session_duration",
            "session_contiguity",
            "calendar_split",
            "expected_size",
            "default_coverage",
        ),
        True,
    )
    rows, metadata = corpus.windows, corpus.metadata
    checks["nonempty"] = bool(rows)
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    persona_intervals: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
    ids: set[str] = set()
    personas = metadata.get("personas", {})
    cutoff = datetime.fromisoformat(metadata["train_calendar_cutoff_utc"])
    for row in rows:
        _audit_row(row, checks)
        row_id = str(row.get("id"))
        checks["unique_ids"] &= row_id not in ids
        ids.add(row_id)
        sessions[str(row.get("session_id"))].append(row)
        persona = personas.get(row.get("persona_id"), {})
        fixed = persona.get("fixed_split")
        index = persona.get("index")
        expected_fixed = (
            "test_persona"
            if index == metadata["personas_per_role"] - 1
            else "calibration"
            if index == metadata["personas_per_role"] - 2
            else None
        )
        split = row.get("split")
        checks["persona_split_isolation"] &= (
            bool(persona)
            and fixed == expected_fixed
            and row.get("role") == persona.get("role")
            and (split == fixed if fixed else split in {"train", "test_future"})
        )
        try:
            start = datetime.fromisoformat(row["window_start_utc"])
            end = datetime.fromisoformat(row["window_end_utc"])
            if start.utcoffset() == timedelta(0) and end.utcoffset() == timedelta(0):
                persona_intervals[str(row.get("persona_id"))].append((start, end))
            checks["utc_five_minute_windows"] &= (
                start.utcoffset() == timedelta(0)
                and end.utcoffset() == timedelta(0)
                and (end - start).total_seconds() == _WINDOW_SECONDS
                and start.minute % 5 == 0
                and start.second == 0
                and start.microsecond == 0
            )
            local = start.astimezone(timezone(timedelta(hours=persona["utc_offset_hours"])))
            checks["local_clock"] &= local.hour == row.get("local_hour") and row.get(
                "timezone"
            ) == persona.get("timezone")
            if fixed is None:
                checks["calendar_split"] &= end <= cutoff if split == "train" else start >= cutoff
        except (KeyError, TypeError, ValueError, OverflowError):
            checks["utc_five_minute_windows"] = False
            checks["local_clock"] = False
            checks["calendar_split"] = False
    for group in sessions.values():
        checks["session_split_isolation"] &= all(
            len({str(row.get(key)) for row in group}) == 1
            for key in ("persona_id", "split", "scenario")
        )
        checks["session_duration"] &= 2 <= len(group) <= 12
        starts = sorted(str(row.get("window_start_utc")) for row in group)
        try:
            checks["session_contiguity"] &= all(
                (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() == 300
                for a, b in zip(starts, starts[1:], strict=False)
            )
        except ValueError:
            checks["session_contiguity"] = False
    for intervals in persona_intervals.values():
        ordered = sorted(intervals)
        checks["non_overlapping_windows"] &= all(
            earlier[1] <= later[0] for earlier, later in zip(ordered, ordered[1:], strict=False)
        )
    expected = (
        len(ROLES) * metadata["personas_per_role"] * metadata["days"] * metadata["windows_per_day"]
    )
    checks["expected_size"] = len(rows) == expected
    # Timestamp errors are already reported above; coverage remains available
    # for the other dimensions even when a caller supplies a damaged timestamp.
    try:
        coverage = _coverage(rows)
    except (KeyError, TypeError, ValueError):
        coverage = {"windows": len(rows), "invalid_timestamps": True}
    if len(rows) >= 1000:
        proportions = coverage.get("label_proportions", {})
        checks["class_proportions"] = (
            0.03 <= proportions.get("0", 0) <= 0.9
            and 0.03 <= proportions.get("1", 0) <= 0.9
            and 0.01 <= proportions.get("unlabelled", 0) <= 0.8
        )
    if (
        metadata["personas_per_role"] >= 6
        and metadata["days"] >= 28
        and metadata["windows_per_day"] >= 48
    ):
        checks["default_coverage"] = (
            coverage.get("role_count", 0) >= 16
            and coverage.get("app_count", 0) >= 70
            and coverage.get("scenario_count", 0) >= 20
            and len(coverage.get("local_hours", [])) == 24
            and len(coverage.get("utc_weekdays", [])) == 7
            and len(coverage.get("timezones", [])) >= 4
            and len(coverage.get("dual_label_apps", [])) >= 10
        )
    return {"checks": checks, "coverage": coverage, "passed": all(checks.values())}
