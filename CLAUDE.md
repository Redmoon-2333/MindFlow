# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Backend (Python 3.11+, conda env: mindflow)
cd mindflow-app/backend

# Install dependencies
pip install -r requirements.txt

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_api.py -v

# Run a single test
python -m pytest tests/test_features.py::test_calculate_focus_score -v

# Start dev server (with hot reload)
uvicorn mindflow.main:app --reload --host 127.0.0.1 --port 8765

# Train ML models with synthetic data
python -m mindflow.analyzer.train

# Frontend (when ready)
cd mindflow-app/frontend
npm install && npm run dev
```

API docs: `http://localhost:8765/docs` after starting the backend.

## Architecture

MindFlow is a local-first intelligent focus assistant that monitors computer usage, analyzes behavior patterns, and generates personalized anti-procrastination interventions.

```
Frontend (React/TS) ←→ Backend (FastAPI :8765) ←→ Collector (Win32 polling)
                              ↓
                         SQLite (data/mindflow.db)
```

**Module dependency chain:** `config` → `models` → `collector` → `analyzer` → `api`

| Module | Path | Responsibility |
|--------|------|----------------|
| `config` | `backend/mindflow/config.py` | Pydantic BaseSettings from `.env`, all tunable params |
| `models` | `backend/mindflow/models/` | SQLAlchemy ORM: User, ActivityLog, FocusSession, DailyReport |
| `collector` | `backend/mindflow/collector/` | Win32 active window polling + APScheduler background ticks |
| `analyzer` | `backend/mindflow/analyzer/` | Focus scoring, session detection, ML clustering/classification/HMM |
| `api` | `backend/mindflow/api/` | REST endpoints + WebSocket real-time push |
| `llm` | `backend/mindflow/llm/` | Placeholder (Phase 3) |
| `intervention` | `backend/mindflow/intervention/` | Placeholder (Phase 4) |

## Key Design Decisions

- **All data local**: SQLite via SQLAlchemy, no cloud upload. Privacy-first.
- **Collector is a global singleton**: `collector` instance in `scheduler.py` is imported by both `api/routes.py` and `api/websocket.py`.
- **Naive UTC datetimes**: All `DateTime` columns use naive UTC (`datetime.now(timezone.utc).replace(tzinfo=None)`) for compatibility with `datetime.combine()` queries in the analyzer.
- **Idempotent daily reports**: `generate_daily_report()` checks for existing record before computing; `identify_focus_sessions()` skips if sessions already exist for that day.
- **Duration estimation**: `duration_seconds` in ActivityLog uses `collect_interval_seconds` (config-based) rather than actual measured intervals.

## Current State

**Phase 0 (Demo) — Backend complete, Frontend pending:**
- ✅ Data collection, behavior analysis, ML models, REST API, WebSocket, tests (31 passing)
- ❌ React frontend (assigned to teammates Zhang Hao & Yang Zhijie)
- ❌ End-to-end verification (Task 0.7)

**Phases 1-5**: Planning only, see `docs/implementation-plan.md`.

## Dataset Context

Two external datasets in `data/datasets/`:
- `manictime/`: 44 CSV files of real user activity logs (ManicTime exports)
- `awt-labelled/`: Academic Work Tracker labeled data with preprocessing notebook

Synthetic data generator at `analyzer/data_pipeline.py::generate_synthetic_data()` models realistic weekday/weekend behavior patterns with Chinese app ecosystem awareness.
