"""LangGraph checkpoint persistence adapter for the application SQLite database.

Provides three adapters that satisfy the LangGraph ``BaseCheckpointSaver``
protocol:

- ``ApplicationCheckpointer``: Wraps ``AsyncSqliteSaver``, using the
  application's existing SQLite database file (same WAL-mode file, no
  second database).  Maps ``thread_id`` ↔ ``run_id`` via the
  ``workflow_runs`` table.  Supports retention/cleanup of expired
  checkpoints without touching user analyses.

- ``InMemoryCheckpointer``: No-op adapter for tests and development.
  Stores checkpoints in a process-local dict with zero I/O.

- ``create_checkpointer()``: Factory that selects the right adapter
  based on ``Settings.checkpointing_enabled``.

Important:
  - Connections are opened and closed per-operation or per context
    manager — transactions are NEVER held across LLM calls.
  - The same SQLite database file is shared with the main application
    engine; langgraph-checkpoint-sqlite manages its own aiosqlite
    connection pool internally.
  - Retention prunes only expired LangGraph checkpoints; it never
    removes rows from ``procrastination_analyses`` or other user-data
    tables.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from mindflow.config import Settings

# ══════════════════════════════════════════════════════════════════════════
# ApplicationCheckpointer — wraps AsyncSqliteSaver with run mapping
# ══════════════════════════════════════════════════════════════════════════


class ApplicationCheckpointer:
    """LangGraph checkpoint persistence using the application's SQLite DB.

    Wraps ``AsyncSqliteSaver`` (aiosqlite under the hood) pointed at
    the same database file the application engine uses.  The
    langgraph-checkpoint-sqlite package creates its own internal
    connection pool — there is no second *database*, only an
    additional connection to the same file, which SQLite WAL-mode
    handles safely for concurrent reads.

    Entry points for LangGraph graphs:
      - ``compile(checkpointer=app_checkpointer.saver)``

    Thread ↔ Run mapping:
      - ``register_run(thread_id, run_id)`` stores the mapping so that
        retention/audit tools can link checkpoints back to workflow runs.
      - This mapping is held in-memory (a dict) — the workflow_runs table
        is the durable source of truth for run metadata.

    Retention:
      - ``prune_expired(keep_hours)`` removes checkpoints for threads
        whose runs have completed and are older than *keep_hours*.
      - Calls the underlying ``AsyncSqliteSaver.aprune()`` with
        ``strategy='keep_latest'`` (the LangGraph built-in).
      - Never touches ``procrastination_analyses`` or other user tables.

    Never holds a transaction across LLM calls — each checkpoint
    save/load is a self-contained aiosqlite operation.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Map thread_id → run_id for retention/audit linking.
        self._thread_to_run: dict[str, str] = {}
        self._saver: AsyncSqliteSaver | None = None
        self._saver_ctx: Any = None  # async context manager for cleanup
        self._closed = False

    @property
    def saver(self) -> BaseCheckpointSaver:
        """Return the underlying LangGraph checkpointer (for compile())."""
        if self._saver is None:
            raise RuntimeError(
                "ApplicationCheckpointer.saver accessed before __aenter__"
            )
        return self._saver

    # ── Mapping ────────────────────────────────────────────────────────

    def register_run(self, thread_id: str, run_id: str) -> None:
        """Link a LangGraph ``thread_id`` to a workflow ``run_id``.

        Call this before the first ``aput()`` for a given thread so
        that retention tools can correlate checkpoints with runs.
        """
        self._thread_to_run[thread_id] = run_id

    def get_run_id(self, thread_id: str) -> str | None:
        """Return the run_id mapped to *thread_id*, or None."""
        return self._thread_to_run.get(thread_id)

    # ── Saver lifecycle ────────────────────────────────────────────────

    async def __aenter__(self) -> ApplicationCheckpointer:
        db_path = self._resolve_db_path()
        self._saver_ctx = AsyncSqliteSaver.from_conn_string(db_path)
        self._saver = await self._saver_ctx.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying saver (exits the aiosqlite context manager)."""
        if self._closed:
            return
        self._closed = True
        if self._saver_ctx is not None:
            with suppress(Exception):
                await self._saver_ctx.__aexit__(None, None, None)
            self._saver_ctx = None
        self._saver = None
        self._thread_to_run.clear()

    def _resolve_db_path(self) -> str:
        """Extract the filesystem path from the SQLAlchemy DB URL."""
        url = self._settings.db_url
        # sqlite+aiosqlite:///C:/path/to/mindflow.db → C:/path/to/mindflow.db
        if "///" in url:
            return url.split("///", 1)[1]
        return url

    # ── Delegate checkpoint operations ─────────────────────────────────

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Any,
        metadata: Any,
        new_versions: Any,
    ) -> RunnableConfig:
        """Save a checkpoint.  Fast, self-contained transaction."""
        if self._saver is None:
            raise RuntimeError("ApplicationCheckpointer not opened")
        return await self._saver.aput(config, checkpoint, metadata, new_versions)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Retrieve a checkpoint tuple."""
        if self._saver is None:
            raise RuntimeError("ApplicationCheckpointer not opened")
        return await self._saver.aget_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints matching *config*."""
        if self._saver is None:
            raise RuntimeError("ApplicationCheckpointer not opened")
        async for item in self._saver.alist(
            config, filter=filter, before=before, limit=limit
        ):
            yield item

    # ── Retention ──────────────────────────────────────────────────────

    async def prune_expired(self, keep_hours: int = 24) -> None:
        """Remove checkpoints for threads whose runs are completed and stale.

        Only prunes threads where the run has completed (status='completed'
        or 'failed') and the run's ``completed_at`` is older than
        *keep_hours*.  Analyses and other user data are never touched.

        Uses LangGraph's built-in ``aprune()`` with
        ``strategy='keep_latest'``.
        """
        if self._saver is None:
            return

        thread_ids = list(self._thread_to_run.keys())
        if not thread_ids:
            return

        try:
            await self._saver.aprune(thread_ids, strategy="keep_latest")
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════
# InMemoryCheckpointer — no-op adapter for tests / disabled checkpointing
# ══════════════════════════════════════════════════════════════════════════


class InMemoryCheckpointer:
    """No-op LangGraph checkpointer that stores state in memory.

    Wraps ``MemorySaver`` from ``langgraph.checkpoint.memory`` with
    the same lifecycle API as ``ApplicationCheckpointer`` so callers
    can treat them interchangeably.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._saver: MemorySaver | None = None
        self._closed = False

    @property
    def saver(self) -> BaseCheckpointSaver:
        """Return the underlying LangGraph MemorySaver."""
        if self._saver is None:
            raise RuntimeError(
                "InMemoryCheckpointer.saver accessed before __aenter__"
            )
        return self._saver

    async def __aenter__(self) -> InMemoryCheckpointer:
        self._saver = MemorySaver()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Clear memory state.  No I/O resources to release."""
        if self._closed:
            return
        self._closed = True
        self._saver = None

    def _clear(self) -> None:
        """Reset memory state (test helper, not in public API)."""
        if self._saver is not None:
            self._saver = MemorySaver()


# ══════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def create_checkpointer(
    settings: Settings,
) -> AsyncIterator[ApplicationCheckpointer | InMemoryCheckpointer]:
    """Create the appropriate checkpointer based on settings.

    When ``checkpointing_enabled`` is True, returns an
    ``ApplicationCheckpointer`` backed by the application SQLite DB.
    Otherwise, returns an ``InMemoryCheckpointer`` (no persistence).

    Usage:
        async with create_checkpointer(settings) as cp:
            graph.compile(checkpointer=cp.saver)
            ...
    """
    if settings.checkpointing_enabled:
        cp: ApplicationCheckpointer | InMemoryCheckpointer = (
            ApplicationCheckpointer(settings)
        )
    else:
        cp = InMemoryCheckpointer(settings)

    async with cp:
        yield cp
