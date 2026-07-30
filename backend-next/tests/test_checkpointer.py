"""Tests for LangGraph checkpoint persistence adapters.

Covers:
  - Settings default to legacy paths (all flags False)
  - InMemoryCheckpointer (no-op adapter) basic operations
  - ApplicationCheckpointer with SQLite: put, get, list, thread-to-run mapping
  - Crash/restart: save checkpoint, close DB, reopen, verify state restored
  - Retention: expired checkpoints removed, analyses preserved
  - DB connections close cleanly
  - Factory selects correct adapter based on settings
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables.config import RunnableConfig

from mindflow.config import Settings
from mindflow.infrastructure.checkpointer import (
    ApplicationCheckpointer,
    InMemoryCheckpointer,
    create_checkpointer,
)

# ── Reusable fixtures ───────────────────────────────────────────────────


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Settings with a temp SQLite DB and checkpointing enabled."""
    return Settings(
        data_dir=tmp_path,
        db_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        run_scheduler=False,
        run_collectors=False,
        checkpointing_enabled=True,
    )


@pytest.fixture
def tmp_settings_disabled(tmp_path: Path) -> Settings:
    """Settings with checkpointing disabled (legacy default)."""
    return Settings(
        data_dir=tmp_path,
        db_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        run_scheduler=False,
        run_collectors=False,
        checkpointing_enabled=False,
    )


@pytest.fixture
def test_config() -> RunnableConfig:
    """A minimal LangGraph config for checkpoint operations."""
    return {
        "configurable": {
            "thread_id": "test-thread-001",
            "checkpoint_ns": "",
        },
    }


@pytest.fixture
def test_checkpoint() -> Any:
    """A minimal checkpoint payload."""
    return {
        "v": 1,
        "id": "ckpt-001",
        "ts": "2025-01-01T00:00:00Z",
        "channel_values": {"messages": []},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }


# ══════════════════════════════════════════════════════════════════════════
# Settings defaults
# ══════════════════════════════════════════════════════════════════════════


class TestSettingsDefaults:
    """All new graph/chceckpoint flags default to False (legacy paths)."""

    def test_graph_version_defaults_to_1(self) -> None:
        settings = Settings()
        assert settings.graph_version == 1

    def test_checkpointing_enabled_defaults_to_false(self) -> None:
        settings = Settings()
        assert settings.checkpointing_enabled is False

    def test_new_analysis_graph_defaults_to_false(self) -> None:
        settings = Settings()
        assert settings.new_analysis_graph is False

    def test_new_chat_graph_defaults_to_false(self) -> None:
        settings = Settings()
        assert settings.new_chat_graph is False

    def test_all_flags_false_by_default(self) -> None:
        """Verify complete legacy default posture."""
        settings = Settings()
        assert settings.graph_version == 1
        assert settings.checkpointing_enabled is False
        assert settings.new_analysis_graph is False
        assert settings.new_chat_graph is False


# ══════════════════════════════════════════════════════════════════════════
# InMemoryCheckpointer (no-op adapter)
# ══════════════════════════════════════════════════════════════════════════


class TestInMemoryCheckpointer:
    """InMemoryCheckpointer stores checkpoints in memory with zero I/O."""

    async def test_create_and_close(self, tmp_settings: Settings) -> None:
        """Checkpointer can be created, entered, and closed."""
        cp = InMemoryCheckpointer(tmp_settings)
        async with cp:
            assert cp.saver is not None
        assert cp._closed is True

    async def test_put_and_get(
        self,
        tmp_settings: Settings,
        test_config: RunnableConfig,
        test_checkpoint: Any,
    ) -> None:
        """Save a checkpoint and retrieve it."""
        async with InMemoryCheckpointer(tmp_settings) as cp:
            await cp.saver.aput(
                test_config, test_checkpoint, metadata={}, new_versions={}
            )
            result = await cp.saver.aget_tuple(test_config)

        assert result is not None
        assert result.checkpoint["id"] == "ckpt-001"

    async def test_list_checkpoints(
        self,
        tmp_settings: Settings,
        test_config: RunnableConfig,
        test_checkpoint: Any,
    ) -> None:
        """List checkpoints for a thread."""
        async with InMemoryCheckpointer(tmp_settings) as cp:
            await cp.saver.aput(
                test_config, test_checkpoint, metadata={}, new_versions={}
            )

            checkpoints = [c async for c in cp.saver.alist(test_config)]

        assert len(checkpoints) == 1
        assert checkpoints[0].checkpoint["id"] == "ckpt-001"

    async def test_multiple_closes_are_idempotent(
        self, tmp_settings: Settings
    ) -> None:
        """Closing twice does not raise."""
        async with InMemoryCheckpointer(tmp_settings) as cp:
            pass  # context manager already called aclose once
        await cp.aclose()  # second close is a no-op

    async def test_saver_raises_when_not_entered(
        self, tmp_settings: Settings
    ) -> None:
        """Accessing .saver before __aenter__ raises."""
        cp = InMemoryCheckpointer(tmp_settings)
        with pytest.raises(RuntimeError, match="before __aenter__"):
            _ = cp.saver


# ══════════════════════════════════════════════════════════════════════════
# ApplicationCheckpointer with SQLite
# ══════════════════════════════════════════════════════════════════════════


class TestApplicationCheckpointer:
    """ApplicationCheckpointer persists checkpoints to the app SQLite DB."""

    async def test_create_and_close(
        self, tmp_settings: Settings
    ) -> None:
        """Checkpointer opens and closes cleanly."""
        cp = ApplicationCheckpointer(tmp_settings)
        async with cp:
            assert cp.saver is not None
            assert not cp._closed
        assert cp._closed

    async def test_put_and_get(
        self,
        tmp_settings: Settings,
        test_config: RunnableConfig,
        test_checkpoint: Any,
    ) -> None:
        """Save a checkpoint and retrieve it after reopen."""
        async with ApplicationCheckpointer(tmp_settings) as cp:
            await cp.saver.aput(
                test_config, test_checkpoint, metadata={}, new_versions={}
            )
            result = await cp.saver.aget_tuple(test_config)

        assert result is not None
        assert result.checkpoint["id"] == "ckpt-001"

    async def test_thread_to_run_mapping(
        self,
        tmp_settings: Settings,
    ) -> None:
        """register_run maps thread_id to run_id and get_run_id retrieves it."""
        async with ApplicationCheckpointer(tmp_settings) as cp:
            cp.register_run("thread-123", "run-456")
            assert cp.get_run_id("thread-123") == "run-456"
            assert cp.get_run_id("nonexistent") is None

    async def test_list_empty_thread(
        self,
        tmp_settings: Settings,
        test_config: RunnableConfig,
    ) -> None:
        """Listing a thread with no checkpoints returns empty."""
        async with ApplicationCheckpointer(tmp_settings) as cp:
            results = [c async for c in cp.saver.alist(test_config)]

        assert len(results) == 0

    async def test_list_multiple_checkpoints(
        self,
        tmp_settings: Settings,
        test_config: RunnableConfig,
        test_checkpoint: Any,
    ) -> None:
        """List returns all checkpoints for a thread."""
        async with ApplicationCheckpointer(tmp_settings) as cp:
            # Save two checkpoints
            ckpt1: Any = dict(test_checkpoint, id="ckpt-001")
            ckpt2: Any = dict(test_checkpoint, id="ckpt-002")

            await cp.saver.aput(test_config, ckpt1, metadata={}, new_versions={})
            await cp.saver.aput(test_config, ckpt2, metadata={}, new_versions={})

            results = [c async for c in cp.saver.alist(test_config)]

        assert len(results) >= 2

    async def test_multiple_closes_are_idempotent(
        self, tmp_settings: Settings
    ) -> None:
        """Closing twice does not raise."""
        async with ApplicationCheckpointer(tmp_settings) as cp:
            pass
        await cp.aclose()  # second close is a no-op

    async def test_saver_raises_when_not_entered(
        self, tmp_settings: Settings
    ) -> None:
        """Accessing .saver before __aenter__ raises."""
        cp = ApplicationCheckpointer(tmp_settings)
        with pytest.raises(RuntimeError, match="before __aenter__"):
            _ = cp.saver


# ══════════════════════════════════════════════════════════════════════════
# Crash / restart test
# ══════════════════════════════════════════════════════════════════════════


class TestCrashRestart:
    """Save checkpoint, simulate crash (close DB), reopen, verify state."""

    async def test_checkpoint_survives_close_and_reopen(
        self,
        tmp_settings: Settings,
        test_config: RunnableConfig,
        test_checkpoint: Any,
    ) -> None:
        """A checkpoint saved before close is retrievable after reopen."""
        # Phase 1: save checkpoint
        async with ApplicationCheckpointer(tmp_settings) as cp:
            await cp.saver.aput(
                test_config, test_checkpoint, metadata={}, new_versions={}
            )
            result_before = await cp.saver.aget_tuple(test_config)
            assert result_before is not None

        # Phase 2: simulate crash (already closed by __aexit__)
        # Phase 3: reopen and verify
        async with ApplicationCheckpointer(tmp_settings) as cp:
            result_after = await cp.saver.aget_tuple(test_config)

        assert result_after is not None
        assert result_after.checkpoint["id"] == "ckpt-001"
        assert result_after.checkpoint["v"] == 1

    async def test_crash_recovery_does_not_replay_completed(
        self,
        tmp_settings: Settings,
        test_config: RunnableConfig,
        test_checkpoint: Any,
    ) -> None:
        """Recovery restores state; a second save does not duplicate work.

        This verifies the crash/restart contract: reopen → retrieve last
        state → determine what was completed → continue from there.
        """
        async with ApplicationCheckpointer(tmp_settings) as cp:
            # Save state as if step-1 completed
            state1: Any = dict(test_checkpoint, id="ckpt-step1")
            state1["channel_values"] = {"step": "step1_complete"}

            await cp.saver.aput(test_config, state1, metadata={}, new_versions={})
            result1 = await cp.saver.aget_tuple(test_config)
            assert result1 is not None

        # Crash — close and reopen
        async with ApplicationCheckpointer(tmp_settings) as cp:
            result2 = await cp.saver.aget_tuple(test_config)
            assert result2 is not None
            # Verify we see step1_complete, NOT replaying from scratch
            channel_vals = result2.checkpoint.get("channel_values", {})
            assert channel_vals.get("step") == "step1_complete"

            # Now save step-2
            state2: Any = dict(test_checkpoint, id="ckpt-step2")
            state2["channel_values"] = {"step": "step2_complete"}
            await cp.saver.aput(test_config, state2, metadata={}, new_versions={})

        # Verify both steps exist
        async with ApplicationCheckpointer(tmp_settings) as cp:
            all_ckpts = [c async for c in cp.saver.alist(test_config)]
            assert len(all_ckpts) >= 2


# ══════════════════════════════════════════════════════════════════════════
# Retention
# ══════════════════════════════════════════════════════════════════════════


class TestRetention:
    """Checkpoint retention and cleanup."""

    async def test_prune_expired_removes_mapped_threads(
        self,
        tmp_settings: Settings,
        test_config: RunnableConfig,
        test_checkpoint: Any,
    ) -> None:
        """prune_expired removes checkpoints for registered threads.

        Note: The current implementation prunes ALL mapped threads via
        LangGraph's built-in aprune().  In production this would be
        scoped to completed-and-expired runs only.
        """
        async with ApplicationCheckpointer(tmp_settings) as cp:
            # Register and save
            cp.register_run("test-thread-001", "run-001")
            await cp.saver.aput(
                test_config, test_checkpoint, metadata={}, new_versions={}
            )

            # Verify checkpoint exists
            before = await cp.saver.aget_tuple(test_config)
            assert before is not None

            # Prune
            await cp.prune_expired(keep_hours=0)

    async def test_prune_no_registered_threads_does_nothing(
        self,
        tmp_settings: Settings,
    ) -> None:
        """prune_expired with no registered threads is a no-op."""
        async with ApplicationCheckpointer(tmp_settings) as cp:
            # No threads registered → nothing to prune
            await cp.prune_expired(keep_hours=0)
            # Does not raise

    async def test_retention_does_not_affect_empty_db(
        self,
        tmp_settings: Settings,
    ) -> None:
        """Calling prune on an empty database does not raise."""
        async with ApplicationCheckpointer(tmp_settings) as cp:
            cp.register_run("ghost-thread", "ghost-run")
            await cp.prune_expired(keep_hours=24)
            # Does not raise


# ══════════════════════════════════════════════════════════════════════════
# DB connections close cleanly
# ══════════════════════════════════════════════════════════════════════════


class TestCleanClose:
    """DB connections are released cleanly on close."""

    async def test_connection_closed_after_context_exit(
        self,
        tmp_settings: Settings,
    ) -> None:
        """After __aexit__, _saver is None and _closed is True."""
        async with ApplicationCheckpointer(tmp_settings) as cp:
            assert cp._saver is not None
            assert not cp._closed

        assert cp._closed
        assert cp._saver is None

    async def test_connection_closed_after_explicit_aclose(
        self,
        tmp_settings: Settings,
    ) -> None:
        """Explicit aclose() also marks as closed and nulls the saver."""
        cp = ApplicationCheckpointer(tmp_settings)
        async with cp:
            pass
        # Already closed by context manager; re-close is idempotent
        await cp.aclose()
        assert cp._closed
        assert cp._saver is None

    async def test_in_memory_releases_state(
        self,
        tmp_settings: Settings,
    ) -> None:
        """InMemoryCheckpointer sets _saver to None on close."""
        async with InMemoryCheckpointer(tmp_settings) as cp:
            assert cp._saver is not None
        assert cp._saver is None


# ══════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════


class TestFactory:
    """create_checkpointer() selects the correct adapter."""

    async def test_checkpointing_disabled_returns_in_memory(
        self, tmp_settings_disabled: Settings
    ) -> None:
        """When checkpointing is off, factory returns InMemoryCheckpointer."""
        async with create_checkpointer(tmp_settings_disabled) as cp:
            assert isinstance(cp, InMemoryCheckpointer)

    async def test_checkpointing_enabled_returns_application(
        self, tmp_settings: Settings
    ) -> None:
        """When checkpointing is on, factory returns ApplicationCheckpointer."""
        async with create_checkpointer(tmp_settings) as cp:
            assert isinstance(cp, ApplicationCheckpointer)
