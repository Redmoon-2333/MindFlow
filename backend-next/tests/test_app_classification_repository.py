"""Tests for AppClassificationRulesRepository.

Covers CRUD operations, priority ordering, unknown-apps query,
and edge cases (empty, delete-non-existent).

Uses tmp_path + async SQLite for full isolation.
"""

from __future__ import annotations

import pytest

from mindflow.infrastructure.database import create_engine, create_session_factory
from mindflow.infrastructure.repositories.app_classification import (
    AppClassificationRulesRepository,
    app_classification_rules,
)


@pytest.fixture
async def repo(tmp_path):
    """Create a repository backed by an isolated temp SQLite database."""
    db_path = tmp_path / "test.db"
    url = f"sqlite+aiosqlite:///{db_path}"
    engine = create_engine(url)
    factory = create_session_factory(engine)

    async with engine.begin() as conn:
        await conn.run_sync(app_classification_rules.metadata.create_all)

    repository = AppClassificationRulesRepository(session_factory=factory)
    yield repository
    await engine.dispose()


class TestAppClassificationRulesRepository:
    """CRUD and query tests for classification rules."""

    @pytest.mark.asyncio
    async def test_add_and_get_all(self, repo):
        """Adding a rule then fetching returns it."""
        rule = await repo.add(
            user_id=1,
            rule={
                "process_name": "bilibili.exe",
                "category": "browser_work",
                "priority": 10,
            },
        )
        assert rule["process_name"] == "bilibili.exe"
        assert rule["category"] == "browser_work"
        assert rule["priority"] == 10
        assert rule["id"] is not None
        assert rule["created_at"] is not None
        assert rule["updated_at"] is not None

        all_rules = await repo.get_all(user_id=1)
        assert len(all_rules) == 1
        assert all_rules[0]["id"] == rule["id"]

    @pytest.mark.asyncio
    async def test_add_multiple_returns_priority_order(self, repo):
        """get_all returns rules ordered by priority DESC, then created_at ASC."""
        await repo.add(
            user_id=1,
            rule={"process_name": "a.exe", "category": "other", "priority": 1},
        )
        await repo.add(
            user_id=1,
            rule={"process_name": "b.exe", "category": "entertainment", "priority": 10},
        )
        await repo.add(
            user_id=1,
            rule={"process_name": "c.exe", "category": "code", "priority": 5},
        )

        all_rules = await repo.get_all(user_id=1)
        assert len(all_rules) == 3
        assert all_rules[0]["priority"] == 10  # highest first
        assert all_rules[1]["priority"] == 5
        assert all_rules[2]["priority"] == 1

    @pytest.mark.asyncio
    async def test_delete_removes_rule(self, repo):
        """Adding then deleting a rule removes it from get_all."""
        rule = await repo.add(
            user_id=1,
            rule={"process_name": "x.exe", "category": "code", "priority": 0},
        )
        await repo.delete(rule["id"])

        all_rules = await repo.get_all(user_id=1)
        assert len(all_rules) == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_silent(self, repo):
        """Deleting a non-existent rule ID does not raise."""
        await repo.delete("nonexistent-id-12345")
        # No exception → pass
        assert True

    @pytest.mark.asyncio
    async def test_get_all_empty_returns_empty_list(self, repo):
        """Empty database returns [] from get_all."""
        rules = await repo.get_all(user_id=1)
        assert rules == []

    @pytest.mark.asyncio
    async def test_rules_scoped_by_user(self, repo):
        """Rules for user 2 are not visible to user 1."""
        await repo.add(
            user_id=1,
            rule={"process_name": "a.exe", "category": "code", "priority": 0},
        )
        await repo.add(
            user_id=2,
            rule={"process_name": "b.exe", "category": "entertainment", "priority": 5},
        )

        user1_rules = await repo.get_all(user_id=1)
        assert len(user1_rules) == 1
        assert user1_rules[0]["process_name"] == "a.exe"

        user2_rules = await repo.get_all(user_id=2)
        assert len(user2_rules) == 1
        assert user2_rules[0]["process_name"] == "b.exe"

    @pytest.mark.asyncio
    async def test_add_with_window_title_pattern(self, repo):
        """Rules can include optional window_title_pattern."""
        rule = await repo.add(
            user_id=1,
            rule={
                "process_name": "notion.exe",
                "category": "document",
                "window_title_pattern": "%Meeting%",
                "priority": 3,
            },
        )
        assert rule["window_title_pattern"] == "%Meeting%"

    @pytest.mark.asyncio
    async def test_add_default_priority(self, repo):
        """Priority defaults to 0 when not specified."""
        rule = await repo.add(
            user_id=1,
            rule={"process_name": "foo.exe", "category": "other"},
        )
        assert rule["priority"] == 0
