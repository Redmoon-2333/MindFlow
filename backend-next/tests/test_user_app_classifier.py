"""Tests for UserAppClassifier — domain-level app classification with
user rules, bilibili productive-learning heuristic, and built-in fallback.

Covers:
  - Bilibili productive learning heuristic (lecture vs anime)
  - Standard app classification unchanged
  - User rule matching (exact, case-insensitive, title pattern)
  - Rule priority (first match wins)
  - Async classification with user rules override
  - Productivity scoring
"""

from __future__ import annotations

import pytest

from mindflow.domain.app_classification import UserAppClassifier

# ── Fake repository for async rule tests ──────────────────────────────────


class FakeRulesRepo:
    """Fake ClassificationRulesProtocol that returns preset rules."""

    def __init__(self, rules=None):
        self._rules = rules or []

    async def get_all(self, user_id: int) -> list[dict]:
        return self._rules


# ── Helper: classifier without rules (heuristics only) ───────────────────


def _classifier() -> UserAppClassifier:
    """Create a UserAppClassifier with no user rules (heuristics only)."""
    return UserAppClassifier(rules_repo=None)


# ── Bilibili productive learning heuristic ────────────────────────────────


class TestUserAppClassifier:
    """Tests for the bilibili special case and general classification."""

    @pytest.mark.asyncio
    async def test_bilibili_lecture_is_productive(self):
        """bilibili with math lecture → browser_work via heuristic."""
        c = _classifier()
        result = await c.classify("bilibili.exe", "高等数学第3讲")
        assert result == "browser_work"

    @pytest.mark.asyncio
    async def test_bilibili_anime_is_entertainment(self):
        """bilibili with anime title → entertainment (falls through to base classifier)."""
        c = _classifier()
        result = await c.classify("bilibili.exe", "番剧推荐")
        assert result == "entertainment"

    @pytest.mark.asyncio
    async def test_bv_id_is_productive(self):
        """Any process with a bilibili BV ID → browser_work via heuristic."""
        c = _classifier()
        result = await c.classify("chrome.exe", "BV1xx411c7mD")
        assert result == "browser_work"

    @pytest.mark.asyncio
    async def test_standard_apps_unchanged(self):
        """Standard apps still classify correctly via base classifier."""
        c = _classifier()

        code_result = await c.classify("code.exe", "main.py - VS Code")
        assert code_result == "code"

        notion_result = await c.classify("notion.exe", "Project Notes")
        assert notion_result == "document"

        chrome_result = await c.classify("chrome.exe", "GitHub")
        assert chrome_result == "browser_work"

    @pytest.mark.asyncio
    async def test_unknown_app_is_other(self):
        """Unknown process falls through to base classifier → other."""
        c = _classifier()
        result = await c.classify("foobar_unknown_app.exe", "random window")
        assert result == "other"

    @pytest.mark.asyncio
    async def test_kaoyan_is_productive_learning(self):
        """考研-related title → productive learning heuristic."""
        c = _classifier()
        result = await c.classify("bilibili.exe", "考研数学真题讲解")
        assert result == "browser_work"


# ── User rule matching ────────────────────────────────────────────────────


class TestUserRuleMatching:
    """Tests for the _rule_matches static method on UserAppClassifier."""

    def test_rule_process_name_exact_match(self):
        """Rule matches when process name is identical (after lower/strip)."""
        rule = {"process_name": "bilibili.exe", "category": "browser_work"}
        assert UserAppClassifier._rule_matches(rule, "bilibili.exe", "any title") is True

    def test_rule_process_name_case_insensitive(self):
        """Rule matches process name case-insensitively."""
        rule = {"process_name": "Bilibili.EXE", "category": "browser_work"}
        assert UserAppClassifier._rule_matches(rule, "bilibili.exe", "any") is True

    def test_rule_process_name_whitespace_trimmed(self):
        """Rule matches after stripping whitespace from process name."""
        rule = {"process_name": "  bilibili.exe  ", "category": "browser_work"}
        assert UserAppClassifier._rule_matches(rule, "bilibili.exe", "any") is True

    def test_rule_process_name_no_match(self):
        """Rule does not match when process name is different."""
        rule = {"process_name": "code.exe", "category": "code"}
        assert UserAppClassifier._rule_matches(rule, "bilibili.exe", "any") is False

    def test_rule_with_title_pattern_match(self):
        """Rule with window_title_pattern matches when pattern matches title."""
        rule = {
            "process_name": "bilibili.exe",
            "window_title_pattern": "%高等数学%",
            "category": "browser_work",
        }
        assert (
            UserAppClassifier._rule_matches(rule, "bilibili.exe", "高等数学第3讲")
            is True
        )

    def test_rule_with_title_pattern_no_match(self):
        """Rule with window_title_pattern does not match when title differs."""
        rule = {
            "process_name": "bilibili.exe",
            "window_title_pattern": "%文档%",
            "category": "document",
        }
        assert (
            UserAppClassifier._rule_matches(rule, "bilibili.exe", "番剧推荐")
            is False
        )

    def test_rule_with_title_pattern_partial_multiple_percents(self):
        """% wildcard matches any substring, multiple % work."""
        rule = {
            "process_name": "bilibili.exe",
            "window_title_pattern": "%第%讲%",
            "category": "browser_work",
        }
        assert (
            UserAppClassifier._rule_matches(rule, "bilibili.exe", "高等数学第3讲总结")
            is True
        )

    def test_rule_empty_title_pattern_matches_any_title(self):
        """Rule with no title pattern matches any window title for that process."""
        rule = {"process_name": "bilibili.exe", "category": "entertainment"}
        assert (
            UserAppClassifier._rule_matches(rule, "bilibili.exe", "random window")
            is True
        )
        assert UserAppClassifier._rule_matches(rule, "bilibili.exe", "") is True


# ── Rule priority ordering (first match wins) ────────────────────────────


class TestRulePriority:
    """Tests for rule priority ordering in async classification."""

    @pytest.mark.asyncio
    async def test_priority_first_match_wins(self):
        """Higher priority rule's category is used even if lower-priority also matches."""
        rules = [
            {
                "process_name": "bilibili.exe",
                "category": "browser_work",
                "priority": 10,
                "window_title_pattern": "%高等数学%",
            },
            {
                "process_name": "bilibili.exe",
                "category": "entertainment",
                "priority": 1,
            },
        ]
        repo = FakeRulesRepo(rules)
        c = UserAppClassifier(rules_repo=repo)
        result = await c.classify("bilibili.exe", "高等数学第3讲")
        assert result == "browser_work"

    @pytest.mark.asyncio
    async def test_lower_priority_used_when_higher_does_not_match_title(self):
        """When a higher-priority rule has a title pattern that doesn't match,
        the next rule is checked."""
        rules = [
            {
                "process_name": "bilibili.exe",
                "category": "document",
                "priority": 10,
                "window_title_pattern": "%论文%",
            },
            {
                "process_name": "bilibili.exe",
                "category": "entertainment",
                "priority": 1,
            },
        ]
        repo = FakeRulesRepo(rules)
        c = UserAppClassifier(rules_repo=repo)
        result = await c.classify("bilibili.exe", "番剧推荐")
        assert result == "entertainment"


# ── Async classification with user rules ──────────────────────────────────


class TestAsyncUserAppClassifier:
    """Async classification tests with user-defined rules."""

    @pytest.mark.asyncio
    async def test_user_rules_override_productive_heuristic(self):
        """A user rule classifying bilibili as entertainment overrides even
        productive-learning windows."""
        rules = [
            {
                "process_name": "bilibili.exe",
                "category": "entertainment",
                "priority": 10,
            },
        ]
        repo = FakeRulesRepo(rules)
        c = UserAppClassifier(rules_repo=repo)
        result = await c.classify("bilibili.exe", "高等数学第3讲")
        assert result == "entertainment"

    @pytest.mark.asyncio
    async def test_falls_back_when_no_rules(self):
        """When no user rules exist, falls through to heuristic + base classifier."""
        repo = FakeRulesRepo([])
        c = UserAppClassifier(rules_repo=repo)

        result = await c.classify("code.exe", "main.py")
        assert result == "code"

    @pytest.mark.asyncio
    async def test_rules_from_fake_repo(self):
        """FakeRulesRepo returns preset rules correctly."""
        rules = [
            {"process_name": "x.exe", "category": "code", "priority": 1},
            {"process_name": "y.exe", "category": "entertainment", "priority": 2},
        ]
        repo = FakeRulesRepo(rules)
        retrieved = await repo.get_all(1)
        assert len(retrieved) == 2
        assert retrieved[0]["process_name"] == "x.exe"

    @pytest.mark.asyncio
    async def test_user_rule_with_no_title_pattern_matches_same_process(self):
        """A user rule without title pattern matches any window for the process."""
        rules = [
            {
                "process_name": "notion.exe",
                "category": "document",
                "priority": 5,
            },
        ]
        repo = FakeRulesRepo(rules)
        c = UserAppClassifier(rules_repo=repo)
        result = await c.classify("notion.exe", "random window")
        assert result == "document"

    @pytest.mark.asyncio
    async def test_user_rule_different_process_no_effect(self):
        """A user rule for one process does not affect other processes."""
        rules = [
            {"process_name": "bilibili.exe", "category": "code", "priority": 5},
        ]
        repo = FakeRulesRepo(rules)
        c = UserAppClassifier(rules_repo=repo)
        result = await c.classify("unknown.exe", "anything")
        assert result == "other"


# ── classify_sync (no async I/O path) ────────────────────────────────────


class TestClassifySync:
    """Tests for the synchronous classification path."""

    def test_sync_uses_heuristic_only(self):
        """classify_sync skips user rules, uses heuristic + base."""
        c = _classifier()
        result = c.classify_sync("bilibili.exe", "高等数学第3讲")
        assert result == "browser_work"

    def test_sync_anime_stays_entertainment(self):
        """classify_sync for anime → entertainment."""
        c = _classifier()
        result = c.classify_sync("bilibili.exe", "番剧推荐")
        assert result == "entertainment"


# ── Productivity scoring ──────────────────────────────────────────────────


class TestProductivityScore:
    """Tests for get_productivity_score() static method."""

    def test_code_is_1(self):
        assert UserAppClassifier.get_productivity_score("code") == 1.0

    def test_document_is_1(self):
        assert UserAppClassifier.get_productivity_score("document") == 1.0

    def test_browser_work_is_1(self):
        assert UserAppClassifier.get_productivity_score("browser_work") == 1.0

    def test_communication_is_0_5(self):
        assert UserAppClassifier.get_productivity_score("communication") == 0.5

    def test_entertainment_is_0(self):
        assert UserAppClassifier.get_productivity_score("entertainment") == 0.0

    def test_social_is_0(self):
        assert UserAppClassifier.get_productivity_score("social") == 0.0

    def test_other_is_0_3(self):
        assert UserAppClassifier.get_productivity_score("other") == 0.3

    def test_unknown_category_defaults_to_0_3(self):
        assert UserAppClassifier.get_productivity_score("nonexistent") == 0.3
