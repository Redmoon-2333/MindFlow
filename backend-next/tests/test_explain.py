"""Tests for SHAP-based ModelExplainer.

Covers graceful degradation when shap is not installed, feature importance
ranking, per-sample explanations, and TrainingSummary backward compatibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from mindflow.train.models.types import TrainingSummary

# ── Module-level helpers (must precede decorators) ────────────────────────────


def _shap_available() -> bool:
    """Check if shap is installed in the current environment."""
    try:
        import shap  # noqa: F401

        return True
    except ImportError:
        return False


class _MockClassifier:
    """Minimal classifier stub for testing graceful degradation."""

    def __init__(self, feature_names: list[str]) -> None:
        self.model = self
        self.scaler = None
        self.feature_names_ = feature_names
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones(len(X), dtype=np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        probs = np.zeros((n, 2), dtype=np.float64)
        probs[:, 1] = 0.7
        probs[:, 0] = 0.3
        return probs


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def feature_names() -> list[str]:
    return [
        "unique_app_count",
        "switch_frequency",
        "productivity_ratio",
        "entertainment_ratio",
    ]


@pytest.fixture
def sample_X() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.beta(2, 5, (60, 4)).astype(np.float64)


@pytest.fixture
def sample_row() -> np.ndarray:
    rng = np.random.default_rng(99)
    return rng.beta(2, 5, 4).astype(np.float64)


# ── ModelExplainer ────────────────────────────────────────────────────────────


class TestModelExplainer:
    """Tests for ModelExplainer with and without shap installed."""

    def test_graceful_degradation_no_shap(
        self, feature_names: list[str], sample_X: np.ndarray
    ) -> None:
        """explain() should return a graceful message when shap is absent."""
        from mindflow.train.explain import ModelExplainer

        explainer = ModelExplainer(
            _MockClassifier(feature_names), feature_names
        )
        # Force unavailable
        explainer._available = False

        result = explainer.explain(sample_X)
        assert "summary" in result
        assert "SHAP not installed" in result["summary"]
        assert result["feature_importance"] == []

    def test_explain_sample_graceful_no_shap(
        self, feature_names: list[str], sample_row: np.ndarray
    ) -> None:
        """explain_sample should return sentinel values when shap absent."""
        from mindflow.train.explain import ModelExplainer

        explainer = ModelExplainer(
            _MockClassifier(feature_names), feature_names
        )
        explainer._available = False

        result = explainer.explain_sample(sample_row)
        assert result["prediction"] == -1
        assert result["confidence"] == 0.0
        assert result["top_factors"] == []
        assert "SHAP not installed" in result.get("note", "")

    @pytest.mark.skipif(not _shap_available(), reason="shap not installed")
    def test_explain_returns_ranking(
        self, feature_names: list[str], sample_X: np.ndarray
    ) -> None:
        """explain() should return feature_importance sorted descending."""
        from mindflow.train.models.classifier import FocusClassifier

        y = np.array([1 if i < 30 else 0 for i in range(60)], dtype=np.int32)
        clf = FocusClassifier()
        clf.fit(sample_X, y, feature_names)

        from mindflow.train.explain import ModelExplainer

        explainer = ModelExplainer(clf, feature_names)
        assert explainer._available

        result = explainer.explain(sample_X, max_samples=50)
        ranked = result["feature_importance"]
        assert len(ranked) == len(feature_names)
        # Verify descending order
        for i in range(len(ranked) - 1):
            assert ranked[i]["importance"] >= ranked[i + 1]["importance"]
        assert "Top features" in result["summary"]

    @pytest.mark.skipif(not _shap_available(), reason="shap not installed")
    def test_explain_sample_top_factors(
        self, feature_names: list[str], sample_row: np.ndarray
    ) -> None:
        """explain_sample should return top-3 factors with impact direction."""
        from mindflow.train.models.classifier import FocusClassifier

        rng = np.random.default_rng(42)
        X = rng.beta(2, 5, (60, 4)).astype(np.float64)
        y = np.array([1 if i < 30 else 0 for i in range(60)], dtype=np.int32)
        clf = FocusClassifier()
        clf.fit(X, y, feature_names)

        from mindflow.train.explain import ModelExplainer

        explainer = ModelExplainer(clf, feature_names)
        result = explainer.explain_sample(sample_row)

        assert result["prediction"] in (0, 1)
        assert 0.0 <= result["confidence"] <= 1.0
        assert len(result["top_factors"]) == 3
        for factor in result["top_factors"]:
            assert factor["feature"] in feature_names
            assert factor["impact"] in ("positive", "negative")
            assert factor["magnitude"] >= 0.0

    @pytest.mark.skipif(not _shap_available(), reason="shap not installed")
    def test_explain_with_scaler(self, sample_X: np.ndarray) -> None:
        """ModelExplainer should transparently apply FocusClassifier scaler."""
        from mindflow.train.models.classifier import FocusClassifier

        feat_names = [f"f{i}" for i in range(14)]
        y = np.array(
            [1 if i < 15 else 0 for i in range(len(sample_X))], dtype=np.int32
        )
        clf = FocusClassifier()
        clf.fit(sample_X, y, feat_names)
        assert clf._is_fitted

        from mindflow.train.explain import ModelExplainer

        explainer = ModelExplainer(clf, feat_names)
        result = explainer.explain(sample_X, max_samples=30)
        assert len(result["feature_importance"]) == len(feat_names)


# ── TrainingSummary backward compatibility ────────────────────────────────────


class TestTrainingSummaryExplanation:
    def test_default_explanation_empty(self) -> None:
        """TrainingSummary should default to empty dict for explanation."""
        summary = TrainingSummary(
            clustering={"n_clusters": 3},
            classifier={"accuracy": 0.85},
            hmm={"n_states": 4},
        )
        assert summary.explanation == {}

    def test_explicit_explanation(self) -> None:
        """TrainingSummary should accept explanation kwarg."""
        explanation = {
            "feature_importance": [
                {"name": "productivity_ratio", "importance": 0.42},
                {"name": "switch_frequency", "importance": 0.31},
            ],
            "summary": "Top features: productivity_ratio, switch_frequency",
        }
        summary = TrainingSummary(
            clustering={},
            classifier={},
            hmm={},
            explanation=explanation,
        )
        assert summary.explanation == explanation
