"""Behavioral feature extraction from activity events.

Ported from ``backend/mindflow/analyzer/data_pipeline.py``.
Key differences vs. the original:
  - Input is ``list[ActivityEvent]`` instead of ``pandas.DataFrame`` — no
    pandas dependency in the domain boundary.
  - Output is ``list[dict]`` (compatible with ``BaselineModel.update()`` and
    ``ConsensusLabeler.label_dataframe()``).
  - The ``AppClassifier`` and ``TitleAnalyzer`` are kept as pure dict-based
    classifiers with no external dependencies.

Feature columns (14 total per 30-minute window):
  - ``unique_app_count``, ``switch_frequency``, ``productivity_ratio``,
    ``entertainment_ratio``, ``social_ratio``, ``max_app_duration``,
    ``idle_ratio``, ``hour_of_day``, ``day_of_week``
  - ``title_code_ratio``, ``title_doc_ratio``, ``title_url_ratio``,
    ``title_meeting_ratio``, ``title_entertainment_ratio``
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

from mindflow.domain.events import ActivityEvent


class TitleAnalyzer:
    """Objective window title analysis — no app classification dependency.

    Extracts signal ratios from window titles by detecting:
      - Code file extensions (.py, .js, .tsx, …) and IDE patterns
      - Document extensions (.md, .pdf, .docx, …)
      - Browser URL patterns (http, www, .com …)
      - Meeting platform keywords (zoom, teams, meet …)
      - Entertainment keywords (bilibili, youtube, netflix …)
    """

    CODE_EXTENSIONS = frozenset({
        ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".cpp", ".c", ".h",
        ".rs", ".go", ".rb", ".php", ".swift", ".kt", ".scala", ".sql",
        ".sh", ".bash", ".yaml", ".yml", ".toml", ".json", ".xml", ".md",
        ".rst", ".ipynb",
    })
    CODE_KEYWORDS = frozenset({
        "vscode", "pycharm", "intellij", "vim", "nvim", "emacs",
        "terminal", "debug", "compile", "merge", "pull request",
    })
    DOC_EXTENSIONS = frozenset({".pdf", ".docx", ".doc", ".xlsx", ".pptx", ".tex"})
    DOC_KEYWORDS = frozenset({"obsidian", "notion", "evernote", "onenote", "typora"})
    BROWSER_KEYWORDS = frozenset({
        "http://", "https://", "www.", ".com", ".org", ".cn", ".io",
    })
    MEETING_KEYWORDS = frozenset({
        "zoom", "teams", "meet", "webex", "tencent meeting", "dingtalk",
    })
    ENTERTAINMENT_KEYWORDS = frozenset({
        "bilibili", "youtube", "netflix", "douyin", "tiktok", "anime",
        "steam", "spotify", "iqiyi", "youku",
    })

    FEATURE_KEYS = [
        "title_code_ratio",
        "title_doc_ratio",
        "title_url_ratio",
        "title_meeting_ratio",
        "title_entertainment_ratio",
    ]

    def analyze(self, title: str) -> dict[str, float]:
        """Analyze a single window title and return signal ratios.

        Returns a dict with keys: is_code_editor, is_document, is_browser,
        is_meeting, is_likely_entertainment — each 0.0 or 1.0.
        """
        lower = title.lower().strip()
        return {
            "is_code_editor": 1.0 if self._is_code(lower) else 0.0,
            "is_document": 1.0 if self._is_document(lower) else 0.0,
            "is_browser": 1.0 if self._is_browser(lower) else 0.0,
            "is_meeting": 1.0 if self._is_meeting(lower) else 0.0,
            "is_likely_entertainment": 1.0 if self._is_entertainment(lower) else 0.0,
        }

    def _is_code(self, title: str) -> bool:
        if any(ext in title for ext in self.CODE_EXTENSIONS):
            return True
        return any(kw in title for kw in self.CODE_KEYWORDS)

    def _is_document(self, title: str) -> bool:
        if any(ext in title for ext in self.DOC_EXTENSIONS):
            return True
        return any(kw in title for kw in self.DOC_KEYWORDS)

    def _is_browser(self, title: str) -> bool:
        return any(kw in title for kw in self.BROWSER_KEYWORDS)

    def _is_meeting(self, title: str) -> bool:
        return any(kw in title for kw in self.MEETING_KEYWORDS)

    def _is_entertainment(self, title: str) -> bool:
        return any(kw in title for kw in self.ENTERTAINMENT_KEYWORDS)


class AppClassifier:
    """Classify applications into productivity categories.

    Uses process name matching and window title keyword heuristics.
    No external dependencies — pure dict-based lookup.
    """

    PRODUCTIVITY_APPS: dict[str, list[str]] = {
        "code": [
            "code", "vscode", "pycharm", "intellij", "eclipse", "sublime",
            "nvim", "vim", "android studio", "visual studio", "cursor",
            "claude", "terminal", "powershell", "cmd", "warp", "alacritty",
            "xcode", "rstudio", "spyder", "datagrip",
        ],
        "document": [
            "word", "excel", "powerpoint", "wps", "notion", "obsidian",
            "typora", "pdf", "evernote", "onenote", "outlook",
        ],
        "browser_work": [
            "github", "stackoverflow", "docs", "jupyter", "colab", "arxiv",
            "scholar", "gitlab", "bitbucket",
        ],
        "communication": [
            "teams", "slack", "dingtalk", "feishu", "wechat", "qq",
            "discord", "telegram", "zoom", "meet",
        ],
        "entertainment": [
            "bilibili", "youtube", "netflix", "douyin", "tiktok", "game",
            "steam", "epic", "iqiyi", "youku", "spotify",
        ],
        "social": [
            "weibo", "twitter", "zhihu", "reddit", "douban", "xiaohongshu",
            "facebook", "instagram",
        ],
    }

    def __init__(self) -> None:
        self._lowercase_map: dict[str, str] = {}
        for category, app_list in self.PRODUCTIVITY_APPS.items():
            for app in app_list:
                self._lowercase_map[app.lower()] = category

        self._title_keywords: dict[str, list[str]] = {
            "browser_work": [
                "github", "stackoverflow", "jupyter", "colab", "docs", "documentation",
            ],
            "entertainment": ["bilibili", "youtube", "netflix", "game", "anime"],
            "social": ["weibo", "twitter", "reddit", "zhihu"],
        }

    def classify(self, process_name: str, window_title: str) -> str:
        """Classify into: code, document, browser_work, communication,
        entertainment, social, other."""
        pname = str(process_name).lower().strip()
        wtitle = str(window_title).lower().strip()

        if pname in self._lowercase_map:
            return self._lowercase_map[pname]

        for app_name, category in self._lowercase_map.items():
            if app_name in pname:
                return category

        for category, keywords in self._title_keywords.items():
            if any(kw in wtitle or kw in pname for kw in keywords):
                return category

        if any(
            browser in pname
            for browser in ["chrome", "firefox", "edge", "safari"]
        ):
            return "browser_work"

        return "other"

    @staticmethod
    def get_productivity_score(category: str) -> float:
        """Return 0.0-1.0 productivity score for a given category."""
        scores = {
            "code": 1.0,
            "document": 1.0,
            "browser_work": 1.0,
            "communication": 0.5,
            "entertainment": 0.0,
            "social": 0.0,
            "other": 0.3,
        }
        return scores.get(category, 0.3)


class BehaviorFeatureExtractor:
    """Extract behavioral features from raw activity events.

    Groups events into fixed-size time windows (default 30 min) and computes
    14 behavioral features per window.

    Args:
        window_minutes: Size of each time window in minutes.
    """

    FEATURE_NAMES = [
        "unique_app_count",
        "switch_frequency",
        "productivity_ratio",
        "entertainment_ratio",
        "social_ratio",
        "max_app_duration",
        "idle_ratio",
        "hour_of_day",
        "day_of_week",
        "title_code_ratio",
        "title_doc_ratio",
        "title_url_ratio",
        "title_meeting_ratio",
        "title_entertainment_ratio",
    ]

    def __init__(self, window_minutes: int = 30) -> None:
        self.window_minutes = window_minutes
        self.app_classifier = AppClassifier()
        self.title_analyzer = TitleAnalyzer()
        self._feature_names: list[str] = []

    def extract_session_features(
        self, events: Sequence[ActivityEvent]
    ) -> list[dict[str, Any]]:
        """Extract features per time window from activity events.

        Args:
            events: Chronologically ordered activity events.

        Returns:
            List of feature dicts, one per non-empty window, with keys
            matching ``FEATURE_NAMES`` plus ``window_start`` (ISO string).
        """
        if not events:
            return []

        # Convert to internal dicts for processing (no pandas dependency)
        raw_rows: list[dict[str, Any]] = []
        for ev in events:
            raw_rows.append({
                "timestamp": ev.timestamp_utc,
                "process_name": ev.data.process_name,
                "window_title": ev.data.window_title,
                "duration_seconds": ev.duration_s,
                "is_idle": 1 if ev.data.is_idle else 0,
            })

        min_ts = min(r["timestamp"] for r in raw_rows)
        max_ts = max(r["timestamp"] for r in raw_rows)
        window_delta = timedelta(minutes=self.window_minutes)

        # Build window boundaries
        window_start = self._floor_timestamp(min_ts)
        boundaries: list[Any] = []
        current = window_start
        while current <= max_ts:
            boundaries.append(current)
            current += window_delta

        if len(boundaries) < 2:
            return []

        records: list[dict[str, Any]] = []
        for i in range(len(boundaries) - 1):
            w_start = boundaries[i]
            w_end = boundaries[i + 1]

            window_rows = [
                r for r in raw_rows if w_start <= r["timestamp"] < w_end
            ]
            if not window_rows:
                continue

            total_seconds = sum(r["duration_seconds"] for r in window_rows)
            idle_seconds = sum(
                r["duration_seconds"] for r in window_rows if r["is_idle"]
            )
            active_seconds = total_seconds - idle_seconds
            if active_seconds <= 0:
                continue

            idle_ratio = idle_seconds / max(total_seconds, 0.01)

            # Classify apps
            for r in window_rows:
                category = self.app_classifier.classify(
                    str(r.get("process_name", "")),
                    str(r.get("window_title", "")),
                )
                r["app_category"] = category

            productivity_seconds = sum(
                r["duration_seconds"]
                for r in window_rows
                if r.get("app_category") in ("code", "document", "browser_work")
            )
            entertainment_seconds = sum(
                r["duration_seconds"]
                for r in window_rows
                if r.get("app_category") == "entertainment"
            )
            social_seconds = sum(
                r["duration_seconds"]
                for r in window_rows
                if r.get("app_category") == "social"
            )

            unique_apps = len({r["process_name"] for r in window_rows})

            process_list = [r["process_name"] for r in window_rows]
            switches = sum(
                1 for j in range(1, len(process_list))
                if process_list[j] != process_list[j - 1]
            )
            hours_in_window = self.window_minutes / 60.0
            switch_freq = switches / hours_in_window if hours_in_window > 0 else 0.0

            app_durations: dict[str, float] = defaultdict(float)
            for r in window_rows:
                app_durations[r["process_name"]] += r["duration_seconds"]
            max_app_duration = max(app_durations.values()) if app_durations else 0.0

            # Title-based features (no app classification dependency)
            title_features = [
                self.title_analyzer.analyze(str(r.get("window_title", "")))
                for r in window_rows
            ]
            n_titles = max(len(title_features), 1)
            code_ratio = sum(tf["is_code_editor"] for tf in title_features) / n_titles
            doc_ratio = sum(tf["is_document"] for tf in title_features) / n_titles
            url_ratio = sum(tf["is_browser"] for tf in title_features) / n_titles
            meeting_ratio = sum(tf["is_meeting"] for tf in title_features) / n_titles
            entertainment_title_ratio = (
                sum(tf["is_likely_entertainment"] for tf in title_features) / n_titles
            )

            records.append({
                "window_start": w_start.isoformat(),
                "unique_app_count": unique_apps,
                "switch_frequency": round(switch_freq, 4),
                "productivity_ratio": round(
                    productivity_seconds / active_seconds, 4
                ),
                "entertainment_ratio": round(
                    entertainment_seconds / active_seconds, 4
                ),
                "social_ratio": round(social_seconds / active_seconds, 4),
                "max_app_duration": round(max_app_duration, 2),
                "idle_ratio": round(idle_ratio, 4),
                "hour_of_day": int(w_start.hour),
                "day_of_week": int(w_start.weekday()),
                "title_code_ratio": round(code_ratio, 4),
                "title_doc_ratio": round(doc_ratio, 4),
                "title_url_ratio": round(url_ratio, 4),
                "title_meeting_ratio": round(meeting_ratio, 4),
                "title_entertainment_ratio": round(entertainment_title_ratio, 4),
            })

        if records:
            self._feature_names = list(self.FEATURE_NAMES)

        return records

    @staticmethod
    def _floor_timestamp(dt: Any) -> Any:
        """Floor a datetime to the nearest hour (coarse alignment)."""
        return dt.replace(minute=0, second=0, microsecond=0)

    def get_feature_names(self) -> list[str]:
        """Return ordered list of feature names."""
        return self._feature_names if self._feature_names else list(self.FEATURE_NAMES)
