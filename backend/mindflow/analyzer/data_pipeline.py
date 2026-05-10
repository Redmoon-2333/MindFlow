"""Feature engineering pipeline for MindFlow behavior data."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from mindflow.analyzer.title_analyzer import TitleAnalyzer


class BehaviorFeatureExtractor:
    """Extract behavioral features from raw activity logs."""

    def __init__(self, window_minutes: int = 30):
        self.window_minutes = window_minutes
        self.app_classifier = AppClassifier()
        self.title_analyzer = TitleAnalyzer()
        self._feature_names: list[str] = []

    def extract_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features per time window.

        Input columns: timestamp, process_name, window_title, duration_seconds, is_idle

        Returns DataFrame with features per window:
        - window_start: timestamp
        - unique_app_count: int
        - switch_frequency: float (switches per hour)
        - productivity_ratio: float (productivity app time / total time)
        - entertainment_ratio: float
        - social_ratio: float
        - max_app_duration: float (longest single app usage)
        - idle_ratio: float
        - hour_of_day: int
        - day_of_week: int
        """
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["app_category"] = df.apply(
            lambda row: self.app_classifier.classify(
                str(row.get("process_name", "")),
                str(row.get("window_title", ""))
            ),
            axis=1,
        )
        df["productivity_score"] = df["app_category"].apply(
            self.app_classifier.get_productivity_score
        )

        start_time = df["timestamp"].min()
        end_time = df["timestamp"].max()
        window_delta = timedelta(minutes=self.window_minutes)

        window_boundaries = pd.date_range(
            start=start_time.floor(f"{self.window_minutes}min"),
            end=end_time.ceil(f"{self.window_minutes}min"),
            freq=f"{self.window_minutes}min",
        )

        if len(window_boundaries) < 2:
            return pd.DataFrame()

        records: list[dict] = []
        for i in range(len(window_boundaries) - 1):
            window_start = window_boundaries[i]
            window_end = window_boundaries[i + 1]

            mask = (df["timestamp"] >= window_start) & (df["timestamp"] < window_end)
            window_df = df.loc[mask]

            if window_df.empty:
                continue

            total_seconds = float(window_df["duration_seconds"].sum())
            idle_seconds = float(
                window_df.loc[window_df["is_idle"].astype(bool), "duration_seconds"].sum()
            )
            active_seconds = total_seconds - idle_seconds

            if active_seconds <= 0:
                continue

            idle_ratio = idle_seconds / max(total_seconds, 0.01)

            productivity_seconds = float(
                window_df.loc[
                    window_df["app_category"].isin(["code", "document", "browser_work"]),
                    "duration_seconds",
                ].sum()
            )
            entertainment_seconds = float(
                window_df.loc[
                    window_df["app_category"] == "entertainment", "duration_seconds"
                ].sum()
            )
            social_seconds = float(
                window_df.loc[
                    window_df["app_category"] == "social", "duration_seconds"
                ].sum()
            )

            unique_apps = window_df["process_name"].nunique()

            process_list = window_df["process_name"].tolist()
            switches = sum(
                1 for j in range(1, len(process_list)) if process_list[j] != process_list[j - 1]
            )
            hours_in_window = self.window_minutes / 60.0
            switch_freq = switches / hours_in_window if hours_in_window > 0 else 0.0

            app_durations = (
                window_df.groupby("process_name")["duration_seconds"].sum()
            )
            max_app_duration = float(app_durations.max()) if len(app_durations) > 0 else 0.0

            # Title-based features (objective, no app classification)
            title_features = [
                self.title_analyzer.analyze(str(t))
                for t in window_df.get("window_title", pd.Series([""] * len(window_df)))
            ]
            code_ratio = sum(1 for tf in title_features if tf["is_code_editor"]) / max(1, len(title_features))
            doc_ratio = sum(1 for tf in title_features if tf["is_document"]) / max(1, len(title_features))
            url_ratio = sum(1 for tf in title_features if tf["is_browser"]) / max(1, len(title_features))
            meeting_ratio = sum(1 for tf in title_features if tf["is_meeting"]) / max(1, len(title_features))
            entertainment_title_ratio = sum(1 for tf in title_features if tf["is_likely_entertainment"]) / max(1, len(title_features))

            records.append(
                {
                    "window_start": window_start,
                    "unique_app_count": unique_apps,
                    "switch_frequency": round(switch_freq, 4),
                    "productivity_ratio": round(productivity_seconds / active_seconds, 4),
                    "entertainment_ratio": round(entertainment_seconds / active_seconds, 4),
                    "social_ratio": round(social_seconds / active_seconds, 4),
                    "max_app_duration": round(max_app_duration, 2),
                    "idle_ratio": round(idle_ratio, 4),
                    "hour_of_day": int(window_start.hour),
                    "day_of_week": int(window_start.weekday()),
                    "title_code_ratio": round(code_ratio, 4),
                    "title_doc_ratio": round(doc_ratio, 4),
                    "title_url_ratio": round(url_ratio, 4),
                    "title_meeting_ratio": round(meeting_ratio, 4),
                    "title_entertainment_ratio": round(entertainment_title_ratio, 4),
                }
            )

        result = pd.DataFrame(records)
        if not result.empty:
            self._feature_names = [
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
        return result

    def extract_daily_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sessions into daily profiles. Returns daily feature vectors."""
        sessions = self.extract_session_features(df)
        if sessions.empty:
            return pd.DataFrame()

        sessions["date"] = pd.to_datetime(sessions["window_start"]).dt.date

        feature_cols = [c for c in self._feature_names if c in sessions.columns]

        daily_agg: dict[str, str] = {}
        for col in feature_cols:
            if col in ("hour_of_day", "day_of_week"):
                daily_agg[col] = "median"
            else:
                daily_agg[col] = "mean"

        daily = sessions.groupby("date").agg(daily_agg).reset_index()

        daily["session_count"] = sessions.groupby("date").size().values
        daily["total_active_hours"] = (
            daily["session_count"] * self.window_minutes / 60.0
        )

        self._feature_names = [
            "session_count",
            "total_active_hours",
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
        return daily

    def get_feature_names(self) -> list[str]:
        """Return ordered list of feature names."""
        if not self._feature_names:
            return [
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
        return self._feature_names


class AppClassifier:
    """Classify applications into productivity categories."""

    PRODUCTIVITY_APPS = {
        "code": [
            "code", "vscode", "pycharm", "intellij", "eclipse", "sublime",
            "nvim", "vim", "android studio", "visual studio", "cursor",
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

    def __init__(self):
        self._lowercase_map: dict[str, str] = {}
        for category, app_list in self.PRODUCTIVITY_APPS.items():
            for app in app_list:
                self._lowercase_map[app.lower()] = category

        self._title_keywords: dict[str, list[str]] = {
            "browser_work": ["github", "stackoverflow", "jupyter", "colab", "docs", "documentation"],
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
            for kw in keywords:
                if kw in wtitle or kw in pname:
                    return category

        if any(browser in pname for browser in ["chrome", "firefox", "edge", "safari"]):
            return "browser_work"

        return "other"

    def get_productivity_score(self, category: str) -> float:
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


def generate_synthetic_data(days: int = 7, samples_per_hour: int = 12) -> pd.DataFrame:
    """Generate realistic synthetic activity data.

    Patterns:
    - Morning focus (9-12): code/doc dominant
    - Afternoon mixed (13-17): work + browsing
    - Evening (19-22): entertainment/social dominant

    Returns DataFrame with: timestamp, process_name, window_title, duration_seconds, is_idle
    """
    rng = np.random.default_rng(42)
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    interval_seconds = 3600 // samples_per_hour

    apps_by_pattern = {
        "morning_focus": {
            "apps": ["vscode", "pycharm", "notion", "typora", "terminal"],
            "titles": ["main.py - MindFlow", "analysis.ipynb - Jupyter", "notes - Obsidian",
                       "论文初稿.md - Typora", "terminal - zsh"],
            "weights": [0.35, 0.25, 0.15, 0.15, 0.10],
        },
        "afternoon_mixed": {
            "apps": ["chrome", "teams", "vscode", "wechat", "excel"],
            "titles": ["github.com/MindFlow - Chrome", "Teams Meeting", "app.py - VSCode",
                       "WeChat", "季度报告.xlsx - Excel"],
            "weights": [0.25, 0.20, 0.20, 0.20, 0.15],
        },
        "evening_leisure": {
            "apps": ["bilibili", "youtube", "steam", "wechat", "chrome"],
            "titles": ["B站 - 番剧", "YouTube - Music", "Steam", "WeChat朋友圈",
                       "知乎 - Chrome"],
            "weights": [0.30, 0.20, 0.15, 0.20, 0.15],
        },
        "late_night": {
            "apps": ["bilibili", "douyin", "weibo", "chrome", "wechat"],
            "titles": ["B站 - 深夜档", "抖音", "微博热搜", "reddit - Chrome", "WeChat"],
            "weights": [0.30, 0.25, 0.15, 0.15, 0.15],
        },
        "early_morning": {
            "apps": ["chrome", "notion", "wechat", "calendar", "mail"],
            "titles": ["Gmail - Chrome", "今日计划 - Notion", "WeChat消息", "Calendar", "邮件"],
            "weights": [0.30, 0.25, 0.20, 0.15, 0.10],
        },
    }

    idle_apps = ["", "lock_screen", "screensaver"]
    idle_titles = ["", "锁定", "屏保"]

    rows: list[dict] = []
    current_time = start_date

    for day_offset in range(days):
        day_start = start_date + timedelta(days=day_offset)
        day_of_week = day_start.weekday()
        is_weekend = day_of_week >= 5

        for hour in range(24):
            for sample in range(samples_per_hour):
                ts = day_start + timedelta(hours=hour, seconds=sample * interval_seconds)

                is_idle = 0
                proc_name = ""
                win_title = ""

                if is_weekend:
                    pattern_key = _weekend_pattern(hour, rng)
                else:
                    pattern_key = _weekday_pattern(hour, rng)

                if hour < 2 or hour >= 23:
                    idle_chance = 0.85
                elif pattern_key == "early_morning":
                    idle_chance = 0.25
                elif pattern_key == "evening_leisure":
                    idle_chance = 0.03
                elif pattern_key == "late_night":
                    idle_chance = 0.15
                else:
                    idle_chance = 0.05

                if rng.random() < idle_chance:
                    is_idle = 1
                    idx = rng.integers(0, len(idle_apps))
                    proc_name = idle_apps[idx]
                    win_title = idle_titles[min(idx, len(idle_titles) - 1)]
                    duration = max(1, int(rng.normal(120, 30)))
                else:
                    pattern = apps_by_pattern[pattern_key]
                    apps = pattern["apps"]
                    titles = pattern["titles"]
                    weights = np.array(pattern["weights"])
                    weights = weights / weights.sum()

                    idx = rng.choice(len(apps), p=weights)
                    proc_name = apps[idx]
                    win_title = titles[idx]

                    base_duration = (3600 // samples_per_hour)
                    duration = max(1, int(rng.normal(base_duration, base_duration * 0.2)))

                rows.append(
                    {
                        "timestamp": ts,
                        "process_name": proc_name,
                        "window_title": win_title,
                        "duration_seconds": duration,
                        "is_idle": is_idle,
                    }
                )

    result = pd.DataFrame(rows)
    result = result.sort_values("timestamp").reset_index(drop=True)
    return result


def _weekday_pattern(hour: int, rng: np.random.Generator) -> str:
    """Determine behavior pattern for a weekday hour."""
    if 0 <= hour < 6:
        return "late_night"
    elif 6 <= hour < 9:
        return "early_morning"
    elif 9 <= hour < 12:
        if rng.random() < 0.85:
            return "morning_focus"
        return "afternoon_mixed"
    elif 12 <= hour < 13:
        if rng.random() < 0.5:
            return "afternoon_mixed"
        return "evening_leisure"
    elif 13 <= hour < 18:
        if rng.random() < 0.80:
            return "afternoon_mixed"
        return "morning_focus"
    elif 18 <= hour < 19:
        return "afternoon_mixed"
    elif 19 <= hour < 22:
        if rng.random() < 0.10:
            return "morning_focus"
        return "evening_leisure"
    else:
        if rng.random() < 0.60:
            return "evening_leisure"
        return "late_night"


def _weekend_pattern(hour: int, rng: np.random.Generator) -> str:
    """Determine behavior pattern for a weekend hour."""
    if 0 <= hour < 7:
        return "late_night"
    elif 7 <= hour < 10:
        return "early_morning"
    elif 10 <= hour < 13:
        if rng.random() < 0.30:
            return "morning_focus"
        return "evening_leisure"
    elif 13 <= hour < 18:
        if rng.random() < 0.45:
            return "afternoon_mixed"
        return "evening_leisure"
    elif 18 <= hour < 22:
        return "evening_leisure"
    else:
        if rng.random() < 0.70:
            return "evening_leisure"
        return "late_night"
