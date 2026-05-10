"""Title-based feature extraction — no app classification needed.

Replaces AppClassifier's hardcoded app→category mapping with objective
window title analysis. Works with any application, known or unknown.
"""

import re
from urllib.parse import urlparse


class TitleAnalyzer:
    """Extract objective features from window titles without app classification.

    Uses pattern matching on URL schemes, file extensions, and structural
    keywords — not a maintained list of application names.
    """

    # URL detection — matches browser URL bars and standalone URLs
    URL_PATTERN = re.compile(
        r"(?:https?://|www\.|[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}/)[^\s]*",
        re.IGNORECASE,
    )

    # File extensions that suggest different activity types
    CODE_EXTENSIONS = {
        ".py", ".js", ".ts", ".tsx", ".java", ".go", ".rs", ".cpp",
        ".c", ".h", ".rb", ".php", ".swift", ".kt", ".scala", ".r",
        ".ipynb", ".sql", ".sh", ".bash", ".yml", ".yaml", ".toml",
        ".json", ".xml", ".html", ".css", ".scss", ".vue", ".svelte",
    }
    DOC_EXTENSIONS = {
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".md", ".tex", ".txt", ".csv", ".rtf", ".odt",
    }

    # Keywords that suggest meeting/communication context
    MEETING_KEYWORDS = {
        "zoom", "meet", "teams", "meeting", "会议", "腾讯会议",
        "dingtalk", "飞书", "feishu", "slack", "discord",
    }

    # Title indicators of entertainment content
    ENTERTAINMENT_PATTERNS = [
        re.compile(p, re.IGNORECASE) for p in [
            r"番剧|动漫|anime|episode\s*\d+",
            r"第\d+集|第\d+话",
            r"直播间|live\s*room|直播",
            r"短视频|short\s*video",
            r"steam\s*(library|store|community)",
            r"游戏|game\s*(play|store|library)",
        ]
    ]

    def analyze(self, window_title: str) -> dict:
        """Extract title-based features. Returns dict with keys:

        - url_domain: str or None (browser URL domain)
        - is_browser: bool
        - is_code_editor: bool (file extension in CODE_EXTENSIONS)
        - is_document: bool (file extension in DOC_EXTENSIONS)
        - is_meeting: bool (meeting keyword detected)
        - is_likely_entertainment: bool (entertainment pattern match)
        - file_extension: str or None
        """
        title = str(window_title).strip() if window_title else ""
        result = {
            "url_domain": None,
            "is_browser": False,
            "is_code_editor": False,
            "is_document": False,
            "is_meeting": False,
            "is_likely_entertainment": False,
            "file_extension": None,
        }

        if not title:
            return result

        url_match = self.URL_PATTERN.search(title)
        if url_match:
            result["is_browser"] = True
            try:
                parsed = urlparse(
                    url_match.group()
                    if "://" in url_match.group()
                    else f"https://{url_match.group()}"
                )
                domain = parsed.netloc or parsed.path.split("/")[0]
                domain = domain.replace("www.", "").lower()
                result["url_domain"] = domain
            except Exception:
                pass

        for ext in self.CODE_EXTENSIONS:
            if f".{ext.lstrip('.')}" in title.lower() or ext in title.lower():
                result["file_extension"] = ext
                result["is_code_editor"] = True
                break

        if not result["is_code_editor"]:
            for ext in self.DOC_EXTENSIONS:
                ext_lower = ext.lower()
                if ext_lower in title.lower():
                    result["file_extension"] = ext
                    result["is_document"] = True
                    break

        title_lower = title.lower()
        if any(kw in title_lower for kw in self.MEETING_KEYWORDS):
            result["is_meeting"] = True

        if any(p.search(title) for p in self.ENTERTAINMENT_PATTERNS):
            result["is_likely_entertainment"] = True

        return result


class BehaviorOnlyLabeler:
    """Label focus/distraction from behavior and title features only.

    No app classification. Uses:
    - Switch frequency (objective: count of process changes)
    - Session duration (objective: time spent in one app)
    - Time of day (objective: hour)
    - Idle ratio (objective: mouse/keyboard inactivity)
    - Title features (objective: URL domains, file extensions)
    """

    def label(self, row: dict, title_features: dict) -> tuple[int, float]:
        """Returns (label, confidence) where label=1 means focus."""
        votes: list[tuple[int, float]] = []

        sf = float(row.get("switch_frequency", 30))
        unique = float(row.get("unique_app_count", 5))
        max_dur = float(row.get("max_app_duration", 300))
        idle = float(row.get("idle_ratio", 0))
        hour = int(row.get("hour_of_day", 12))

        # Signal 1: Stability (low switch + few apps + long max duration)
        if sf < 8 and unique <= 3 and max_dur > 600:
            votes.append((1, 0.9))
        elif sf < 15 and unique <= 4:
            votes.append((1, 0.65))
        elif sf > 40 or unique >= 8:
            votes.append((0, 0.7))
        elif sf > 25:
            votes.append((0, 0.5))
        else:
            votes.append((1, 0.2))  # weakly default to focus when ambiguous

        # Signal 2: Idle detection — if user is away, it's neither focus nor distraction
        if idle > 0.7:
            votes.append((1, 0.1))  # abstain — irrelevant
        elif idle > 0.3:
            votes.append((0, 0.3))
        else:
            votes.append((1, 0.4))

        # Signal 3: Time context
        if 9 <= hour <= 11:
            votes.append((1, 0.5))
        elif 14 <= hour <= 17:
            votes.append((1, 0.4))
        elif 0 <= hour <= 5:
            votes.append((0, 0.5))
        else:
            votes.append((1, 0.3))

        # Signal 4: Title-based — entertainment is a strong signal
        if title_features.get("is_likely_entertainment"):
            votes.append((0, 0.95))
        elif title_features.get("is_code_editor"):
            votes.append((1, 0.85))
        elif title_features.get("is_document"):
            votes.append((1, 0.75))
        elif title_features.get("is_meeting"):
            votes.append((1, 0.5))
        elif title_features.get("url_domain"):
            domain = title_features["url_domain"]
            if any(d in domain for d in ["github", "stackoverflow", "arxiv", "scholar"]):
                votes.append((1, 0.8))
            elif any(d in domain for d in ["bilibili", "youtube", "netflix", "douyin"]):
                votes.append((0, 0.8))
            else:
                votes.append((1, 0.3))
        else:
            votes.append((1, 0.25))

        focus_weight = sum(c for v, c in votes if v == 1)
        dist_weight = sum(c for v, c in votes if v == 0)
        total = focus_weight + dist_weight
        if total == 0:
            return 1, 0.0

        label = 1 if focus_weight >= dist_weight else 0
        majority = max(focus_weight, dist_weight)
        confidence = max(0.0, (majority / total - 0.5) * 2.0)
        return label, round(confidence, 4)
