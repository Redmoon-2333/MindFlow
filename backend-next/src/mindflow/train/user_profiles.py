"""Student archetype profiles and procrastination episode definitions.

Defines 30 student archetypes (5 grades × 6 majors) and 6 procrastination
episode types as frozen dataclasses. Used by the synthetic data generator
to produce realistic, persona-driven activity logs.

Each archetype captures:
  - Grade-level schedule parameters (wake/sleep, rigidity, weekend delay)
  - Major-specific app ecosystems per time-of-day pattern
  - Procrastination tendencies (probability, episode preferences)
  - Exam-mode behavior modifiers
  - Expected QA statistics for validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StudentArchetype:
    """Complete behavioral model of a student persona.

    Captures schedule, app ecosystem, procrastination profile, and
    expected aggregate statistics for a specific grade × major combination.
    """

    profile_id: str
    grade: str   # Chinese label: "大一", "大二", ...
    major: str   # Chinese label: "计算机/软件", ...

    # ── Schedule ──
    typical_wake_hour: int       # hour of day (0-23)
    typical_sleep_hour: int      # hour of day (0-23, 0 = midnight)
    weekend_delay_hours: int     # how much later weekend wake/sleep shifts
    schedule_rigidity: float     # 0-1, how rigid the daily schedule is

    # ── App ecosystem by time pattern ──
    # Keys: early_morning, morning_focus, afternoon_mixed,
    #       evening_leisure, late_night
    primary_apps: dict[str, list[str]]
    primary_titles: dict[str, list[str]]
    primary_weights: dict[str, list[float]]

    # ── Procrastination ──
    daily_proc_probability: float    # 0-1 base chance of procrastinating on a given day
    episode_type_weights: dict[str, float]  # episode name → relative weight
    weekend_multiplier: float        # × proc_prob on weekends

    # ── Exam mode modifiers ──
    exam_productivity_bump: float       # 0-1 increase in focus during exam period
    exam_procrastination_change: float

    # ── Expected QA stats (approximate) ──
    expected_focus_score_mean: float
    expected_idle_ratio_mean: float
    expected_switch_frequency_mean: float   # switches per hour
    expected_entertainment_ratio_mean: float


@dataclass(frozen=True)
class ProcrastinationEpisode:
    """A distinct type of procrastination session.

    Defines the apps used, duration range, time-of-day window, day
    preference (weekday/weekend/any), and expected behavioral metrics
    during the episode.
    """

    name: str
    apps: list[str]
    titles: list[str]
    weights: list[float]
    min_duration_hours: float
    max_duration_hours: float
    earliest_hour: int      # earliest start hour (0-23)
    latest_hour: int        # latest possible start hour (0-23)
    day_bias: str           # "any", "weekday_only", "weekend_only"
    switch_frequency: float  # expected app switches per hour during episode
    idle_ratio: float        # expected idle proportion during episode


# ═══════════════════════════════════════════════════════════════════════════
# Episode Type Definitions (6 types)
# ═══════════════════════════════════════════════════════════════════════════

EPISODES: dict[str, ProcrastinationEpisode] = {
    "binge_watching": ProcrastinationEpisode(
        name="binge_watching",
        apps=["bilibili", "youtube", "iqiyi", "netflix", "tencent_video"],
        titles=[
            "B站 - 追番中",
            "YouTube - Recommended",
            "爱奇艺 - 热播剧",
            "Netflix - Continue Watching",
            "腾讯视频 - 综艺",
        ],
        weights=[0.35, 0.20, 0.20, 0.10, 0.15],
        min_duration_hours=1.5,
        max_duration_hours=5.0,
        earliest_hour=19,
        latest_hour=0,  # midnight
        day_bias="any",
        switch_frequency=1.5,
        idle_ratio=0.02,
    ),
    "doom_scrolling": ProcrastinationEpisode(
        name="doom_scrolling",
        apps=["weibo", "zhihu", "douyin", "xiaohongshu", "chrome"],
        titles=[
            "微博 - 热搜",
            "知乎 - 推荐",
            "抖音 - For You",
            "小红书 - 发现",
            "Chrome - 摸鱼中",
        ],
        weights=[0.30, 0.25, 0.25, 0.15, 0.05],
        min_duration_hours=0.5,
        max_duration_hours=3.0,
        earliest_hour=10,
        latest_hour=2,  # 2am
        day_bias="any",
        switch_frequency=12.0,
        idle_ratio=0.01,
    ),
    "gaming_session": ProcrastinationEpisode(
        name="gaming_session",
        apps=["steam", "lol_client", "genshin_impact", "valorant", "epic_games"],
        titles=[
            "Steam - Library",
            "League of Legends",
            "原神",
            "VALORANT",
            "Epic Games Launcher",
        ],
        weights=[0.25, 0.25, 0.20, 0.15, 0.15],
        min_duration_hours=1.0,
        max_duration_hours=6.0,
        earliest_hour=18,
        latest_hour=1,
        day_bias="weekend_only",
        switch_frequency=2.0,
        idle_ratio=0.03,
    ),
    "social_media_spiral": ProcrastinationEpisode(
        name="social_media_spiral",
        apps=["wechat", "weibo", "douyin", "xiaohongshu", "qq"],
        titles=[
            "微信 - 朋友圈",
            "微博 - 超话",
            "抖音 - Live",
            "小红书 - 笔记",
            "QQ - 群聊",
        ],
        weights=[0.35, 0.20, 0.20, 0.15, 0.10],
        min_duration_hours=0.5,
        max_duration_hours=2.5,
        earliest_hour=8,
        latest_hour=1,
        day_bias="any",
        switch_frequency=10.0,
        idle_ratio=0.02,
    ),
    "inspiration_browsing": ProcrastinationEpisode(
        name="inspiration_browsing",
        apps=["pinterest", "behance", "dribbble", "instagram", "bilibili"],
        titles=[
            "Pinterest - Home Feed",
            "Behance - Discover",
            "Dribbble - Shots",
            "Instagram - Explore",
            "B站 - 设计区",
        ],
        weights=[0.25, 0.25, 0.20, 0.15, 0.15],
        min_duration_hours=1.0,
        max_duration_hours=4.0,
        earliest_hour=14,
        latest_hour=2,
        day_bias="any",
        switch_frequency=8.0,
        idle_ratio=0.04,
    ),
    "crash_and_burn": ProcrastinationEpisode(
        name="crash_and_burn",
        apps=["bilibili", "douyin", "weibo", "steam", "wechat"],
        titles=[
            "B站 - 随便看看",
            "抖音 - 停不下来",
            "微博 - 吃瓜",
            "Steam - 新游戏",
            "微信 - 聊天",
        ],
        weights=[0.25, 0.25, 0.20, 0.15, 0.15],
        min_duration_hours=3.0,
        max_duration_hours=8.0,
        earliest_hour=14,
        latest_hour=0,
        day_bias="weekend_only",
        switch_frequency=4.0,
        idle_ratio=0.06,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Shared app/title templates per major
# ═══════════════════════════════════════════════════════════════════════════

def _cs_apps() -> dict[str, dict[str, list[Any]]]:
    """CS major: IDE-heavy, GitHub, LeetCode with gaming at night."""
    return {
        "early_morning": {
            "apps": ["chrome", "wechat", "notion", "calendar", "mail"],
            "titles": [
                "Google - Chrome",
                "微信 - 消息",
                "今日计划 - Notion",
                "Calendar",
                "Gmail",
            ],
            "weights": [0.30, 0.20, 0.20, 0.15, 0.15],
        },
        "morning_focus": {
            "apps": ["vscode", "pycharm", "terminal", "github_desktop", "leetcode"],
            "titles": [
                "main.py - VSCode",
                "project - PyCharm",
                "Terminal - zsh",
                "GitHub Desktop",
                "LeetCode - 刷题",
            ],
            "weights": [0.30, 0.22, 0.20, 0.15, 0.13],
        },
        "afternoon_mixed": {
            "apps": ["vscode", "chrome_stackoverflow", "wechat", "jupyter", "docker"],
            "titles": [
                "app.tsx - VSCode",
                "Stack Overflow - Chrome",
                "微信 - 群聊",
                "analysis.ipynb - Jupyter",
                "Docker Desktop",
            ],
            "weights": [0.28, 0.22, 0.20, 0.18, 0.12],
        },
        "evening_leisure": {
            "apps": ["bilibili", "steam", "wechat", "chrome_reddit", "youtube"],
            "titles": [
                "B站 - 科技区",
                "Steam",
                "微信 - 朋友圈",
                "Reddit - r/programming",
                "YouTube - Tech",
            ],
            "weights": [0.28, 0.22, 0.20, 0.18, 0.12],
        },
        "late_night": {
            "apps": ["bilibili", "douyin", "steam", "wechat", "chrome_github"],
            "titles": [
                "B站 - 深夜档",
                "抖音 - For You",
                "Steam - Gaming",
                "微信",
                "GitHub - Trending",
            ],
            "weights": [0.30, 0.22, 0.20, 0.18, 0.10],
        },
    }


def _ee_apps() -> dict[str, dict[str, list[Any]]]:
    """EE major: MATLAB, Keil, Altium, CAD, simulation-heavy."""
    return {
        "early_morning": {
            "apps": ["chrome", "wechat", "calendar", "mail", "notion"],
            "titles": [
                "Google - Chrome",
                "微信 - 消息",
                "Calendar",
                "Gmail",
                "课程表 - Notion",
            ],
            "weights": [0.28, 0.22, 0.20, 0.18, 0.12],
        },
        "morning_focus": {
            "apps": ["matlab", "vscode", "keil", "multisim", "terminal"],
            "titles": [
                "MATLAB - simulation.m",
                "firmware.c - VS Code",
                "Keil uVision5",
                "Multisim - Circuit",
                "Terminal",
            ],
            "weights": [0.28, 0.20, 0.20, 0.18, 0.14],
        },
        "afternoon_mixed": {
            "apps": ["altium", "matlab", "wechat", "chrome", "cad"],
            "titles": [
                "Altium Designer - PCB",
                "MATLAB - Script",
                "微信 - 实验室群",
                "数据手册 - Chrome",
                "AutoCAD - Drawing",
            ],
            "weights": [0.25, 0.22, 0.20, 0.18, 0.15],
        },
        "evening_leisure": {
            "apps": ["bilibili", "douyin", "steam", "wechat", "youtube"],
            "titles": [
                "B站 - 电子DIY",
                "抖音 - For You",
                "Steam",
                "微信 - 朋友圈",
                "YouTube - EEVblog",
            ],
            "weights": [0.28, 0.22, 0.20, 0.18, 0.12],
        },
        "late_night": {
            "apps": ["bilibili", "douyin", "steam", "wechat", "chrome"],
            "titles": [
                "B站 - 番剧",
                "抖音",
                "Steam",
                "微信 - 聊天",
                "Chrome",
            ],
            "weights": [0.30, 0.22, 0.20, 0.18, 0.10],
        },
    }


def _liberal_arts_apps() -> dict[str, dict[str, list[Any]]]:
    """Liberal Arts: Word, Zotero, CNKI, DeepL, reading-heavy."""
    return {
        "early_morning": {
            "apps": ["wechat", "chrome", "calendar", "reading_app", "mail"],
            "titles": [
                "微信 - 消息",
                "Google - Chrome",
                "Calendar",
                "微信读书",
                "Gmail",
            ],
            "weights": [0.30, 0.22, 0.18, 0.18, 0.12],
        },
        "morning_focus": {
            "apps": ["word", "zotero", "deepl", "chrome_cnki", "notion"],
            "titles": [
                "论文.docx - Word",
                "Zotero - References",
                "DeepL - 翻译",
                "知网 - Chrome",
                "大纲 - Notion",
            ],
            "weights": [0.30, 0.22, 0.18, 0.18, 0.12],
        },
        "afternoon_mixed": {
            "apps": ["word", "chrome_cnki", "wechat", "zotero", "wps"],
            "titles": [
                "读书笔记.docx - Word",
                "知网 - 文献检索",
                "微信 - 读书群",
                "Zotero",
                "WPS Office",
            ],
            "weights": [0.28, 0.22, 0.20, 0.18, 0.12],
        },
        "evening_leisure": {
            "apps": ["wechat", "weibo", "bilibili", "zhihu", "chrome"],
            "titles": [
                "微信 - 朋友圈",
                "微博 - 热搜",
                "B站 - 纪录片",
                "知乎 - 推荐",
                "Chrome - 闲逛",
            ],
            "weights": [0.26, 0.22, 0.20, 0.18, 0.14],
        },
        "late_night": {
            "apps": ["weibo", "zhihu", "bilibili", "wechat", "reading_app"],
            "titles": [
                "微博 - 深夜冲浪",
                "知乎 - 睡前阅读",
                "B站",
                "微信 - 群聊",
                "微信读书 - 小说",
            ],
            "weights": [0.28, 0.24, 0.20, 0.16, 0.12],
        },
    }


def _business_apps() -> dict[str, dict[str, list[Any]]]:
    """Business: Excel, PowerPoint, Wind, Stata, case-platforms."""
    return {
        "early_morning": {
            "apps": ["wechat", "chrome", "calendar", "mail", "notion"],
            "titles": [
                "微信 - 消息",
                "Bloomberg - Chrome",
                "Calendar",
                "Outlook",
                "日程 - Notion",
            ],
            "weights": [0.30, 0.20, 0.18, 0.18, 0.14],
        },
        "morning_focus": {
            "apps": ["excel", "stata", "wind", "chrome", "powerpoint"],
            "titles": [
                "数据分析.xlsx - Excel",
                "Stata - regression.do",
                "Wind金融终端",
                "研究报告 - Chrome",
                "演示文稿.pptx - PowerPoint",
            ],
            "weights": [0.30, 0.20, 0.18, 0.18, 0.14],
        },
        "afternoon_mixed": {
            "apps": ["excel", "powerpoint", "wechat", "chrome", "feishu"],
            "titles": [
                "财务报表.xlsx - Excel",
                "pre.pptx - PowerPoint",
                "微信 - 实习群",
                "案例平台 - Chrome",
                "飞书 - 协作",
            ],
            "weights": [0.26, 0.22, 0.20, 0.18, 0.14],
        },
        "evening_leisure": {
            "apps": ["douyin", "wechat", "chrome", "bilibili", "tencent_meeting"],
            "titles": [
                "抖音 - For You",
                "微信 - 朋友圈",
                "财经新闻 - Chrome",
                "B站 - 财经区",
                "腾讯会议 - 社团",
            ],
            "weights": [0.26, 0.24, 0.20, 0.16, 0.14],
        },
        "late_night": {
            "apps": ["douyin", "wechat", "chrome_finance", "bilibili", "weibo"],
            "titles": [
                "抖音",
                "微信 - 聊天",
                "雪球 - Chrome",
                "B站",
                "微博 - 热搜",
            ],
            "weights": [0.28, 0.24, 0.18, 0.18, 0.12],
        },
    }


def _design_apps() -> dict[str, dict[str, list[Any]]]:
    """Design: Photoshop, Figma, Blender, Behance, Pinterest, irregular bursts."""
    return {
        "early_morning": {
            "apps": ["chrome", "wechat", "calendar", "notion", "mail"],
            "titles": [
                "Inspiration - Chrome",
                "微信 - 消息",
                "Calendar",
                "灵感 - Notion",
                "Gmail",
            ],
            "weights": [0.28, 0.22, 0.20, 0.18, 0.12],
        },
        "morning_focus": {
            "apps": ["photoshop", "figma", "illustrator", "blender", "chrome"],
            "titles": [
                "poster.psd - Photoshop",
                "app_ui - Figma",
                "logo.ai - Illustrator",
                "scene.blend - Blender",
                "Tutorial - Chrome",
            ],
            "weights": [0.26, 0.24, 0.18, 0.16, 0.16],
        },
        "afternoon_mixed": {
            "apps": ["figma", "photoshop", "wechat", "chrome", "after_effects"],
            "titles": [
                "design_v2 - Figma",
                "retouch.psd - Photoshop",
                "微信 - 设计群",
                "Dribbble - Chrome",
                "motion.aep - After Effects",
            ],
            "weights": [0.28, 0.22, 0.18, 0.18, 0.14],
        },
        "evening_leisure": {
            "apps": ["behance", "pinterest", "instagram", "bilibili", "wechat"],
            "titles": [
                "Behance - Discover",
                "Pinterest - Home",
                "Instagram - Explore",
                "B站 - 设计教程",
                "微信 - 朋友圈",
            ],
            "weights": [0.26, 0.24, 0.20, 0.16, 0.14],
        },
        "late_night": {
            "apps": ["behance", "pinterest", "bilibili", "instagram", "wechat"],
            "titles": [
                "Behance - 深夜灵感",
                "Pinterest - Mood Board",
                "B站 - 绘画过程",
                "Instagram - Reels",
                "微信",
            ],
            "weights": [0.28, 0.24, 0.20, 0.16, 0.12],
        },
    }


def _medical_apps() -> dict[str, dict[str, list[Any]]]:
    """Medical: Anki, PubMed, SPSS, EndNote, lab apps, high discipline."""
    return {
        "early_morning": {
            "apps": ["anki", "chrome", "wechat", "calendar", "mail"],
            "titles": [
                "Anki - 每日复习",
                "PubMed - Chrome",
                "微信 - 消息",
                "Calendar",
                "Gmail",
            ],
            "weights": [0.35, 0.20, 0.18, 0.15, 0.12],
        },
        "morning_focus": {
            "apps": ["anki", "pubmed", "uptodate", "chrome", "notion"],
            "titles": [
                "Anki - 系统复习",
                "PubMed - Literature",
                "UpToDate - Clinical",
                "课本PDF - Chrome",
                "学习计划 - Notion",
            ],
            "weights": [0.32, 0.22, 0.18, 0.16, 0.12],
        },
        "afternoon_mixed": {
            "apps": ["spss", "endnote", "anki", "wechat", "lab_app"],
            "titles": [
                "SPSS - data.sav",
                "EndNote - Library",
                "Anki - 刷卡片",
                "微信 - 课题组",
                "Lab - Protocol",
            ],
            "weights": [0.22, 0.20, 0.22, 0.18, 0.18],
        },
        "evening_leisure": {
            "apps": ["bilibili", "wechat", "chrome", "reading_app", "anki"],
            "titles": [
                "B站 - 医学纪录片",
                "微信 - 朋友圈",
                "医学论坛 - Chrome",
                "微信读书",
                "Anki - 睡前复习",
            ],
            "weights": [0.24, 0.22, 0.18, 0.18, 0.18],
        },
        "late_night": {
            "apps": ["anki", "bilibili", "wechat", "chrome", "reading_app"],
            "titles": [
                "Anki - 熬夜背书",
                "B站",
                "微信 - 聊天",
                "丁香园 - Chrome",
                "微信读书",
            ],
            "weights": [0.30, 0.22, 0.20, 0.16, 0.12],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Grade-level schedule parameters
# ═══════════════════════════════════════════════════════════════════════════

_GRADE_PARAMS: dict[str, dict[str, Any]] = {
    "freshman": {
        "grade_label": "大一",
        "wake": 7,
        "sleep": 23,
        "rigidity": 0.85,
        "weekend_delay": 2,
        "daily_proc_base": 0.22,
        "weekend_multiplier": 1.6,
        "exam_bump": 0.15,
        "exam_proc_change": -0.15,
        "focus_bonus": 0.08,
        "idle_bonus": -0.02,
        "switch_bonus": -1.0,
        "entertainment_bonus": -0.05,
    },
    "sophomore": {
        "grade_label": "大二",
        "wake": 8,
        "sleep": 0,  # midnight
        "rigidity": 0.70,
        "weekend_delay": 2,
        "daily_proc_base": 0.28,
        "weekend_multiplier": 1.5,
        "exam_bump": 0.15,
        "exam_proc_change": -0.10,
        "focus_bonus": 0.03,
        "idle_bonus": -0.01,
        "switch_bonus": -0.5,
        "entertainment_bonus": 0.00,
    },
    "junior": {
        "grade_label": "大三",
        "wake": 8,
        "sleep": 1,
        "rigidity": 0.50,
        "weekend_delay": 3,
        "daily_proc_base": 0.25,
        "weekend_multiplier": 1.4,
        "exam_bump": 0.18,
        "exam_proc_change": -0.12,
        "focus_bonus": 0.05,
        "idle_bonus": 0.00,
        "switch_bonus": 0.0,
        "entertainment_bonus": -0.02,
    },
    "senior": {
        "grade_label": "大四",
        "wake": 10,
        "sleep": 2,
        "rigidity": 0.30,
        "weekend_delay": 3,
        "daily_proc_base": 0.38,
        "weekend_multiplier": 1.3,
        "exam_bump": 0.10,
        "exam_proc_change": 0.05,
        "focus_bonus": -0.05,
        "idle_bonus": 0.03,
        "switch_bonus": 1.0,
        "entertainment_bonus": 0.08,
    },
    "grad": {
        "grade_label": "研一/研二",
        "wake": 9,
        "sleep": 2,
        "rigidity": 0.25,
        "weekend_delay": 1,
        "daily_proc_base": 0.18,
        "weekend_multiplier": 1.3,
        "exam_bump": 0.20,
        "exam_proc_change": -0.15,
        "focus_bonus": 0.10,
        "idle_bonus": -0.01,
        "switch_bonus": -1.5,
        "entertainment_bonus": -0.10,
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# Major-level base parameters
# ═══════════════════════════════════════════════════════════════════════════

_MAJOR_PARAMS: dict[str, dict[str, Any]] = {
    "cs": {
        "major_label": "计算机/软件",
        "apps_fn": _cs_apps,
        "focus_base": 0.55,
        "idle_base": 0.08,
        "switch_base": 5.0,
        "entertainment_base": 0.30,
        "episode_weights": {
            "gaming_session": 0.35,
            "binge_watching": 0.25,
            "doom_scrolling": 0.20,
            "social_media_spiral": 0.15,
            "inspiration_browsing": 0.05,
            "crash_and_burn": 0.00,
        },
    },
    "ee": {
        "major_label": "电子信息/自动化",
        "apps_fn": _ee_apps,
        "focus_base": 0.50,
        "idle_base": 0.09,
        "switch_base": 4.5,
        "entertainment_base": 0.32,
        "episode_weights": {
            "gaming_session": 0.30,
            "binge_watching": 0.25,
            "doom_scrolling": 0.20,
            "social_media_spiral": 0.15,
            "inspiration_browsing": 0.05,
            "crash_and_burn": 0.05,
        },
    },
    "liberal_arts": {
        "major_label": "人文/社科",
        "apps_fn": _liberal_arts_apps,
        "focus_base": 0.40,
        "idle_base": 0.15,
        "switch_base": 6.0,
        "entertainment_base": 0.40,
        "episode_weights": {
            "doom_scrolling": 0.40,
            "social_media_spiral": 0.30,
            "binge_watching": 0.20,
            "inspiration_browsing": 0.05,
            "gaming_session": 0.03,
            "crash_and_burn": 0.02,
        },
    },
    "business": {
        "major_label": "经管/商科",
        "apps_fn": _business_apps,
        "focus_base": 0.45,
        "idle_base": 0.12,
        "switch_base": 5.5,
        "entertainment_base": 0.35,
        "episode_weights": {
            "social_media_spiral": 0.30,
            "doom_scrolling": 0.25,
            "binge_watching": 0.20,
            "gaming_session": 0.10,
            "inspiration_browsing": 0.10,
            "crash_and_burn": 0.05,
        },
    },
    "design": {
        "major_label": "设计/艺术",
        "apps_fn": _design_apps,
        "focus_base": 0.38,
        "idle_base": 0.18,
        "switch_base": 7.0,
        "entertainment_base": 0.50,
        "episode_weights": {
            "inspiration_browsing": 0.35,
            "social_media_spiral": 0.25,
            "binge_watching": 0.15,
            "doom_scrolling": 0.15,
            "gaming_session": 0.05,
            "crash_and_burn": 0.05,
        },
    },
    "medical": {
        "major_label": "医学/药学",
        "apps_fn": _medical_apps,
        "focus_base": 0.65,
        "idle_base": 0.06,
        "switch_base": 3.5,
        "entertainment_base": 0.20,
        "episode_weights": {
            "crash_and_burn": 0.40,
            "binge_watching": 0.25,
            "doom_scrolling": 0.15,
            "social_media_spiral": 0.15,
            "gaming_session": 0.03,
            "inspiration_browsing": 0.02,
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Builder function
# ═══════════════════════════════════════════════════════════════════════════

def _build_archetype(grade_key: str, major_key: str) -> StudentArchetype:
    """Build a single archetype from grade + major template parameters."""
    gp = _GRADE_PARAMS[grade_key]
    mp = _MAJOR_PARAMS[major_key]
    apps_data = mp["apps_fn"]()

    def _clamp(value: float, lo: float = 0.05, hi: float = 0.95) -> float:
        return max(lo, min(hi, value))

    return StudentArchetype(
        profile_id=f"{grade_key}_{major_key}",
        grade=gp["grade_label"],
        major=mp["major_label"],
        typical_wake_hour=gp["wake"],
        typical_sleep_hour=gp["sleep"],
        weekend_delay_hours=gp["weekend_delay"],
        schedule_rigidity=gp["rigidity"],
        primary_apps={k: v["apps"] for k, v in apps_data.items()},
        primary_titles={k: v["titles"] for k, v in apps_data.items()},
        primary_weights={k: v["weights"] for k, v in apps_data.items()},
        daily_proc_probability=_clamp(
            gp["daily_proc_base"], lo=0.05, hi=0.80
        ),
        episode_type_weights=mp["episode_weights"],
        weekend_multiplier=gp["weekend_multiplier"],
        exam_productivity_bump=gp["exam_bump"],
        exam_procrastination_change=gp["exam_proc_change"],
        expected_focus_score_mean=_clamp(
            mp["focus_base"] + gp["focus_bonus"]
        ),
        expected_idle_ratio_mean=_clamp(
            mp["idle_base"] + gp["idle_bonus"], lo=0.02, hi=0.60
        ),
        expected_switch_frequency_mean=max(
            1.5,
            mp["switch_base"] + gp["switch_bonus"],
        ),
        expected_entertainment_ratio_mean=_clamp(
            mp["entertainment_base"] + gp["entertainment_bonus"],
            lo=0.05,
            hi=0.80,
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# All 30 Archetypes (5 grades × 6 majors)
# ═══════════════════════════════════════════════════════════════════════════

_GRADES = ["freshman", "sophomore", "junior", "senior", "grad"]
_MAJORS = ["cs", "ee", "liberal_arts", "business", "design", "medical"]

PROFILES: dict[str, StudentArchetype] = {
    f"{g}_{m}": _build_archetype(g, m)
    for g in _GRADES
    for m in _MAJORS
}


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def get_archetype(profile_id: str) -> StudentArchetype:
    """Look up a student archetype by profile ID (e.g. ``"junior_cs"``).

    Raises:
        KeyError: If ``profile_id`` does not match any known archetype.
    """
    return PROFILES[profile_id]


def get_episode(name: str) -> ProcrastinationEpisode:
    """Look up a procrastination episode type by name (e.g. ``"binge_watching"``).

    Raises:
        KeyError: If ``name`` does not match any known episode type.
    """
    return EPISODES[name]


def list_archetype_ids() -> list[str]:
    """Return all available archetype profile IDs, sorted alphabetically."""
    return sorted(PROFILES.keys())


def list_episode_names() -> list[str]:
    """Return all available episode type names, sorted alphabetically."""
    return sorted(EPISODES.keys())
