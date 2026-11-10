"""Explicit software, profession and intention recipes for the synthetic base prior.

These are simulation fixtures, not production app-classification rules. A task
context describes the observed tool; it deliberately does not determine focus.
For example a game test and recreational gaming both observe entertainment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppSpec:
    key: str
    name: str
    process: str
    context: str
    domain: str = ""


def _web(key: str, name: str, domain: str, context: str = "reading") -> AppSpec:
    return AppSpec(key, name, "chrome.exe", context, domain)


_APPS = (
    AppSpec("word", "Microsoft Word", "winword.exe", "writing"),
    AppSpec("excel", "Microsoft Excel", "excel.exe", "writing"),
    AppSpec("powerpoint", "Microsoft PowerPoint", "powerpnt.exe", "writing"),
    AppSpec("onenote", "Microsoft OneNote", "onenote.exe", "writing"),
    AppSpec("acrobat", "Adobe Acrobat", "acrobat.exe", "reading"),
    AppSpec("zotero", "Zotero", "zotero.exe", "reading"),
    AppSpec("obsidian", "Obsidian", "obsidian.exe", "writing"),
    AppSpec("notepad", "Notepad++", "notepad++.exe", "writing"),
    AppSpec("vscode", "Visual Studio Code", "code.exe", "coding"),
    AppSpec("pycharm", "PyCharm", "pycharm64.exe", "coding"),
    AppSpec("idea", "IntelliJ IDEA", "idea64.exe", "coding"),
    AppSpec("terminal", "Windows Terminal", "windowsterminal.exe", "coding"),
    AppSpec("docker", "Docker Desktop", "docker desktop.exe", "coding"),
    AppSpec("postman", "Postman", "postman.exe", "coding"),
    AppSpec("dbeaver", "DBeaver", "dbeaver.exe", "coding"),
    AppSpec("pgadmin", "pgAdmin", "pgadmin4.exe", "coding"),
    AppSpec("rstudio", "RStudio", "rstudio.exe", "coding"),
    AppSpec("matlab", "MATLAB", "matlab.exe", "coding"),
    AppSpec("stata", "Stata", "stata.exe", "coding"),
    AppSpec("spss", "IBM SPSS Statistics", "stats.exe", "coding"),
    AppSpec("powerbi", "Microsoft Power BI", "pbidesktop.exe", "writing"),
    AppSpec("tableau", "Tableau Desktop", "tableau.exe", "writing"),
    AppSpec("quickbooks", "QuickBooks Desktop", "qbw32.exe", "writing"),
    AppSpec("sage", "Sage 50", "peachw.exe", "writing"),
    AppSpec("photoshop", "Adobe Photoshop", "photoshop.exe", "writing"),
    AppSpec("illustrator", "Adobe Illustrator", "illustrator.exe", "writing"),
    AppSpec("figma", "Figma", "figma.exe", "writing"),
    AppSpec("affinity", "Affinity Designer", "designer.exe", "writing"),
    AppSpec("blender", "Blender", "blender.exe", "writing"),
    AppSpec("autocad", "AutoCAD", "acad.exe", "writing"),
    AppSpec("solidworks", "SOLIDWORKS", "sldworks.exe", "writing"),
    AppSpec("fusion", "Autodesk Fusion", "fusion360.exe", "writing"),
    AppSpec("revit", "Autodesk Revit", "revit.exe", "writing"),
    AppSpec("freecad", "FreeCAD", "freecad.exe", "writing"),
    AppSpec("premiere", "Adobe Premiere Pro", "adobe premiere pro.exe", "writing"),
    AppSpec("resolve", "DaVinci Resolve", "resolve.exe", "writing"),
    AppSpec("aftereffects", "Adobe After Effects", "afterfx.exe", "writing"),
    AppSpec("obs", "OBS Studio", "obs64.exe", "writing"),
    AppSpec("vlc", "VLC media player", "vlc.exe", "entertainment"),
    AppSpec("audition", "Adobe Audition", "adobe audition.exe", "writing"),
    AppSpec("audacity", "Audacity", "audacity.exe", "writing"),
    AppSpec("ableton", "Ableton Live", "ableton live.exe", "writing"),
    AppSpec("reaper", "REAPER", "reaper.exe", "writing"),
    AppSpec("flstudio", "FL Studio", "fl64.exe", "writing"),
    AppSpec("musescore", "MuseScore", "musescore4.exe", "writing"),
    AppSpec("spotify", "Spotify", "spotify.exe", "entertainment"),
    AppSpec("zoom", "Zoom", "zoom.exe", "meeting"),
    AppSpec("teams", "Microsoft Teams", "ms-teams.exe", "meeting"),
    AppSpec("slack", "Slack", "slack.exe", "social"),
    AppSpec("discord", "Discord", "discord.exe", "social"),
    AppSpec("wechat", "WeChat", "wechat.exe", "social"),
    AppSpec("outlook", "Microsoft Outlook", "outlook.exe", "meeting"),
    AppSpec("thunderbird", "Mozilla Thunderbird", "thunderbird.exe", "meeting"),
    AppSpec("scrivener", "Scrivener", "scrivener.exe", "writing"),
    AppSpec("typora", "Typora", "typora.exe", "writing"),
    AppSpec("antconc", "AntConc", "antconc.exe", "reading"),
    AppSpec("trados", "Trados Studio", "sdltradosstudio.exe", "writing"),
    AppSpec("memoq", "memoQ", "memoq.exe", "writing"),
    AppSpec("unity", "Unity Editor", "unity.exe", "coding"),
    AppSpec("unreal", "Unreal Editor", "unrealeditor.exe", "coding"),
    AppSpec("steam", "Steam", "steam.exe", "entertainment"),
    AppSpec("minecraft", "Minecraft", "javaw.exe", "entertainment"),
    AppSpec("godot", "Godot Engine", "godot.exe", "coding"),
    AppSpec("wireshark", "Wireshark", "wireshark.exe", "coding"),
    _web("arxiv", "arXiv", "arxiv.org"),
    _web("scholar", "Google Scholar", "scholar.google.com"),
    _web("pubmed", "PubMed", "pubmed.ncbi.nlm.nih.gov"),
    _web("cnki", "CNKI", "cnki.net"),
    _web("overleaf", "Overleaf", "overleaf.com", "writing"),
    _web("moodle", "Moodle", "moodle.org"),
    _web("coursera", "Coursera", "coursera.org"),
    _web("edx", "edX", "edx.org"),
    _web("github", "GitHub", "github.com", "coding"),
    _web("gitlab", "GitLab", "gitlab.com", "coding"),
    _web("stackoverflow", "Stack Overflow", "stackoverflow.com", "coding"),
    _web("python_docs", "Python Documentation", "docs.python.org", "coding"),
    _web("grafana", "Grafana Cloud", "grafana.com", "coding"),
    _web("datadog", "Datadog", "app.datadoghq.com", "coding"),
    _web("aws", "AWS Console", "console.aws.amazon.com", "coding"),
    _web("azure", "Azure Portal", "portal.azure.com", "coding"),
    _web("sheets", "Google Sheets", "sheets.google.com", "writing"),
    _web("docs", "Google Docs", "docs.google.com", "writing"),
    _web("notion", "Notion", "notion.so", "writing"),
    _web("jira", "Jira", "atlassian.net", "writing"),
    _web("trello", "Trello", "trello.com", "writing"),
    _web("asana", "Asana", "app.asana.com", "writing"),
    _web("salesforce", "Salesforce", "salesforce.com", "meeting"),
    _web("zendesk", "Zendesk", "zendesk.com", "meeting"),
    _web("intercom", "Intercom", "app.intercom.com", "meeting"),
    _web("gmail", "Gmail", "mail.google.com", "meeting"),
    _web("calendar", "Google Calendar", "calendar.google.com", "meeting"),
    _web("canva", "Canva", "canva.com", "writing"),
    _web("wikipedia", "Wikipedia", "wikipedia.org"),
    _web("deepl", "DeepL", "deepl.com", "writing"),
    _web("linguee", "Linguee", "linguee.com"),
    _web("westlaw", "Westlaw", "westlaw.com"),
    _web("lexis", "LexisNexis", "lexisnexis.com"),
    _web("youtube", "YouTube", "youtube.com", "entertainment"),
    _web("bilibili", "Bilibili", "bilibili.com", "entertainment"),
    _web("netflix", "Netflix", "netflix.com", "entertainment"),
    _web("reddit", "Reddit", "reddit.com", "social"),
    _web("weibo", "Weibo", "weibo.com", "social"),
    _web("x", "X", "x.com", "social"),
    _web("instagram", "Instagram", "instagram.com", "social"),
    AppSpec("unknown", "Unclassified local application", "local-tool.exe", "unknown"),
)
APP_CATALOG: dict[str, AppSpec] = {app.key: app for app in _APPS}


@dataclass(frozen=True)
class RoleSpec:
    key: str
    title: str
    apps: tuple[str, ...]
    specialty: str


ROLES: tuple[RoleSpec, ...] = (
    RoleSpec("student", "学生", ("word", "onenote", "moodle", "coursera", "edx"), "lecture"),
    RoleSpec(
        "researcher",
        "科研人员",
        ("zotero", "arxiv", "scholar", "cnki", "overleaf"),
        "reference_reading",
    ),
    RoleSpec(
        "developer",
        "软件开发",
        (
            "vscode",
            "pycharm",
            "idea",
            "github",
            "gitlab",
            "postman",
            "python_docs",
            "stackoverflow",
        ),
        "deep_work",
    ),
    RoleSpec(
        "devops",
        "开发运维",
        ("terminal", "docker", "grafana", "datadog", "aws", "azure", "wireshark"),
        "incident_response",
    ),
    RoleSpec(
        "finance",
        "财务审计",
        ("excel", "quickbooks", "sage", "sheets", "acrobat"),
        "spreadsheet_audit",
    ),
    RoleSpec(
        "data_analyst",
        "数据分析",
        ("rstudio", "matlab", "stata", "spss", "dbeaver", "pgadmin", "powerbi", "tableau"),
        "data_analysis",
    ),
    RoleSpec(
        "designer",
        "视觉设计",
        ("photoshop", "illustrator", "figma", "affinity", "canva"),
        "design_iteration",
    ),
    RoleSpec(
        "cad_engineer",
        "CAD工程师",
        ("autocad", "solidworks", "fusion", "revit", "freecad", "blender"),
        "design_iteration",
    ),
    RoleSpec(
        "video_editor",
        "视频剪辑",
        ("premiere", "resolve", "aftereffects", "obs", "vlc"),
        "video_review",
    ),
    RoleSpec(
        "music_producer",
        "音乐制作",
        ("audition", "audacity", "ableton", "reaper", "flstudio", "musescore", "spotify"),
        "audio_review",
    ),
    RoleSpec(
        "teacher",
        "教师",
        ("powerpoint", "zoom", "teams", "moodle", "canva", "onenote"),
        "lesson_preparation",
    ),
    RoleSpec(
        "writer", "写作编辑", ("scrivener", "typora", "obsidian", "docs", "notepad"), "writing"
    ),
    RoleSpec(
        "manager",
        "项目管理",
        ("jira", "trello", "asana", "notion", "slack", "outlook"),
        "collaboration",
    ),
    RoleSpec(
        "support_shift",
        "倒班客服",
        ("zendesk", "salesforce", "intercom", "teams", "wechat"),
        "customer_support",
    ),
    RoleSpec(
        "medical_research",
        "医学研究",
        ("pubmed", "spss", "zotero", "acrobat", "excel"),
        "reference_reading",
    ),
    RoleSpec("legal", "法务", ("westlaw", "lexis", "word", "acrobat", "thunderbird"), "writing"),
    RoleSpec(
        "game_qa",
        "游戏测试",
        ("unity", "unreal", "godot", "steam", "minecraft", "discord"),
        "game_testing",
    ),
    RoleSpec(
        "translator", "翻译", ("trados", "memoq", "deepl", "linguee", "antconc", "word"), "writing"
    ),
)


@dataclass(frozen=True)
class ScenarioSpec:
    key: str
    label: int | None
    mode: str
    goal: str
    apps: tuple[str, ...] = ()  # Empty means the persona's actual professional tools.
    specialty_only: bool = False


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("deep_work", 1, "typing", "Complete the assigned professional task"),
    ScenarioSpec(
        "reference_reading",
        1,
        "reading",
        "Read evidence for the current task",
        ("acrobat", "zotero", "arxiv", "wikipedia"),
    ),
    ScenarioSpec(
        "lecture",
        1,
        "watching",
        "Attend the scheduled course",
        ("moodle", "coursera", "edx", "youtube", "bilibili", "zoom"),
    ),
    ScenarioSpec("collaboration", 1, "collaboration", "Coordinate a multi-tool deliverable"),
    ScenarioSpec(
        "incident_response",
        1,
        "collaboration",
        "Resolve the assigned incident",
        specialty_only=True,
    ),
    ScenarioSpec(
        "spreadsheet_audit", 1, "typing", "Reconcile the assigned accounts", specialty_only=True
    ),
    ScenarioSpec("data_analysis", 1, "typing", "Analyse the assigned dataset", specialty_only=True),
    ScenarioSpec(
        "design_iteration", 1, "drawing", "Revise the approved design", specialty_only=True
    ),
    ScenarioSpec("video_review", 1, "watching", "Review footage for the edit", specialty_only=True),
    ScenarioSpec(
        "audio_review",
        1,
        "watching",
        "Listen critically to the commissioned mix",
        specialty_only=True,
    ),
    ScenarioSpec(
        "lesson_preparation", 1, "typing", "Prepare the scheduled lesson", specialty_only=True
    ),
    ScenarioSpec("writing", 1, "typing", "Draft the agreed document"),
    ScenarioSpec(
        "customer_support",
        1,
        "collaboration",
        "Handle the assigned service queue",
        specialty_only=True,
    ),
    ScenarioSpec(
        "game_testing",
        1,
        "gaming",
        "Reproduce a game bug using the test plan",
        ("steam", "minecraft", "unity"),
        specialty_only=True,
    ),
    ScenarioSpec("documentation_reading", 1, "reading", "Read the tool manual"),
    ScenarioSpec("browser_closed", 1, "typing", "Work in local tools with browser closed"),
    ScenarioSpec(
        "goal_drift", 0, "typing", "Avoid the assignment by unrelated work in the same tools"
    ),
    ScenarioSpec(
        "off_topic_reading",
        0,
        "reading",
        "Read unrelated material instead of the task",
        ("acrobat", "zotero", "arxiv", "wikipedia"),
    ),
    ScenarioSpec(
        "chat_drift",
        0,
        "collaboration",
        "Social chat instead of the planned task",
        ("slack", "teams", "wechat", "discord"),
    ),
    ScenarioSpec(
        "video_binge",
        0,
        "watching",
        "Watch unrelated videos instead of the task",
        ("youtube", "bilibili", "netflix", "vlc"),
    ),
    ScenarioSpec(
        "casual_gaming", 0, "gaming", "Play instead of the agreed task", ("steam", "minecraft")
    ),
    ScenarioSpec(
        "social_feed",
        0,
        "reading",
        "Browse feeds instead of the task",
        ("reddit", "weibo", "x", "instagram"),
    ),
    ScenarioSpec(
        "tab_hopping",
        0,
        "collaboration",
        "Switch unrelated tabs to avoid starting",
        ("reddit", "gmail", "youtube", "wikipedia"),
    ),
    ScenarioSpec(
        "inbox_loop",
        0,
        "collaboration",
        "Repeated inbox checks displace the assignment",
        ("gmail", "outlook", "calendar"),
    ),
    ScenarioSpec(
        "planned_break", None, "rest", "Take an intentional break", ("spotify", "youtube", "vlc")
    ),
    ScenarioSpec("away", None, "away", "Step away intentionally"),
    ScenarioSpec("input_missing", None, "missing", "Input collector missing; intent unobserved"),
    ScenarioSpec(
        "unknown_app", None, "typing", "Unclassified tool; intent unobserved", ("unknown",)
    ),
    ScenarioSpec("mixed_intent", None, "collaboration", "Mixed task and unrelated activity"),
    ScenarioSpec(
        "meeting_unclear", None, "watching", "Meeting purpose not established", ("zoom", "teams")
    ),
)
SCENARIO_CATALOG: dict[str, ScenarioSpec] = {scenario.key: scenario for scenario in SCENARIOS}

# Fixed-offset zones keep the recipe independent of OS tzdata/DST revisions.
TIMEZONES: tuple[tuple[str, float], ...] = (
    ("UTC", 0.0),
    ("UTC+08:00", 8.0),
    ("UTC+05:30", 5.5),
    ("UTC+09:00", 9.0),
    ("UTC-04:00", -4.0),
    ("UTC-07:00", -7.0),
    ("UTC+02:00", 2.0),
    ("UTC+10:00", 10.0),
)
