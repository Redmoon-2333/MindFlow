"""Behavioral feature extraction from activity events.

Ported from ``backend/mindflow/analyzer/data_pipeline.py``.
Key differences vs. the original:
  - Input is ``list[ActivityEvent]`` instead of ``pandas.DataFrame`` — no
    pandas dependency in the domain boundary.
  - Output is ``list[dict]`` (compatible with ``BaselineModel.update()`` and
    ``ConsensusLabeler.label_dataframe()``).
  - The ``AppClassifier`` and ``TitleAnalyzer`` are kept as pure dict-based
    classifiers with no external dependencies.

Feature columns (17 total per 30-minute window):
  - ``unique_app_count``, ``switch_frequency``, ``productivity_ratio``,
    ``entertainment_ratio``, ``social_ratio``, ``max_app_duration``,
    ``idle_ratio``, ``hour_of_day``, ``day_of_week``
  - ``title_code_ratio``, ``title_doc_ratio``, ``title_url_ratio``,
    ``title_meeting_ratio``, ``title_entertainment_ratio``
  - ``activity_entropy``, ``context_switch_cost``,
    ``temporal_decay_weight``
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

from mindflow.domain.events import ActivityEvent
from mindflow.domain.features import title_features as _domain_title_features


class TitleAnalyzer:
    """Window title analysis delegating to domain.features.title_features().

    Single source of truth for title-based signal extraction.  The domain
    function handles URL detection, file-extension matching, meeting-keyword
    detection, and entertainment-pattern matching.  This wrapper converts the
    boolean ``TitleFeatures`` result to the 0.0/1.0 float dict that the
    training pipeline expects.
    """

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
        tf = _domain_title_features(title)
        return {
            "is_code_editor": 1.0 if tf.is_code_editor else 0.0,
            "is_document": 1.0 if tf.is_document else 0.0,
            "is_browser": 1.0 if tf.is_browser else 0.0,
            "is_meeting": 1.0 if tf.is_meeting else 0.0,
            "is_likely_entertainment": 1.0 if tf.is_likely_entertainment else 0.0,
        }


class AppClassifier:
    """Classify applications into productivity categories.

    Uses process name matching and window title keyword heuristics.
    No external dependencies — pure dict-based lookup.
    """

    PRODUCTIVITY_APPS: dict[str, list[str]] = {
        "code": [
            # IDEs & code editors
            "code", "vscode", "vscodium", "codium", "pycharm", "intellij",
            "eclipse", "sublime", "nvim", "vim", "gvim", "emacs",
            "android studio", "androidstudio", "visual studio", "cursor",
            "claude", "xcode", "rstudio", "spyder", "datagrip",
            "webstorm", "goland", "clion", "rider", "phpstorm", "rubymine",
            "fleet", "studio64", "netbeans", "atom", "notepad++",
            "helix", "zed", "windsurf", "trae", "brackets", "komodo",
            "bluej", "codeblocks", "devcpp", "qtcreator", "geany",
            "jupyter", "jupyter-lab", "thonny", "notepad2", "notepad3",
            "textpad", "pspad", "akelpad", "editplus", "ultraedit",
            "scite", "textadept", "lapce", "devenv",
            # Debuggers & disassemblers
            "windbg", "x64dbg", "ilspy", "dnspy", "dotpeek",
            # Specialized dev tools
            "wechatdevtools", "devecostudio64",
            # AI coding assistants
            "lingma", "comate", "marscode", "tabnine",
            # Terminals & shells
            "terminal", "powershell", "cmd", "warp", "alacritty",
            "windows-terminal", "wt", "conemu", "cmder", "tabby", "hyper",
            "mobaxterm", "putty", "termius", "xshell", "finalshell",
            "windterm", "fluentterminal", "mintty", "securecrt", "royalts",
            "ttermpro", "zoc", "powershell_ise", "pwsh", "bash", "wsl",
            "nushell",
            # Version control
            "github", "gitkraken", "sourcetree", "fork", "gitextensions",
            "smartgit", "tortoisegit", "tortoisesvn", "tower", "gitahead",
            "git-cola", "gitup", "sublime_merge", "git", "git-lfs",
            "git-credential-manager", "gh", "glab",
            # Database tools
            "dbeaver", "dbeaver-cli", "navicat", "tableplus", "pgadmin",
            "mysql-workbench", "mysqlworkbench", "mongodb-compass", "compass",
            "robo3t", "studio-3t", "ssms", "azuredatastudio", "sqlitebrowser",
            "heidisql", "beekeeper-studio", "sqlectron", "postbird",
            "sqlyog", "dbvis", "plsqldev", "sqldeveloper", "dataspell",
            "aqua", "mysql", "mysqld", "postgresql", "mongod", "redis-server",
            "redis-cli", "sqlite3",
            # API testing & HTTP clients
            "postman", "insomnia", "bruno", "soapui", "swagger",
            "charles", "charles-proxy", "fiddler", "wireshark",
            "burpsuite", "burp", "httpie", "curl", "apifox", "apipost",
            "eolinker", "http-toolkit", "jmeter", "k6", "newman",
            "artillery",
            # DevOps & containers
            "docker", "docker-desktop", "dockerd", "rancher-desktop",
            "podman", "podman-desktop", "lens", "kubectl", "minikube",
            "kind", "helm", "k9s", "terraform", "packer", "vagrant",
            "vault", "consul", "nomad", "pulumi", "ngrok", "cloudflared",
            "ansible", "jenkins", "gitlab-runner", "teamcity", "octopus",
            # Math, science & engineering
            "matlab", "mathematica", "maple", "geogebra", "mathcad",
            "mathcadprime", "octave", "scilab", "wxmaxima", "spss", "stata",
            "sas", "jmp", "minitab", "rgui", "julia", "jamovi", "jasp",
            "eviews", "gretl", "autodesk", "autocad", "acad", "solidworks",
            "sldworks", "inventor", "fusion360", "revit", "sketchup",
            "catia", "freecad", "kicad", "eagle", "ltspice", "pspice",
            "altium", "dxp", "proteus", "isis", "keil", "uv4", "quartus",
            "vivado", "labview", "multisim", "ansys", "ansyswbu", "comsol",
            "fluent", "cfd", "origin", "originpro", "prism", "graphpad",
            "sigmaplot", "pscad", "tableau", "powerbi", "pbi", "qlik",
            "qlikview", "knime", "rapidminer", "alteryx", "talend", "pentaho",
            "grafana",
            # Cloud CLIs
            "aws", "az", "gcloud", "doctl", "heroku", "vercel", "netlify",
            "wrangler", "firebase", "supabase",
        ],
        "document": [
            # Office suites
            "word", "winword", "excel", "powerpoint", "powerpnt",
            "wps", "wpp", "et", "msaccess", "mspub", "publisher", "visio",
            "project", "libreoffice", "soffice", "openoffice", "onlyoffice",
            "polari", "softmaker", "freeoffice", "hancom", "zoho",
            "google-docs", "pages", "keynote", "numbers",
            # Note-taking & knowledge management
            "notion", "obsidian", "typora", "pdf", "evernote", "onenote",
            "joplin", "logseq", "roam", "bear", "simplenote", "milanote",
            "heptabase", "yuque", "yinxiang", "youdaonote", "wiznote",
            "mubu", "siyuan", "wolai", "flowus", "anytype", "goodnotes",
            "notability", "scrivener", "liquidtext", "marginnote",
            # Reference managers
            "zotero", "mendeley", "endnote", "citavi", "paperpile",
            "readcube", "noteexpress", "qiqqa", "jabref", "bibdesk",
            "calibre",
            # Flashcard & study apps
            "anki", "quizlet", "memrise", "remnote", "supermemo",
            "mnemosyne", "brainscape", "studyblue", "cram",
            # Email clients
            "outlook", "thunderbird", "spark-mail", "sparkmail", "mailbird",
            "emclient", "postbox", "foxmail", "mailmaster", "thebat",
            "inky", "mailspring", "canarymail", "edisonmail", "airmail",
            "claws-mail", "pegasusmail",
            # Calendar & task management
            "google-calendar", "fantastical", "sunrise", "any-do",
            "todoist", "ticktick",
        ],
        "browser_work": [
            # Dev & research platforms
            "github", "stackoverflow", "gitlab", "bitbucket",
            "docs", "jupyter", "colab", "arxiv", "scholar",
            # Learning & course platforms
            "coursera", "edx", "udemy", "udacity", "khanacademy",
            "skillshare", "pluralsight", "linkedin-learning", "codecademy",
            "datacamp", "brilliant", "futurelearn", "masterclass",
            # Chinese learning platforms
            "chaoxing", "cxexam", "netease-cloud-class", "xuetangx",
            "zhihuishu", "xueersi", "yuanfudao", "zuoyebang", "gaotu",
            "txclass", "koolearn", "kaochong", "fenbi", "offcn", "huatu",
            # Language learning
            "duolingo", "rosettastone", "babbel", "busuu", "eudic",
            "youdaodict", "iciba", "deepl", "linguee", "translate",
            # Cloud storage
            "onedrive", "googledrive", "dropbox", "mega", "megasync",
            "pcloud", "box", "icloud", "baidunetdisk", "aliyundrive",
            "ecloud", "115", "quark", "jianguoyun", "nextcloud",
            "syncthing", "resiliosync", "owncloud", "weiyun", "hcyun",
            # Download managers
            "idman", "jdownloader", "fdm", "motrix", "aria2", "eagleget",
            "xdm", "qbittorrent", "utorrent", "bittorrent", "transmission",
            "deluge", "tixati", "thunder", "bitcomet", "bitspirit", "emule",
            # System utilities
            "ccleaner", "everything", "ditto", "sharex", "greenshot",
            "lightshot", "snagit", "powertoys", "listary", "wox",
            "flow-launcher", "utools", "quicker", "autohotkey", "launchy",
            "geek", "revo", "hwmonitor", "speccy", "cpuz", "gpuz", "hwinfo",
            "7zip", "winrar", "bandizip", "snipaste", "devtoys",
            # Font managers
            "fontbase", "suitcase", "rightfont", "fontcreator", "fontlab",
            # Password managers
            "bitwarden", "1password", "lastpass", "dashlane", "keepass",
            "keepassxc", "roboform", "enpass", "nordpass",
            # Photo management
            "lightroom", "captureone", "darktable", "rawtherapee", "luminar",
            "acdsee", "faststone", "xnview", "irfanview", "digikam",
            "photoscape", "zoner",
        ],
        "communication": [
            # Messaging apps
            "teams", "ms-teams", "slack", "dingtalk", "feishu", "lark",
            "wechat", "weixin", "wxwork", "tim", "qq", "telegram",
            "discord", "whatsapp", "signal", "messenger", "line", "viber",
            "kakaotalk", "skype", "wire", "threema", "element", "zulip",
            "mattermost", "rocket-chat", "flock", "chanty", "twist",
            "aliwangwang", "qianniu",
            # Video conferencing
            "zoom", "meet", "google-meet", "cisco-webex", "webex",
            "gotomeeting", "bluejeans", "whereby", "jitsi-meet",
            "ringcentral", "wemeet", "tencent-meeting", "huawei-meeting",
            "xiaoyu-yilian", "xiaoyu",
            # Remote desktop
            "mstsc", "teamviewer", "anydesk", "parsec", "rustdesk",
            "splashtop", "vncviewer", "tightvnc", "realvnc", "ultravnc",
            "chrome-remote-desktop", "anyviewer", "awesun", "supremo",
            "nomachine",
            # Project management & collaboration
            "trello", "clickup", "asana", "monday", "linear", "airtable",
            "miro", "figma", "mural", "basecamp", "wrike", "smartsheet",
            "confluence", "meistertask",
        ],
        "entertainment": [
            # Video streaming
            "bilibili", "youtube", "netflix", "iqiyi", "youku",
            "disney", "disneyplus", "hbomax", "hulu", "amazon-video",
            "primevideo", "crunchyroll", "twitch", "qqlive", "mgtv",
            "sohuvideo", "xigua", "funshion", "pptv", "cbox", "miguvideo",
            "douyu", "huya", "cclive", "acfun", "letv",
            # Music streaming
            "spotify", "qqmusic", "cloudmusic", "kugou", "kuwo",
            "itunes", "applemusic", "tidal", "deezer", "pandora",
            "soundcloud", "bandcamp",
            # Game launchers & platforms
            "game", "steam", "epic", "epicgames", "galaxy", "gog",
            "uplay", "ubisoft", "ubisoftconnect", "ea", "origin",
            "battle-net", "battle.net", "riot", "riotclient", "xbox",
            "xboxapp", "rockstar", "rockstargames", "wegame", "tenprotect",
            "360game", "4399",
            # Android emulators
            "bluestacks", "ldplayer", "nox", "mumu", "memu", "koplayer",
            "gameloop",
            # PC games
            "leagueoflegends", "leagueclient", "valorant", "cs2", "csgo",
            "dota2", "r5apex", "apex", "fortnite", "fortniteclient",
            "pubg", "tslgame", "gta5", "minecraft", "javaw", "roblox",
            "robloxplayer", "overwatch", "wow", "ffxiv", "ffxiv_dx11",
            "genshinimpact", "yuanshen", "starrail", "wuthering",
            "wutheringwaves", "eldenring", "bg3", "bg3_dx11", "cyberpunk2077",
            "warthunder", "pathofexile", "diablo", "diablo4", "hearthstone",
            "lostark", "blackdesert", "blackdesert64", "maplestory", "dnf",
            "crossfire", "naraka", "shooter", "factorio", "stardew",
            "terraria", "rimworld", "sims", "ts4", "starcraft", "sc2",
            "rocketleague", "rainbowsix", "deadbydaylight", "warframe",
            "destiny2", "cod", "dota", "tf2", "overwatch2",
            "mygame", "myserver", "nsh", "wyx", "jx3", "jx3client",
            # Emulators
            "retroarch", "dolphin", "pcsx2", "rpcs3", "cemu", "yuzu",
            "ryujinx", "citra", "ppsspp", "mame", "xenia", "xemu",
            "duckstation", "melonds", "desmume", "visualboyadvance",
            "mgba", "snes9x", "zsnes", "project64", "epsxe", "flycast",
            "redream", "nulldc", "mednafen",
            # Media players
            "vlc", "potplayer", "potplayermini64", "mpc-hc", "mpc-be",
            "kmplayer", "gom", "smplayer", "plex", "jellyfin", "emby",
            "kodi", "5kplayer", "qqplayer", "stormplayer", "baofeng",
            "xlplayer", "qvodplayer",
            # Audio/DAW
            "audacity", "flstudio", "fl64", "ableton", "abletonlive",
            "cubase", "reaper", "studio-one", "studioone", "lmms",
            "bitwig", "reason", "cakewalk", "wavelab", "izotope",
            # Novel & reading platforms
            "qidian", "ireader", "migureader", "kuaikanmanhua", "kindle",
            "audible", "wattpad", "webnovel", "webtoon", "ximalaya",
            "qingting", "dedao",
            # Short video & short drama
            "douyin", "tiktok", "kwai", "weishi", "huoshan", "zuiyou",
            "pipixia", "meipai", "miaopai", "yangshipin", "likee",
            "triller", "haokan", "hongguo",
        ],
        "social": [
            # Chinese social
            "weibo", "tieba", "maimai", "jike", "hupu", "xueqiu",
            "zsxq", "soul", "momo", "tantan", "jimubox",
            "csdn", "juejin", "segmentfault", "oschina", "sspai",
            "nga", "stage1st", "chiphell", "tgfc", "52pojie",
            # International social
            "twitter", "twitterx", "facebook", "instagram", "threads",
            "mastodon", "bluesky", "tumblr", "pinterest", "snapchat",
            "linkedin", "quora", "hackernews", "producthunt", "devto",
            "medium", "substack", "bereal", "nextdoor", "v2ex",
            # Forums & communities
            "zhihu", "reddit", "stackoverflow", "douban", "xiaohongshu",
            # Live streaming (social aspect)
            "douyu", "huya", "chushou", "longzhu", "quanmin",
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
        "activity_entropy",
        "context_switch_cost",
        "temporal_decay_weight",
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
            active_rows = [row for row in window_rows if not row["is_idle"]]

            # Classify apps
            for r in window_rows:
                category = self.app_classifier.classify(
                    str(r.get("process_name", "")),
                    str(r.get("window_title", "")),
                )
                r["app_category"] = category

            productivity_seconds = sum(
                row["duration_seconds"]
                for row in active_rows
                if row.get("app_category") in ("code", "document", "browser_work")
            )
            entertainment_seconds = sum(
                row["duration_seconds"]
                for row in active_rows
                if row.get("app_category") == "entertainment"
            )
            social_seconds = sum(
                row["duration_seconds"]
                for row in active_rows
                if row.get("app_category") == "social"
            )

            unique_apps = len({row["process_name"] for row in active_rows})

            # NOTE: We do NOT delegate to domain.features.switch_rate_per_hour
            # here because the semantics differ: the domain function filters
            # idle events and divides by actual time span, while the training
            # pipeline counts all process switches (including idle) and divides
            # by fixed window duration.  Consolidating would change feature
            # distributions and break trained models.
            process_list = [row["process_name"] for row in active_rows]
            switches = sum(
                1 for j in range(1, len(process_list))
                if process_list[j] != process_list[j - 1]
            )
            hours_in_window = self.window_minutes / 60.0
            switch_freq = switches / hours_in_window if hours_in_window > 0 else 0.0

            app_durations: dict[str, float] = defaultdict(float)
            for row in active_rows:
                app_durations[row["process_name"]] += row["duration_seconds"]
            max_app_duration = max(app_durations.values()) if app_durations else 0.0

            # Title-based features (no app classification dependency)
            title_features = [
                self.title_analyzer.analyze(str(row.get("window_title", "")))
                for row in active_rows
            ]
            n_titles = max(len(title_features), 1)
            code_ratio = sum(tf["is_code_editor"] for tf in title_features) / n_titles
            doc_ratio = sum(tf["is_document"] for tf in title_features) / n_titles
            url_ratio = sum(tf["is_browser"] for tf in title_features) / n_titles
            meeting_ratio = sum(tf["is_meeting"] for tf in title_features) / n_titles
            entertainment_title_ratio = (
                sum(tf["is_likely_entertainment"] for tf in title_features) / n_titles
            )

            # ── Novel behavioral features (IEEE 2026 Ensemble ML + HMM paper) ─

            # activity_entropy: Shannon entropy of app category distribution
            # H = -sum(p_i * log2(p_i)), clamped to [0,1] via / log2(7)
            category_times: dict[str, float] = defaultdict(float)
            for row in active_rows:
                category = row.get("app_category", "other")
                category_times[category] += row["duration_seconds"]
            raw_entropy = 0.0
            for cat_time in category_times.values():
                p = cat_time / active_seconds
                if p > 0:
                    raw_entropy -= p * math.log2(p)
            activity_entropy = (
                raw_entropy / math.log2(7) if raw_entropy > 0 else 0.0
            )

            # context_switch_cost: weighted switch count by category distance
            # Transitions between distant categories (e.g. code→entertainment)
            # cost more than nearby ones. Normalized per minute of active time.
            _CATEGORY_ORDER: dict[str, int] = {
                "code": 0,
                "document": 1,
                "browser_work": 2,
                "communication": 3,
                "other": 4,
                "entertainment": 5,
                "social": 6,
            }
            switch_cost_total = 0.0
            for j in range(1, len(active_rows)):
                prev_cat = active_rows[j - 1].get("app_category", "other")
                curr_cat = active_rows[j].get("app_category", "other")
                if prev_cat != curr_cat:
                    prev_int = _CATEGORY_ORDER.get(prev_cat, 4)
                    curr_int = _CATEGORY_ORDER.get(curr_cat, 4)
                    switch_cost_total += abs(curr_int - prev_int) / 6.0
            context_switch_cost = (
                switch_cost_total / (active_seconds / 60.0)
                if active_seconds > 0
                else 0.0
            )

            # temporal_decay_weight: exponential decay weighting for recency bias
            # Higher value = events clustered toward the end of the window.
            window_seconds = (w_end - w_start).total_seconds()
            if window_seconds > 0 and active_rows:
                t_max_ts = max(row["timestamp"] for row in active_rows)
                weights: list[float] = []
                for row in active_rows:
                    delta = (t_max_ts - row["timestamp"]).total_seconds()
                    weights.append(math.exp(-delta / window_seconds))
                temporal_decay_weight = sum(weights) / len(weights)
            else:
                temporal_decay_weight = 0.0

            records.append({
                "window_start": w_start.isoformat(),
                "unique_app_count": unique_apps,
                "switch_frequency": round(switch_freq, 4),
                "productivity_ratio": round(
                    min(max(productivity_seconds / active_seconds, 0.0), 1.0), 4
                ),
                "entertainment_ratio": round(
                    min(max(entertainment_seconds / active_seconds, 0.0), 1.0), 4
                ),
                "social_ratio": round(
                    min(max(social_seconds / active_seconds, 0.0), 1.0), 4
                ),
                "max_app_duration": round(max_app_duration, 2),
                "idle_ratio": round(min(max(idle_ratio, 0.0), 1.0), 4),
                "hour_of_day": int(w_start.hour),
                "day_of_week": int(w_start.weekday()),
                "title_code_ratio": round(code_ratio, 4),
                "title_doc_ratio": round(doc_ratio, 4),
                "title_url_ratio": round(url_ratio, 4),
                "title_meeting_ratio": round(meeting_ratio, 4),
                "title_entertainment_ratio": round(entertainment_title_ratio, 4),
                "activity_entropy": round(min(max(activity_entropy, 0.0), 1.0), 4),
                "context_switch_cost": round(context_switch_cost, 4),
                "temporal_decay_weight": round(temporal_decay_weight, 4),
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
