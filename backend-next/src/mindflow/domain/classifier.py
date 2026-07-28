"""Built-in application classifier — domain layer.

Classifies applications into productivity categories using process name
matching and window title keyword heuristics.  No external dependencies
beyond the Python standard library.

Moved from ``mindflow.train.features`` (training pipeline) to resolve
a layer violation — ``domain`` should not depend on ``train``.

The ``AppClassifier`` is the pure, synchronous classifier.  User-custom
overrides live in ``mindflow.domain.app_classification.UserAppClassifier``.
"""

from __future__ import annotations


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
