"""Standalone Tkinter popup for Windows intervention responses."""

from __future__ import annotations

import json
import os
import sys
import time
import tkinter as tk
import urllib.request
from collections.abc import Callable
from contextlib import suppress
from functools import partial
from pathlib import Path
from tkinter import ttk
from typing import Any, Literal, TypedDict, cast

Response = Literal["accepted", "dismissed", "ignored"]

ACTION_RESPONSES: dict[str, Response] = {
    "accept": "accepted",
    "reject": "dismissed",
    "ignore": "ignored",
    "close": "ignored",
    "timeout": "ignored",
}

_BUTTON_RESPONSES: dict[str, Response] = {
    "接受": ACTION_RESPONSES["accept"],
    "拒绝": ACTION_RESPONSES["reject"],
    "暂时忽略": ACTION_RESPONSES["ignore"],
}


class PopupPayload(TypedDict):
    title: str
    body: str
    intervention_id: str
    api_url: str
    timeout_s: int


def post_response(
    api_url: str,
    auth_token: str,
    response: Response,
    latency_s: float,
    opener: Callable[..., Any] | None = None,
) -> bool:
    request = urllib.request.Request(
        api_url,
        data=json.dumps(
            {"response": response, "latency_s": latency_s},
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    open_request = opener or urllib.request.urlopen
    with open_request(request, timeout=5.0):
        return True


class InterventionPopup:
    """Small always-on-top dialog that records one intervention response."""

    def __init__(self, payload: PopupPayload, ready_path: Path) -> None:
        self._payload = payload
        self._ready_path = ready_path
        self._started_at = time.monotonic()
        self._root: tk.Tk | None = None
        self._responded = False

    def show(self) -> None:
        root = tk.Tk()
        self._root = root
        root.withdraw()
        root.title(self._payload["title"])
        root.resizable(False, False)
        root.attributes("-topmost", True)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.configure(background="#f7f7f7")

        style = ttk.Style(root)
        style.configure("MindFlow.TFrame", background="#f7f7f7")
        style.configure(
            "MindFlow.Title.TLabel",
            background="#f7f7f7",
            foreground="#202124",
            font=("Microsoft YaHei UI", 14, "bold"),
        )
        style.configure(
            "MindFlow.Body.TLabel",
            background="#f7f7f7",
            foreground="#3c4043",
            font=("Microsoft YaHei UI", 10),
        )

        content = ttk.Frame(root, padding=(28, 24, 28, 22), style="MindFlow.TFrame")
        content.pack(fill="both", expand=True)

        ttk.Label(
            content,
            text=self._payload["title"],
            style="MindFlow.Title.TLabel",
            wraplength=464,
            justify="left",
        ).pack(fill="x")
        ttk.Label(
            content,
            text=self._payload["body"],
            style="MindFlow.Body.TLabel",
            wraplength=464,
            justify="left",
        ).pack(fill="x", pady=(14, 22))

        buttons = ttk.Frame(content, style="MindFlow.TFrame")
        buttons.pack(fill="x")
        for button_text in ("接受", "拒绝", "暂时忽略"):
            ttk.Button(
                buttons,
                text=button_text,
                command=partial(self._on_action, button_text),
                width=12,
            ).pack(side="left", expand=True, padx=4)

        root.update_idletasks()
        width = 520
        height = max(220, root.winfo_reqheight())
        x = max(0, (root.winfo_screenwidth() - width) // 2)
        y = max(0, (root.winfo_screenheight() - height) // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")
        root.deiconify()
        root.lift()
        root.update()

        self._ready_path.write_text("ready", encoding="utf-8")
        root.after(self._payload["timeout_s"] * 1000, self._on_timeout)
        root.mainloop()

    def _on_action(self, button_text: str) -> None:
        self._finish(_BUTTON_RESPONSES[button_text])

    def _on_close(self) -> None:
        self._finish(ACTION_RESPONSES["close"])

    def _on_timeout(self) -> None:
        self._finish(ACTION_RESPONSES["timeout"])

    def _finish(self, response: Response) -> None:
        if self._responded:
            return
        self._responded = True
        latency_s = max(0.0, time.monotonic() - self._started_at)
        with suppress(Exception):
            post_response(
                api_url=self._payload["api_url"],
                auth_token=os.environ.get("MINDFLOW_POPUP_TOKEN", ""),
                response=response,
                latency_s=latency_s,
            )
        if self._root is not None:
            self._root.destroy()


def _load_payload(payload_path: Path) -> PopupPayload:
    data = json.loads(payload_path.read_text(encoding="utf-8"))
    return cast("PopupPayload", data)


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    payload = _load_payload(Path(sys.argv[1]))
    InterventionPopup(payload, Path(sys.argv[2])).show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
