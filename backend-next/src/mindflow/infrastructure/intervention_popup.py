"""Standalone Tkinter popup for Windows intervention responses.

The window is hand-painted rather than themed: tk's native widgets cannot be
coloured to match the app (a light system title bar, a grey feather icon and
three identical buttons were what the user objected to). The palette below is
the dashboard's own, copied from ``frontend/src/theme.css`` so the reminder
does not look like a different product.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import sys
import time
import tkinter as tk
import urllib.request
from collections.abc import Callable
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

Response = Literal["accepted", "dismissed", "ignored"]

# frontend/src/theme.css — navy theme.
_BG = "#0a0e14"
_SURFACE = "#121a24"
_SURFACE_2 = "#18222e"
_LINE = "#1f2a36"
_TEXT = "#d7dee6"
_TEXT_DIM = "#8e9baa"
_SIGNAL = "#33b5e5"
_SIGNAL_HOVER = "#5cc6ec"

_FONT = "Microsoft YaHei UI"

_WINDOW_WIDTH = 400
_RIGHT_MARGIN = 24
_BOTTOM_MARGIN = 72
_WRAP_LENGTH = 352
_ALPHA_TARGET = 0.97
_FADE_MS = 160
_FADE_STEPS = 16
_COUNTDOWN_TICK_MS = 200
# Shorter than post_response's 5s default: the window is already hidden by the
# time this runs, so a dead server must not keep the process alive.
_POST_TIMEOUT_S = 2.0

ACTION_RESPONSES: dict[str, Response] = {
    "accept": "accepted",
    "reject": "dismissed",
    "dismiss": "dismissed",
    "ignore": "ignored",
    "close": "ignored",
    "timeout": "ignored",
}

_BUTTON_RESPONSES: dict[str, Response] = {
    "好，试试看": ACTION_RESPONSES["accept"],
    "先忽略": ACTION_RESPONSES["ignore"],
    "不适合我": ACTION_RESPONSES["dismiss"],
}


def _countdown_fraction(elapsed: float, timeout: float) -> float:
    """Fraction of the countdown bar still filled: 1.0 at t=0, 0.0 at timeout.

    Pure so the bar's arithmetic is testable without a display. A non-positive
    timeout has nothing to count down, and an elapsed time past the deadline
    must not draw a negative-width rectangle.
    """
    if timeout <= 0:
        return 0.0
    remaining = 1.0 - elapsed / timeout
    return min(1.0, max(0.0, remaining))


# A space immediately between CJK and an ASCII alphanumeric, in either order.
# Such a space is the only break opportunity in "设定 5 分钟", so Tk prefers it
# and leaves a ragged line. U+00A0 looks the same but is a non-breaking space;
# verified on Tk 8.6.15 / Windows: with NBSP the break moves to the next ASCII
# space, and with NBSP as the only candidate the text stays on one line.
_CJK_ALNUM_SPACE = re.compile(r"(?<=[一-鿿]) (?=[A-Za-z0-9])|(?<=[A-Za-z0-9]) (?=[一-鿿])")


def _cjk_wrap_text(text: str) -> str:
    """Replace the space between CJK and alphanumerics with a hard one.

    Tk wraps on spaces, so a space next to CJK text reads as a preferred break
    even though the two halves belong together ("设定 5 分钟" broke after
    "5"). Joining them with U+00A0 keeps the phrase intact while leaving every
    other space breakable.
    """
    return _CJK_ALNUM_SPACE.sub(" ", text)


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
    *,
    source: str = "human",
    timeout: float = 5.0,
) -> bool:
    """POST one response back to the API.

    ``source="auto"`` marks a dialog timeout default. The server only lets an
    auto response fill an empty slot, so a timeout that fires while the user is
    clicking "好，试试看" can never overwrite the click.
    """
    request = urllib.request.Request(
        api_url,
        data=json.dumps(
            {"response": response, "latency_s": latency_s, "source": source},
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    open_request = opener or urllib.request.urlopen
    with open_request(request, timeout=timeout):
        return True


def _enable_dpi_awareness() -> None:
    """Opt into per-monitor DPI before the first Tk window exists.

    Without this Windows renders the window at 96 DPI and bitmap-scales it, so
    text is blurry on a scaled display. Only the two "this API is absent"
    errors are swallowed; anything else is a real defect worth seeing.
    """
    if sys.platform != "win32":
        return
    with suppress(AttributeError, OSError):
        ctypes.windll.shcore.SetProcessDpiAwareness(1)


class _FlatButton(tk.Label):
    """A label that behaves like a button, so its colours are fully ours."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        text: str,
        on_click: Callable[[str], None],
        scale: float,
        background: str,
        foreground: str,
        hover_background: str,
        hover_foreground: str,
        bold: bool = False,
        border: bool = False,
    ) -> None:
        font = (_FONT, 10, "bold") if bold else (_FONT, 10)
        # Every button carries the same 1px highlight ring, tinted to its own
        # fill when it has no border, so all three boxes share one height and
        # their text baselines line up.
        ring = round(scale)
        super().__init__(
            master,
            text=text,
            font=font,
            background=background,
            foreground=foreground,
            padx=int(14 * scale),
            pady=int(6 * scale),
            cursor="hand2",
            highlightthickness=ring,
            highlightbackground=_LINE if border else background,
            highlightcolor=_LINE if border else background,
        )
        self._background = background
        self._foreground = foreground
        self._hover_background = hover_background
        self._hover_foreground = hover_foreground
        self.bind("<Button-1>", lambda _event: on_click(text))
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)

    def _enter(self, _event: tk.Event[tk.Misc]) -> None:
        self.configure(background=self._hover_background, foreground=self._hover_foreground)

    def _leave(self, _event: tk.Event[tk.Misc]) -> None:
        self.configure(background=self._background, foreground=self._foreground)


class InterventionPopup:
    """Small always-on-top dialog that records one intervention response."""

    def __init__(self, payload: PopupPayload, ready_path: Path) -> None:
        self._payload = payload
        self._ready_path = ready_path
        self._started_at = time.monotonic()
        self._root: tk.Tk | None = None
        self._responded = False
        self._drag_origin: tuple[int, int] | None = None
        self._after_ids: set[str] = set()

    def show(self) -> None:
        _enable_dpi_awareness()
        root = tk.Tk()
        self._root = root
        root.withdraw()
        root.overrideredirect(True)
        root.resizable(False, False)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.0)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.configure(background=_LINE)

        scale = root.winfo_fpixels("1i") / 96.0

        # 1px hairline frame drawn by the outer background.
        border = max(1, round(scale))
        inner = tk.Frame(root, background=_SURFACE)
        inner.pack(fill="both", expand=True, padx=border, pady=border)

        # Countdown rail: a real last row inside the hairline, so it is neither
        # clipped by the window height nor swallowed by the frame. Packed before
        # the body so its height is reserved and counted by winfo_reqheight.
        bar_height = max(2, round(2 * scale))
        countdown = tk.Canvas(
            inner,
            height=bar_height,
            background=_SURFACE,
            highlightthickness=0,
            borderwidth=0,
        )
        countdown.pack(side="bottom", fill="x")
        bar = countdown.create_rectangle(0, 0, 0, bar_height, fill=_SIGNAL, outline="")

        body = tk.Frame(inner, background=_SURFACE)
        body.pack(side="top", fill="both", expand=True)

        content = tk.Frame(body, background=_SURFACE)
        rail_width = max(3, round(3 * scale))
        content.pack(side="left", fill="both", expand=True, padx=(rail_width, 0))

        # Flush with the left edge and full window height, so it runs over the
        # hairline rather than sitting inside it.
        rail = tk.Frame(root, background=_SIGNAL, width=rail_width)
        rail.place(x=0, y=0, width=rail_width, relheight=1.0)

        top_bar = tk.Frame(content, background=_SURFACE)
        top_bar.pack(fill="x", padx=int(20 * scale), pady=(int(16 * scale), 0))

        dot = tk.Label(
            top_bar, text="●", font=(_FONT, 9), background=_SURFACE, foreground=_SIGNAL
        )
        dot.pack(side="left")
        wordmark = tk.Label(
            top_bar,
            text="MindFlow · 专注提示",
            font=(_FONT, 9),
            background=_SURFACE,
            foreground=_TEXT_DIM,
        )
        wordmark.pack(side="left", padx=(int(6 * scale), 0))

        # Children swallow clicks, so the whole bar must opt in to dragging.
        for draggable in (top_bar, dot, wordmark):
            draggable.bind("<Button-1>", self._start_drag)
            draggable.bind("<B1-Motion>", self._on_drag)

        close = tk.Label(
            top_bar,
            text="×",
            font=(_FONT, 9),
            background=_SURFACE,
            foreground=_TEXT_DIM,
            cursor="hand2",
        )
        close.pack(side="right")
        close.bind("<Button-1>", lambda _event: self._on_close())
        close.bind(
            "<Enter>",
            lambda _event: close.configure(foreground=_TEXT),
        )
        close.bind(
            "<Leave>",
            lambda _event: close.configure(foreground=_TEXT_DIM),
        )

        tk.Label(
            content,
            text=_cjk_wrap_text(self._payload["title"]),
            font=(_FONT, 13, "bold"),
            background=_SURFACE,
            foreground=_TEXT,
            wraplength=int(_WRAP_LENGTH * scale),
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=int(20 * scale), pady=(int(10 * scale), 0))

        tk.Label(
            content,
            text=_cjk_wrap_text(self._payload["body"]),
            font=(_FONT, 10),
            background=_SURFACE,
            foreground=_TEXT_DIM,
            wraplength=int(_WRAP_LENGTH * scale),
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=int(20 * scale), pady=(int(6 * scale), 0))

        buttons = tk.Frame(content, background=_SURFACE)
        buttons.pack(anchor="e", padx=int(20 * scale), pady=(int(16 * scale), int(16 * scale)))

        _FlatButton(
            buttons,
            text="不适合我",
            on_click=self._on_action,
            scale=scale,
            background=_SURFACE,
            foreground=_TEXT_DIM,
            hover_background=_SURFACE,
            hover_foreground=_TEXT,
        ).pack(side="left", padx=(0, int(8 * scale)))

        _FlatButton(
            buttons,
            text="先忽略",
            on_click=self._on_action,
            scale=scale,
            background=_SURFACE_2,
            foreground=_TEXT,
            hover_background=_LINE,
            hover_foreground=_TEXT,
            border=True,
        ).pack(side="left", padx=(0, int(8 * scale)))

        _FlatButton(
            buttons,
            text="好，试试看",
            on_click=self._on_action,
            scale=scale,
            background=_SIGNAL,
            foreground=_BG,
            hover_background=_SIGNAL_HOVER,
            hover_foreground=_BG,
            bold=True,
        ).pack(side="left")

        root.update_idletasks()
        width = int(_WINDOW_WIDTH * scale)
        height = root.winfo_reqheight()
        x = max(0, root.winfo_screenwidth() - width - int(_RIGHT_MARGIN * scale))
        y = max(0, root.winfo_screenheight() - height - int(_BOTTOM_MARGIN * scale))
        root.geometry(f"{width}x{height}+{x}+{y}")

        # Keep the signal rail continuous down to the very bottom edge.
        rail.lift()

        root.bind("<Return>", lambda _event: self._on_action("好，试试看"))
        root.bind("<Escape>", lambda _event: self._on_close())

        root.deiconify()
        root.lift()
        root.focus_force()
        root.update()

        self._ready_path.write_text("ready", encoding="utf-8")
        self._fade_in(step=0)
        self._tick_countdown(countdown, bar, bar_height)
        self._schedule(self._payload["timeout_s"] * 1000, self._on_timeout)
        root.mainloop()

    # ── window chrome ───────────────────────────────────────────────────
    def _schedule(self, delay_ms: int, callback: Callable[[], None]) -> None:
        """Queue a callback and remember it, so teardown can cancel it.

        Without this the pending ticks outlive ``destroy()`` and Tk logs
        "invalid command name" for each one.
        """
        if self._root is None:
            return
        after_id = self._root.after(delay_ms, callback)
        if after_id is not None:
            self._after_ids.add(after_id)

    def _cancel_scheduled(self) -> None:
        if self._root is None:
            return
        for after_id in self._after_ids:
            self._root.after_cancel(after_id)
        self._after_ids.clear()

    def _fade_in(self, *, step: int) -> None:
        if self._root is None:
            return
        self._root.attributes("-alpha", _ALPHA_TARGET * step / _FADE_STEPS)
        if step < _FADE_STEPS:
            self._schedule(_FADE_MS // _FADE_STEPS, partial(self._fade_in, step=step + 1))

    def _tick_countdown(self, canvas: tk.Canvas, bar: int, height: int) -> None:
        if self._root is None or self._responded:
            return
        elapsed = time.monotonic() - self._started_at
        fraction = _countdown_fraction(elapsed, float(self._payload["timeout_s"]))
        # Read the mapped width: the canvas is fill="x", so its real width is
        # only known after the window is laid out.
        canvas.coords(bar, 0, 0, canvas.winfo_width() * fraction, height)
        self._schedule(_COUNTDOWN_TICK_MS, partial(self._tick_countdown, canvas, bar, height))

    def _start_drag(self, event: tk.Event[tk.Misc]) -> None:
        if self._root is None:
            return
        self._drag_origin = (
            event.x_root - self._root.winfo_x(),
            event.y_root - self._root.winfo_y(),
        )

    def _on_drag(self, event: tk.Event[tk.Misc]) -> None:
        if self._root is None or self._drag_origin is None:
            return
        x = event.x_root - self._drag_origin[0]
        y = event.y_root - self._drag_origin[1]
        self._root.geometry(f"+{x}+{y}")

    # ── responses ───────────────────────────────────────────────────────
    def _on_action(self, button_text: str) -> None:
        self._finish(_BUTTON_RESPONSES[button_text])

    def _on_close(self) -> None:
        # Closing the window is still a deliberate user action.
        self._finish(ACTION_RESPONSES["close"], source="human")

    def _on_timeout(self) -> None:
        # No user action happened: report it as an automatic default so the
        # server will not let it overwrite a real answer.
        self._finish(ACTION_RESPONSES["timeout"], source="auto")

    def _finish(self, response: Response, *, source: str = "human") -> None:
        if self._responded:
            return
        self._responded = True
        self._cancel_scheduled()
        latency_s = max(0.0, time.monotonic() - self._started_at)
        # Hide first: the POST can block for its whole timeout (server slow or
        # already gone), and the window must still vanish on the same click.
        if self._root is not None:
            self._root.withdraw()
            self._root.update_idletasks()
        with suppress(Exception):
            post_response(
                api_url=self._payload["api_url"],
                auth_token=os.environ.get("MINDFLOW_POPUP_TOKEN", ""),
                response=response,
                latency_s=latency_s,
                source=source,
                timeout=_POST_TIMEOUT_S,
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
