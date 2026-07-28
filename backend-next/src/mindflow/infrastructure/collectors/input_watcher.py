"""Windows Raw Input watcher with aggregate-only buckets."""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from datetime import UTC, datetime
from typing import Any


class InteractionAccumulator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keypress_count = 0
        self._mouse_click_count = 0
        self._scroll_delta = 0
        self._mouse_distance_px = 0.0
        self._input_active_s = 0.0
        self._interaction_burst_count = 0
        self._last_interaction_at: float | None = None

    def _touch(self) -> None:
        now = time.monotonic()
        if self._last_interaction_at is None or now - self._last_interaction_at > 2.0:
            self._interaction_burst_count += 1
        self._last_interaction_at = now

    def record_key(self) -> None:
        with self._lock:
            self._keypress_count += 1
            self._touch()

    def record_click(self) -> None:
        with self._lock:
            self._mouse_click_count += 1
            self._touch()

    def record_scroll(self, delta: int) -> None:
        with self._lock:
            self._scroll_delta += int(delta)
            self._touch()

    def record_move(self, delta_x: int, delta_y: int) -> None:
        with self._lock:
            self._mouse_distance_px += math.hypot(delta_x, delta_y)
            self._touch()

    def record_activity(self, seconds: float) -> None:
        with self._lock:
            self._input_active_s += max(0.0, seconds)

    def snapshot_and_reset(self, duration_s: float) -> dict[str, int | float]:
        with self._lock:
            bucket = {
                "duration_s": float(duration_s),
                "keypress_count": self._keypress_count,
                "mouse_click_count": self._mouse_click_count,
                "scroll_delta": self._scroll_delta,
                "mouse_distance_px": round(self._mouse_distance_px, 2),
                "input_active_s": round(min(self._input_active_s, duration_s), 2),
                "interaction_burst_count": self._interaction_burst_count,
            }
            self._keypress_count = 0
            self._mouse_click_count = 0
            self._scroll_delta = 0
            self._mouse_distance_px = 0.0
            self._input_active_s = 0.0
            self._interaction_burst_count = 0
            self._last_interaction_at = None
            return bucket


class MouseInputState:
    _BUTTON_PAIRS = ((0x0001, 0x0002), (0x0004, 0x0008), (0x0010, 0x0020))

    def __init__(self) -> None:
        self._pressed_buttons = 0

    def process(
        self,
        *,
        mouse_flags: int,
        button_flags: int,
        last_x: int,
        last_y: int,
    ) -> dict[str, int]:
        is_absolute = bool(mouse_flags & 0x0001)
        if is_absolute:
            return {"click_count": 0, "move_x": 0, "move_y": 0}
        click_count = 0
        for button_index, (down_flag, up_flag) in enumerate(self._BUTTON_PAIRS):
            button_mask = 1 << button_index
            if button_flags & down_flag and not self._pressed_buttons & button_mask:
                self._pressed_buttons |= button_mask
                click_count += 1
            if button_flags & up_flag:
                self._pressed_buttons &= ~button_mask
        return {
            "click_count": click_count,
            "move_x": last_x,
            "move_y": last_y,
        }


def run_raw_input_watcher(
    output_queue: Any,
    stop_event: Any,
    bucket_seconds: int = 30,
) -> None:
    if os.name != "nt":
        output_queue.put({"error": "windows_only"})
        return

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    accumulator = InteractionAccumulator()
    mouse_state = MouseInputState()
    bucket_started = datetime.now(UTC)

    wm_input = 0x00FF
    wm_timer = 0x0113
    wm_destroy = 0x0002
    rid_input = 0x10000003
    ridev_inputsink = 0x00000100
    rim_type_mouse = 0
    rim_type_keyboard = 1
    ri_mouse_wheel = 0x0400
    wm_keydown = 0x0100
    wm_syskeydown = 0x0104

    class RAWINPUTDEVICE(ctypes.Structure):
        _fields_ = [
            ("usUsagePage", wintypes.USHORT),
            ("usUsage", wintypes.USHORT),
            ("dwFlags", wintypes.DWORD),
            ("hwndTarget", wintypes.HWND),
        ]

    class RAWINPUTHEADER(ctypes.Structure):
        _fields_ = [
            ("dwType", wintypes.DWORD),
            ("dwSize", wintypes.DWORD),
            ("hDevice", wintypes.HANDLE),
            ("wParam", wintypes.WPARAM),
        ]

    class RAWMOUSEBUTTONS(ctypes.Structure):
        _fields_ = [("usButtonFlags", wintypes.USHORT), ("usButtonData", wintypes.USHORT)]

    class RAWMOUSEUNION(ctypes.Union):
        _fields_ = [("ulButtons", wintypes.ULONG), ("buttons", RAWMOUSEBUTTONS)]

    class RAWMOUSE(ctypes.Structure):
        _fields_ = [
            ("usFlags", wintypes.USHORT),
            ("button_union", RAWMOUSEUNION),
            ("ulRawButtons", wintypes.ULONG),
            ("lLastX", wintypes.LONG),
            ("lLastY", wintypes.LONG),
            ("ulExtraInformation", wintypes.ULONG),
        ]

    class RAWKEYBOARD(ctypes.Structure):
        _fields_ = [
            ("MakeCode", wintypes.USHORT),
            ("Flags", wintypes.USHORT),
            ("Reserved", wintypes.USHORT),
            ("VKey", wintypes.USHORT),
            ("Message", wintypes.UINT),
            ("ExtraInformation", wintypes.ULONG),
        ]

    class RAWDATA(ctypes.Union):
        _fields_ = [("mouse", RAWMOUSE), ("keyboard", RAWKEYBOARD)]

    class RAWINPUT(ctypes.Structure):
        _fields_ = [("header", RAWINPUTHEADER), ("data", RAWDATA)]

    lresult_type = ctypes.c_ssize_t
    wndproc_type = ctypes.WINFUNCTYPE(
        lresult_type,
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    )

    def flush_bucket() -> None:
        nonlocal bucket_started
        now = datetime.now(UTC)
        duration = max(1.0, (now - bucket_started).total_seconds())
        output_queue.put({
            "window_start_utc": bucket_started.isoformat(),
            **accumulator.snapshot_and_reset(duration),
        })
        bucket_started = now

    def window_proc(hwnd: int, message: int, wparam: int, lparam: int) -> int:
        if message == wm_input:
            size = wintypes.UINT(0)
            header_size = ctypes.sizeof(RAWINPUTHEADER)
            user32.GetRawInputData(lparam, rid_input, None, ctypes.byref(size), header_size)
            buffer = ctypes.create_string_buffer(size.value)
            if user32.GetRawInputData(
                lparam,
                rid_input,
                buffer,
                ctypes.byref(size),
                header_size,
            ) == size.value:
                raw = ctypes.cast(buffer, ctypes.POINTER(RAWINPUT)).contents
                if raw.header.dwType == rim_type_keyboard:
                    if raw.data.keyboard.Message in (wm_keydown, wm_syskeydown):
                        accumulator.record_key()
                        accumulator.record_activity(0.1)
                elif raw.header.dwType == rim_type_mouse:
                    mouse = raw.data.mouse
                    flags = mouse.button_union.buttons.usButtonFlags
                    packet = mouse_state.process(
                        mouse_flags=mouse.usFlags,
                        button_flags=flags,
                        last_x=mouse.lLastX,
                        last_y=mouse.lLastY,
                    )
                    if packet["move_x"] or packet["move_y"]:
                        accumulator.record_move(packet["move_x"], packet["move_y"])
                    for _ in range(packet["click_count"]):
                        accumulator.record_click()
                    if (
                        mouse.lLastX
                        or mouse.lLastY
                        or packet["click_count"]
                    ):
                        accumulator.record_activity(0.1)
                    if flags & ri_mouse_wheel:
                        delta = ctypes.c_short(
                            mouse.button_union.buttons.usButtonData
                        ).value
                        accumulator.record_scroll(delta)
                        accumulator.record_activity(0.1)
            return 0
        if message == wm_timer:
            if stop_event.is_set():
                flush_bucket()
                user32.DestroyWindow(hwnd)
            else:
                flush_bucket()
            return 0
        if message == wm_destroy:
            user32.PostQuitMessage(0)
            return 0
        return int(user32.DefWindowProcW(hwnd, message, wparam, lparam))

    callback = wndproc_type(window_proc)

    class WNDCLASS(ctypes.Structure):
        _fields_ = [
            ("style", wintypes.UINT),
            ("lpfnWndProc", wndproc_type),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", wintypes.HINSTANCE),
            ("hIcon", wintypes.HICON),
            ("hCursor", wintypes.HANDLE),
            ("hbrBackground", wintypes.HBRUSH),
            ("lpszMenuName", wintypes.LPCWSTR),
            ("lpszClassName", wintypes.LPCWSTR),
        ]

    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HINSTANCE
    user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASS)]
    user32.RegisterClassW.restype = wintypes.ATOM
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HWND,
        wintypes.HMENU,
        wintypes.HINSTANCE,
        wintypes.LPVOID,
    ]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.DefWindowProcW.argtypes = [
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    ]
    user32.DefWindowProcW.restype = lresult_type
    user32.GetRawInputData.argtypes = [
        wintypes.HANDLE,
        wintypes.UINT,
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.UINT),
        wintypes.UINT,
    ]
    user32.GetRawInputData.restype = wintypes.UINT
    user32.RegisterRawInputDevices.argtypes = [
        ctypes.POINTER(RAWINPUTDEVICE),
        wintypes.UINT,
        wintypes.UINT,
    ]
    user32.RegisterRawInputDevices.restype = wintypes.BOOL
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.DestroyWindow.restype = wintypes.BOOL
    user32.SetTimer.argtypes = [wintypes.HWND, ctypes.c_size_t, wintypes.UINT, wintypes.LPVOID]
    user32.SetTimer.restype = ctypes.c_size_t
    user32.GetMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG),
        wintypes.HWND,
        wintypes.UINT,
        wintypes.UINT,
    ]
    user32.GetMessageW.restype = wintypes.BOOL
    user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.TranslateMessage.restype = wintypes.BOOL
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.restype = lresult_type

    class_name = f"MindFlowRawInputWindow-{os.getpid()}"
    instance = kernel32.GetModuleHandleW(None)
    window_class = WNDCLASS()
    window_class.lpfnWndProc = callback
    window_class.hInstance = instance
    window_class.lpszClassName = class_name
    if not user32.RegisterClassW(ctypes.byref(window_class)):
        output_queue.put({"error": "register_class_failed"})
        return

    pointer_mask = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1
    hwnd_message = wintypes.HWND((-3) & pointer_mask)
    hwnd = user32.CreateWindowExW(
        0,
        class_name,
        class_name,
        0,
        0,
        0,
        0,
        0,
        hwnd_message,
        None,
        instance,
        None,
    )
    if not hwnd:
        output_queue.put({"error": "create_window_failed"})
        return

    devices = (RAWINPUTDEVICE * 2)(
        RAWINPUTDEVICE(0x01, 0x02, ridev_inputsink, hwnd),
        RAWINPUTDEVICE(0x01, 0x06, ridev_inputsink, hwnd),
    )
    if not user32.RegisterRawInputDevices(
        devices,
        2,
        ctypes.sizeof(RAWINPUTDEVICE),
    ):
        output_queue.put({"error": "register_raw_input_failed"})
        user32.DestroyWindow(hwnd)
        return

    user32.SetTimer(hwnd, 1, bucket_seconds * 1000, None)
    message = wintypes.MSG()
    output_queue.put({"status": "running"})
    while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
        user32.TranslateMessage(ctypes.byref(message))
        user32.DispatchMessageW(ctypes.byref(message))


def read_queue_nowait(input_queue: Any) -> dict[str, Any] | None:
    try:
        item = input_queue.get_nowait()
    except queue.Empty:
        return None
    return item if isinstance(item, dict) else None
