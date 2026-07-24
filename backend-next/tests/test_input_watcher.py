from __future__ import annotations

import multiprocessing
import os
import queue
import time

import pytest

from mindflow.infrastructure.collectors.input_watcher import (
    InteractionAccumulator,
    MouseInputState,
    run_raw_input_watcher,
)


def test_accumulator_emits_only_aggregate_metrics() -> None:
    accumulator = InteractionAccumulator()
    accumulator.record_key()
    accumulator.record_click()
    accumulator.record_scroll(120)
    accumulator.record_move(3, 4)
    accumulator.record_activity(0.5)

    bucket = accumulator.snapshot_and_reset(duration_s=30.0)

    assert bucket == {
        "duration_s": 30.0,
        "keypress_count": 1,
        "mouse_click_count": 1,
        "scroll_delta": 120,
        "mouse_distance_px": 5.0,
        "input_active_s": 0.5,
        "interaction_burst_count": 1,
    }
    assert not {"key", "key_code", "x", "y", "coordinates"} & bucket.keys()


def test_accumulator_resets_between_buckets() -> None:
    accumulator = InteractionAccumulator()
    accumulator.record_click()
    accumulator.snapshot_and_reset(duration_s=30.0)

    second = accumulator.snapshot_and_reset(duration_s=30.0)

    assert second["mouse_click_count"] == 0
    assert second["interaction_burst_count"] == 0


@pytest.mark.skipif(os.name != "nt", reason="Windows Raw Input only")
def test_raw_input_watcher_starts_and_stops_on_windows() -> None:
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    stop_event = context.Event()
    process = context.Process(
        target=run_raw_input_watcher,
        args=(output, stop_event, 1),
    )
    process.start()
    messages: list[dict[str, object]] = []
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            messages.append(output.get(timeout=0.5))
        except queue.Empty:
            if not process.is_alive():
                break
        if any(message.get("status") == "running" for message in messages):
            break
    stop_event.set()
    process.join(4)
    if process.is_alive():
        process.terminate()
        process.join(2)

    assert process.exitcode == 0
    assert any(message.get("status") == "running" for message in messages)


def test_mouse_state_ignores_absolute_motion_and_deduplicates_button_down() -> None:
    state = MouseInputState()

    first = state.process(
        mouse_flags=0x0001,
        button_flags=0x0001,
        last_x=50000,
        last_y=40000,
    )
    repeated = state.process(
        mouse_flags=0x0001,
        button_flags=0x0001,
        last_x=51000,
        last_y=41000,
    )
    state.process(mouse_flags=0, button_flags=0x0002, last_x=0, last_y=0)
    next_click = state.process(mouse_flags=0, button_flags=0x0001, last_x=3, last_y=4)

    assert first == {"click_count": 0, "move_x": 0, "move_y": 0}
    assert repeated == {"click_count": 0, "move_x": 0, "move_y": 0}
    assert next_click == {"click_count": 1, "move_x": 3, "move_y": 4}
