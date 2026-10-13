"""Every frame's reported timestamp must match the moment it was taken from.

Bug (found reviewing a 17-minute booth loop, 2026-10-10): uniform frames were
labelled ``start + i / fps``, but ffmpeg's ``fps`` filter does not emit the frame
at the start of each interval. It rounds input timestamps to the nearest output
tick and emits a frame from the middle of each interval. So every uniform frame
was labelled roughly half an interval early: ~10 s on that video, which put a
game's intro at 0:21, ten seconds before it played. A timestamp the code
*computes* from an assumption about ffmpeg is a claim. The fix *reads* each
frame's real pts from ``showinfo``, as the scene and keyframe engines already do.

The oracle is ``clock_clip``: second k is solid grey 16 + 5k, so a frame's own
pixels say which second it came from, independently of anything under test.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import frames
from conftest import CLOCK_SECONDS, CLOCK_STEP

FRAME = 0.11   # one frame at the clip's 10 fps, plus rounding


def _pixel_second(jpeg: str) -> int:
    """Which second of the clock clip this frame shows, read from its pixels."""
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", jpeg,
         "-vf", "scale=8:8,format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    ).stdout
    luma = sum(raw) / len(raw)
    return round((luma - 16) / CLOCK_STEP)


def _assert_labels_true(frames_out: list[dict]) -> None:
    assert frames_out, "engine returned no frames"
    wrong = []
    for fr in frames_out:
        k = _pixel_second(fr["path"])
        t = fr["timestamp_seconds"]
        # The frame came from somewhere inside second k: [k, k+1).
        if not (k - FRAME <= t < k + 1 + FRAME):
            wrong.append(f"labelled {t:.2f}s but shows second {k}")
    assert not wrong, "mislabelled frames: " + "; ".join(wrong)


def test_clock_clip_oracle_works(clock_clip: Path, tmp_path: Path):
    """Positive control: an exact seek must read back as the second it targeted,
    or a passing test below would prove nothing."""
    out, _ = frames.extract_at_timestamps(str(clock_clip), tmp_path, [0.5, 7.5, 23.5, 38.5])
    assert [_pixel_second(f["path"]) for f in out] == [0, 7, 23, 38]


def test_uniform_labels_match_the_frames_shown(clock_clip: Path, tmp_path: Path):
    """5 frames over 40 s is an 8 s interval, so the old half-interval error was ~4 s."""
    out = frames.extract(str(clock_clip), tmp_path, fps=5 / CLOCK_SECONDS, max_frames=5)
    _assert_labels_true(out)


def test_uniform_labels_are_absolute_inside_a_focus_window(clock_clip: Path, tmp_path: Path):
    """With --start/--end, labels must be source time, not offset-from-start."""
    out = frames.extract(str(clock_clip), tmp_path, fps=0.5, max_frames=10,
                         start_seconds=10.0, end_seconds=30.0)
    _assert_labels_true(out)
    assert all(10.0 - FRAME <= f["timestamp_seconds"] <= 30.0 + FRAME for f in out)


def test_uniform_fallback_labels_are_true(clock_clip: Path, tmp_path: Path):
    """The pure-uniform fallback path (too few cuts) goes through extract()."""
    out, meta = frames.extract_scene_or_uniform(
        str(clock_clip), tmp_path, fps=0.25, target_frames=10, max_frames=10,
    )
    assert meta["engine"] in ("uniform", "scene+uniform", "scene")
    _assert_labels_true(out)


def test_keyframe_labels_are_true(clock_clip: Path, tmp_path: Path):
    """Control for an engine that already reads pts from showinfo."""
    out, _ = frames.extract_keyframes(str(clock_clip), tmp_path, max_frames=20)
    _assert_labels_true(out)


@pytest.mark.parametrize("fps,max_frames", [(1.0, 40), (0.1, 4), (2.0, 80)])
def test_uniform_labels_hold_across_densities(clock_clip: Path, tmp_path: Path, fps, max_frames):
    """The error scaled with the interval, so check sparse and dense sampling."""
    out = frames.extract(str(clock_clip), tmp_path, fps=fps, max_frames=max_frames)
    _assert_labels_true(out)


def test_uniform_labels_are_chronological(clock_clip: Path, tmp_path: Path):
    out = frames.extract(str(clock_clip), tmp_path, fps=0.5, max_frames=20)
    times = [f["timestamp_seconds"] for f in out]
    assert times == sorted(times) and len(set(times)) == len(times)
