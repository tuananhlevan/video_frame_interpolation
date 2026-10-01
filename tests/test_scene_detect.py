import numpy as np
import pytest
from upframe.pipeline.scene_detect import (
    compute_frame_difference,
    is_scene_cut,
    is_broadcast_transition,
    detect_cuts
)


def test_identical_frames_have_zero_diff():
    f0 = np.full((100, 100, 3), 128, dtype=np.uint8)
    f1 = np.full((100, 100, 3), 128, dtype=np.uint8)
    diff = compute_frame_difference(f0, f1)
    assert diff == 0.0
    assert not is_scene_cut(f0, f1)
    assert not is_broadcast_transition(f0, f1)


def test_black_to_white_is_scene_cut():
    f0 = np.zeros((100, 100, 3), dtype=np.uint8)
    f1 = np.full((100, 100, 3), 255, dtype=np.uint8)
    diff = compute_frame_difference(f0, f1)
    assert diff >= 0.35
    assert is_scene_cut(f0, f1)
    assert not is_broadcast_transition(f0, f1)  # Cut takes precedence (diff >= 0.35)


def test_broadcast_transition_detection():
    # Intermediate difference typical of 3D animated stinger / graphic wipe / dissolve
    f0 = np.tile(np.linspace(0, 255, 100, dtype=np.uint8)[:, None, None], (1, 100, 3))
    f1 = np.roll(f0, shift=20, axis=0)
    diff = compute_frame_difference(f0, f1)
    assert 0.12 <= diff < 0.35
    assert not is_scene_cut(f0, f1)
    assert is_broadcast_transition(f0, f1)


def test_detect_cuts_list():
    f_black = np.zeros((50, 50, 3), dtype=np.uint8)
    f_white = np.full((50, 50, 3), 255, dtype=np.uint8)
    frames = [f_black, f_black, f_white, f_white]
    cuts = detect_cuts(frames)
    assert cuts == [1]
