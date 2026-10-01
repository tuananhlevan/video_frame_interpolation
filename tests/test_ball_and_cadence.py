"""Unit tests for slow-motion cadence filtering and ball trajectory refinement."""

import numpy as np
import pytest
from upframe.pipeline.cadence import is_duplicate_frame, CadenceHandler
from upframe.pipeline.ball_refiner import BallRefiner


def test_is_duplicate_frame():
    """Verify identical or near-identical frames are correctly flagged as duplicates."""
    f1 = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    f2 = f1.copy()
    assert is_duplicate_frame(f1, f2)

    # Slight noise below threshold
    f2_noisy = f1.copy()
    f2_noisy[0, 0, 0] = 101
    assert is_duplicate_frame(f1, f2_noisy)

    # Distinct frames
    f3 = np.ones((720, 1280, 3), dtype=np.uint8) * 150
    assert not is_duplicate_frame(f1, f3)


def test_cadence_handler():
    """Verify sequence cadence tracking."""
    handler = CadenceHandler()
    f_orig = np.zeros((100, 100, 3), dtype=np.uint8)
    f_dup = f_orig.copy()
    f_new = np.ones((100, 100, 3), dtype=np.uint8) * 80

    assert handler.check_pair(f_orig, f_dup) is True
    assert handler.consecutive_duplicates == 1

    assert handler.check_pair(f_dup, f_new) is False
    assert handler.consecutive_duplicates == 0


def test_ball_refiner_instantiation():
    """Verify BallRefiner initializes with expected thresholds and handles trivial frames."""
    refiner = BallRefiner(min_motion_threshold=20.0, max_motion_threshold=150.0)
    assert refiner.min_motion_threshold == 20.0

    # Test on uniform frames (no ball present) -> returns frame_inter unchanged
    f0 = np.zeros((300, 300, 3), dtype=np.uint8)
    f_inter = np.zeros((300, 300, 3), dtype=np.uint8)
    f1 = np.zeros((300, 300, 3), dtype=np.uint8)

    res = refiner.refine(f0, f_inter, f1, timestep=0.5)
    assert np.array_equal(res, f_inter)


def test_ball_refiner_rejects_shoes_and_socks():
    """Verify BallRefiner never pairs stationary player shoes/socks with balls or other socks."""
    refiner = BallRefiner(min_motion_threshold=25.0, max_motion_threshold=160.0)
    import cv2
    f0 = np.full((300, 300, 3), [40, 120, 40], dtype=np.uint8)
    f1 = np.full((300, 300, 3), [40, 120, 40], dtype=np.uint8)
    f_inter = np.full((300, 300, 3), [40, 120, 40], dtype=np.uint8)

    # Player 1's shoe at (100, 100) in F0 and (103, 100) in F1 (moves 3px)
    cv2.circle(f0, (100, 100), 6, (240, 240, 240), -1)
    cv2.circle(f1, (103, 100), 6, (240, 240, 240), -1)

    # Real ball at (200, 100) in F0 and (202, 100) in F1 (moves 2px)
    cv2.circle(f0, (200, 100), 6, (255, 255, 255), -1)
    cv2.circle(f1, (202, 100), 6, (255, 255, 255), -1)
    cv2.circle(f_inter, (201, 100), 6, (255, 255, 255), -1)

    # BallRefiner must NOT pair shoe (100, 100) with ball (202, 100) across 102px jump!
    res = refiner.refine(f0, f_inter, f1, timestep=0.5)
    assert np.array_equal(res, f_inter)
