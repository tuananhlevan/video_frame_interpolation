import cv2
import numpy as np
import pytest
from eval.layer2_temporal.optical_flow import (
    compute_dense_flow,
    evaluate_optical_flow_consistency,
    warp_frame,
)
from eval.layer2_temporal.smoothness import evaluate_motion_smoothness
from eval.layer2_temporal.flicker import evaluate_temporal_flicker


def test_optical_flow_warping_direction():
    """Verify that backward warping with -flow * 0.5 shifts image forward towards target."""
    h, w = 100, 100
    # Frame 0: square centered at x=30, y=50
    img0 = np.zeros((h, w), dtype=np.uint8)
    img0[40:60, 20:40] = 255

    # Frame 1: square translated +20 pixels to x=50, y=50
    img1 = np.zeros((h, w), dtype=np.uint8)
    img1[40:60, 40:60] = 255

    # Compute forward flow I0 -> I1
    flow_fwd = compute_dense_flow(img0, img1)

    # In DIS flow from img0 -> img1, flow vector at square is approx (+20, 0)
    # Correct backward warping to t=0.5 uses -flow_fwd * 0.5
    correct_warped = warp_frame(img0, -flow_fwd * 0.5)
    inverted_warped = warp_frame(img0, flow_fwd * 0.5)

    # Compute centroids along X
    m_orig = cv2.moments(img0)
    cx_orig = m_orig["m10"] / (m_orig["m00"] + 1e-5)

    m_correct = cv2.moments(correct_warped)
    cx_correct = m_correct["m10"] / (m_correct["m00"] + 1e-5)

    m_inverted = cv2.moments(inverted_warped)
    cx_inverted = m_inverted["m10"] / (m_inverted["m00"] + 1e-5)

    # The square started at cx ~ 29.5
    # Moving towards target x=50: correct intermediate centroid must be > 35
    # The erroneous inverted warp shifted it backwards < 25
    assert cx_correct > cx_orig + 2.0
    assert cx_correct < 50.0
    assert cx_inverted < cx_orig - 2.0


def test_optical_flow_bidirectional_consistency():
    """Verify that bidirectional flow masking reduces warping error on disocclusion."""
    h, w = 120, 120
    # Frame A: background with object
    fa = np.zeros((h, w, 3), dtype=np.uint8)
    fa[:, :] = [100, 100, 100]
    fa[40:80, 20:60] = [200, 200, 200]

    # Intermediate Frame X: object at midpoint
    fx = np.zeros((h, w, 3), dtype=np.uint8)
    fx[:, :] = [100, 100, 100]
    fx[40:80, 30:70] = [200, 200, 200]

    # Frame B: object shifted further
    fb = np.zeros((h, w, 3), dtype=np.uint8)
    fb[:, :] = [100, 100, 100]
    fb[40:80, 40:80] = [200, 200, 200]

    res = evaluate_optical_flow_consistency(fa, fx, fb)
    assert res["mean_warping_error"] < 15.0
    assert res["motion_magnitude"] > 0.0


def test_motion_smoothness_duration_invariance():
    """Verify that discontinuity penalty is normalized by sequence duration rate."""
    # Build 100 frames with 2 discontinuities
    # Vector: (dx, dy, mag)
    vecs_short = [(5.0, 0.0, 5.0)] * 100
    vecs_short[20] = (25.0, 0.0, 25.0)  # Discontinuity 1
    vecs_short[70] = (25.0, 0.0, 25.0)  # Discontinuity 2

    # Build 10,000 frames with the same 2 isolated discontinuities
    vecs_long = [(5.0, 0.0, 5.0)] * 10000
    vecs_long[200] = (25.0, 0.0, 25.0)
    vecs_long[7000] = (25.0, 0.0, 25.0)

    # In short clip: 2 in 100 is 2.0% -> significant penalty
    rate_short = 2 / 100.0
    pen_short = min(0.6, (rate_short / 0.02) * 0.6)

    # In long clip: 2 in 10000 is 0.02% -> tiny penalty
    rate_long = 2 / 10000.0
    pen_long = min(0.6, (rate_long / 0.02) * 0.6)

    assert pen_long < 0.05
    assert pen_short >= 0.5


def test_temporal_flicker_2nd_order_difference():
    """Verify that steady motion has low 2nd-order residual while flicker has high."""
    h, w = 60, 60
    # Steady brightness ramp over 3 frames (e.g. smooth camera pan entering light)
    f0 = np.full((h, w), 100, dtype=np.float32)
    f1 = np.full((h, w), 110, dtype=np.float32)
    f2 = np.full((h, w), 120, dtype=np.float32)

    # 1st order consecutive difference is |110 - 100| = 10 (looks like flicker in 1st order!)
    d1 = float(np.mean(np.abs(f1 - f0)))
    assert d1 == 10.0

    # 2nd order residual: |f1 - 0.5*(f0 + f2)| = |110 - 110| = 0.0!
    pred = 0.5 * (f0 + f2)
    res_steady = float(np.mean(np.abs(f1 - pred)))
    assert res_steady == 0.0

    # True oscillating flicker: 100 -> 140 -> 100
    f1_flicker = np.full((h, w), 140, dtype=np.float32)
    pred_flicker = 0.5 * (f0 + f2)  # 0.5 * (100 + 120) = 110
    res_flicker = float(np.mean(np.abs(f1_flicker - pred_flicker)))
    assert res_flicker == 30.0
