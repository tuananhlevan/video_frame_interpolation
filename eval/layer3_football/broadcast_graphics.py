"""Broadcast Graphics (scoreboard, watermark/logo, LED boards) evaluation."""

from typing import Dict, List, Tuple
import cv2
import numpy as np


def evaluate_broadcast_graphics(
    frames_sequence: List[np.ndarray]
) -> Tuple[float, Dict[str, any]]:
    """Evaluates stability of static broadcast graphics, scoreboards, and TV watermarks.
    
    Verifies temporal persistence first to distinguish genuine fixed graphics
    from moving background video (e.g. spectator crowd in grandstands during camera pans).

    Returns:
        (graphics_score [1.0, 5.0], details)
    """
    if len(frames_sequence) < 3:
        return 5.0, {"score": 5.0, "note": "Insufficient frames", "graphics_detected": False}

    h, w = frames_sequence[0].shape[:2]

    # Typical broadcast graphics locations:
    # 1. Top-left: Scoreboard / Match clock (y: 0 to 18% H, x: 0 to 30% W)
    # 2. Top-right: Channel logo / Watermark (y: 0 to 15% H, x: 75% to 98% W)
    roi_top_left = (slice(0, int(h * 0.18)), slice(0, int(w * 0.30)))
    roi_top_right = (slice(0, int(h * 0.15)), slice(int(w * 0.75), int(w * 0.98)))

    rois = [("top_left_scorebug", roi_top_left), ("top_right_watermark", roi_top_right)]

    roi_jitter_scores: List[float] = []
    detected_graphics_count = 0

    for name, (ys, xs) in rois:
        # Sample patches across sequence
        sample_step = max(1, len(frames_sequence) // 40)
        patches = [
            cv2.cvtColor(f[ys, xs], cv2.COLOR_BGR2GRAY)
            for f in frames_sequence[::sample_step]
        ]
        if not patches:
            continue

        # Check temporal edge persistence across screen space:
        # Genuine graphics have edges that stay fixed at the exact same pixel coordinates
        # across > 60% of frames. Moving crowds or pitch have zero persistent pixels.
        edge_maps = [(cv2.Canny(p, 80, 180) > 0).astype(np.float32) for p in patches]
        mean_edge = np.mean(edge_maps, axis=0)
        persistent_mask = (mean_edge >= 0.60)
        persistent_edge_count = int(np.sum(persistent_mask))

        if persistent_edge_count < 40:
            # Clean feed / moving crowd: no static graphic in this ROI
            continue

        detected_graphics_count += 1

        # Evaluate jitter strictly on the persistent static graphic edges
        diffs = []
        for i in range(len(frames_sequence) - 1):
            patch1 = cv2.cvtColor(frames_sequence[i][ys, xs], cv2.COLOR_BGR2GRAY).astype(np.float32)
            patch2 = cv2.cvtColor(frames_sequence[i + 1][ys, xs], cv2.COLOR_BGR2GRAY).astype(np.float32)
            edge_diff = float(np.mean(np.abs(patch1[persistent_mask] - patch2[persistent_mask])))
            diffs.append(edge_diff)

        if diffs:
            roi_jitter_scores.append(float(np.mean(diffs)))

    if detected_graphics_count == 0:
        return 5.0, {
            "score": 5.0,
            "mean_graphics_jitter": 0.0,
            "graphics_detected": False,
            "note": "No static broadcast graphics detected (clean feed / crowd)"
        }

    avg_jitter = float(np.mean(roi_jitter_scores)) if roi_jitter_scores else 0.0

    # Score: 5.0 (immobile & crisp), degrades if jitter > 3.0
    score = 5.0
    if avg_jitter > 3.0:
        score -= min(3.0, (avg_jitter - 3.0) * 0.4)
    score = max(1.0, min(5.0, score))

    details = {
        "score": round(score, 2),
        "mean_graphics_jitter": round(avg_jitter, 2),
        "graphics_detected": True,
        "jitter_detected": avg_jitter > 5.0
    }

    return float(score), details
