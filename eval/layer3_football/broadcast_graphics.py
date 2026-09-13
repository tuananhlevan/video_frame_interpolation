"""Broadcast Graphics (scoreboard, watermark/logo, LED boards) evaluation."""

from typing import Dict, List, Tuple
import cv2
import numpy as np


def evaluate_broadcast_graphics(
    frames_sequence: List[np.ndarray]
) -> Tuple[float, Dict[str, any]]:
    """Evaluates stability of static broadcast graphics, scoreboards, and TV watermarks.
    
    Returns:
        (graphics_score [1.0, 5.0], details)
    """
    if len(frames_sequence) < 3:
        return 4.5, {"score": 4.5, "note": "Insufficient frames"}

    h, w = frames_sequence[0].shape[:2]

    # Typical broadcast graphics locations:
    # 1. Top-left: Scoreboard / Match clock (e.g. y: 0 to 18% H, x: 0 to 30% W)
    # 2. Top-right: Channel logo / Watermark (e.g. y: 0 to 15% H, x: 75% to 98% W)
    roi_top_left = (slice(0, int(h * 0.18)), slice(0, int(w * 0.30)))
    roi_top_right = (slice(0, int(h * 0.15)), slice(int(w * 0.75), int(w * 0.98)))

    rois = [("top_left_scorebug", roi_top_left), ("top_right_watermark", roi_top_right)]

    roi_jitter_scores: List[float] = []

    for name, (ys, xs) in rois:
        diffs = []
        for i in range(len(frames_sequence) - 1):
            patch1 = cv2.cvtColor(frames_sequence[i][ys, xs], cv2.COLOR_BGR2GRAY).astype(np.float32)
            patch2 = cv2.cvtColor(frames_sequence[i + 1][ys, xs], cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Mask out moving video content by focusing on static crisp text/edges
            edges1 = cv2.Canny(patch1.astype(np.uint8), 80, 180)
            if np.sum(edges1) > 100:  # has graphic edges
                edge_mask = edges1 > 0
                edge_diff = float(np.mean(np.abs(patch1[edge_mask] - patch2[edge_mask])))
                diffs.append(edge_diff)

        if diffs:
            mean_diff = float(np.mean(diffs))
            roi_jitter_scores.append(mean_diff)

    avg_jitter = float(np.mean(roi_jitter_scores)) if roi_jitter_scores else 2.0

    # Score: 5.0 (immobile & crisp), degrades if jitter > 8.0
    score = 5.0
    if avg_jitter > 3.0:
        score -= min(3.0, (avg_jitter - 3.0) * 0.4)
    score = max(1.0, min(5.0, score))

    details = {
        "score": score,
        "mean_graphics_jitter": avg_jitter,
        "jitter_detected": avg_jitter > 5.0
    }

    return float(score), details
