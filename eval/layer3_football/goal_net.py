"""Goal and Net Integrity evaluation."""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


def detect_goal_candidate_roi(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detects if a goal structure (posts, crossbar, net mesh) is present in the frame.
    
    Returns:
        (x, y, w, h) bounding box of goal region if found, else None
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape[:2]

    # Goals are white structures typically near the left, right, or center rear of the field
    _, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
    
    # Exclude lower 20% of screen where bottom graphics / near touchlines lie
    thresh[int(h * 0.85):, :] = 0

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    has_post = False
    has_crossbar = False
    goal_bboxes = []

    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < 100:
            continue
        
        is_post = (bh >= 70 and bw <= 25 and bh / max(1, bw) >= 3.5)
        is_crossbar = (bw >= 120 and bh <= 30 and bw / max(1, bh) >= 4.5)
        if is_post:
            has_post = True
            goal_bboxes.append((bx, by, bw, bh))
        elif is_crossbar:
            has_crossbar = True
            goal_bboxes.append((bx, by, bw, bh))

    is_goal = (has_crossbar and has_post) or (has_crossbar and len(goal_bboxes) >= 1) or (has_post and len(goal_bboxes) >= 2)

    if is_goal and goal_bboxes:
        # Union of goal bounding boxes expanded for net mesh
        min_x = max(0, min(b[0] for b in goal_bboxes) - 40)
        min_y = max(0, min(b[1] for b in goal_bboxes) - 20)
        max_x = min(w, max(b[0] + b[2] for b in goal_bboxes) + 80)
        max_y = min(h, max(b[1] + b[3] for b in goal_bboxes) + 40)
        return (min_x, min_y, max_x - min_x, max_y - min_y)

    return None


def evaluate_goal_net_integrity(
    frames_sequence: List[np.ndarray]
) -> Tuple[float, Dict[str, any]]:
    """Evaluates high-contrast thin structures (goalposts, crossbar, goal net mesh).
    
    Returns:
        (goal_net_score [1.0, 5.0], details)
    """
    if not frames_sequence:
        return 5.0, {"score": 5.0, "note": "No frames to evaluate", "goal_present": False}

    # First verify whether a goal is present in the sequence
    goal_rois = []
    for f in frames_sequence[::max(1, len(frames_sequence) // 10)]:
        roi = detect_goal_candidate_roi(f)
        if roi is not None:
            goal_rois.append(roi)

    if not goal_rois:
        # No goal on screen: no goal net artifacts possible -> pristine 5.0
        return 5.0, {
            "score": 5.0,
            "goal_present": False,
            "mean_mesh_crispness": 0.0,
            "note": "No goal structure detected in clip"
        }

    # A goal is confirmed present: evaluate high-frequency structural preservation
    # Average the detected goal ROI
    avg_gx = int(np.mean([r[0] for r in goal_rois]))
    avg_gy = int(np.mean([r[1] for r in goal_rois]))
    avg_gw = int(np.mean([r[2] for r in goal_rois]))
    avg_gh = int(np.mean([r[3] for r in goal_rois]))

    laplacian_variances: List[float] = []

    for frame in frames_sequence:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        crop = gray[avg_gy:avg_gy + avg_gh, avg_gx:avg_gx + avg_gw]
        if crop.size == 0:
            continue
        _, thresh = cv2.threshold(crop, 200, 255, cv2.THRESH_BINARY)
        lap = cv2.Laplacian(thresh, cv2.CV_32F)
        lap_var = float(lap.var())
        laplacian_variances.append(lap_var)

    mean_lap = float(np.mean(laplacian_variances)) if laplacian_variances else 0.0
    var_lap = float(np.var(laplacian_variances)) if laplacian_variances else 0.0

    score = 5.0
    # If high-frequency variance fluctuates wildly between even and odd frames, net is fluttering
    if len(laplacian_variances) >= 4:
        even_mean = float(np.mean(laplacian_variances[0::2]))
        odd_mean = float(np.mean(laplacian_variances[1::2]))
        diff_ratio = abs(even_mean - odd_mean) / (even_mean + 1e-5)
        if diff_ratio > 0.20:
            score -= min(2.5, diff_ratio * 4.0)

    score = max(1.0, min(5.0, score))

    details = {
        "score": round(score, 2),
        "goal_present": True,
        "mean_mesh_crispness": mean_lap,
        "mesh_energy_variance": var_lap
    }

    return float(score), details
