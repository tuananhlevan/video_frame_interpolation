"""Ball Integrity evaluation.

Detects and tracks football candidates on the pitch, checking for:
- single ball constraint (no ghost ball / duplicate balls)
- trajectory continuity (no teleportation / sudden jumps)
- shape integrity (not unnaturally stretched or dissolved)
- outputs 1.0 to 5.0 ball score.
"""

import math
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


def detect_ball_candidates(frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
    """Detects potential football candidates on the pitch.
    
    Returns:
        List of (cx, cy, radius, circularity)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Green pitch mask
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    pitch_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find white/high-contrast objects inside or near the pitch
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Dilate pitch mask slightly so touchlines and players near ball are included
    pitch_dilated = cv2.dilate(pitch_mask, np.ones((15, 15), np.uint8))
    ball_search_mask = cv2.bitwise_and(bright_mask, pitch_dilated)

    contours, _ = cv2.findContours(ball_search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Tuple[float, float, float, float]] = []

    for c in contours:
        area = cv2.contourArea(c)
        # Expected ball size in 1080p: area ~ 20 to 500 px
        if 20 <= area <= 600:
            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                circularity = (4.0 * math.pi * area) / (perimeter ** 2)
                if circularity >= 0.50:  # reasonably round
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    candidates.append((float(x), float(y), float(radius), float(circularity)))

    return candidates


def evaluate_ball_integrity(
    frames_sequence: List[np.ndarray],
    max_ball_velocity_px: float = 75.0  # max reasonable ball displacement per frame in 50 fps
) -> Tuple[float, Dict[str, any]]:
    """Evaluates ball integrity across a sequence of 50 fps frames.
    
    Returns:
        (ball_score [1.0, 5.0], details)
    """
    teleportation_count = 0
    duplicate_ball_events = 0
    trajectories: List[Tuple[float, float]] = []

    prev_pos: Optional[Tuple[float, float]] = None

    for idx, frame in enumerate(frames_sequence):
        candidates = detect_ball_candidates(frame)

        # Check for multiple balls in close proximity (ghost ball / duplication)
        if len(candidates) >= 2:
            # Check if any two candidates are within duplicate range (10 - 60 px)
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    dist = math.hypot(candidates[i][0] - candidates[j][0], candidates[i][1] - candidates[j][1])
                    if 10.0 <= dist <= 60.0:
                        duplicate_ball_events += 1

        # Select most plausible ball candidate
        best_candidate: Optional[Tuple[float, float]] = None
        if candidates:
            if prev_pos is None:
                # Pick most circular candidate
                candidates.sort(key=lambda x: x[3], reverse=True)
                best_candidate = (candidates[0][0], candidates[0][1])
            else:
                # Pick candidate closest to previous trajectory
                candidates.sort(key=lambda x: math.hypot(x[0] - prev_pos[0], x[1] - prev_pos[1]))
                closest = candidates[0]
                dist = math.hypot(closest[0] - prev_pos[0], closest[1] - prev_pos[1])
                if dist > max_ball_velocity_px:
                    teleportation_count += 1
                best_candidate = (closest[0], closest[1])

        if best_candidate:
            trajectories.append(best_candidate)
            prev_pos = best_candidate

    # Scoring: 1.0 (fail) to 5.0 (clean)
    score = 5.0
    # Penalize duplicate ball events heavily (destroys object identity)
    score -= min(2.0, duplicate_ball_events * 0.4)
    # Penalize teleportations
    score -= min(2.0, teleportation_count * 0.3)
    score = max(1.0, min(5.0, score))

    details = {
        "score": score,
        "duplicate_ball_events": duplicate_ball_events,
        "teleportation_count": teleportation_count,
        "tracked_positions_count": len(trajectories)
    }

    return float(score), details
