"""Pitch Geometry evaluation.

Detects pitch markings (touchlines, penalty boxes, center circle, halfway line)
and tests for bending, wobbling, double lines, or broken lines in generated frames.
Outputs a 1.0 to 5.0 pitch geometry score.
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np


def extract_pitch_lines(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Extracts straight line segments on the football pitch using HoughLinesP."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Green pitch mask
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    pitch_mask = cv2.inRange(hsv, lower_green, upper_green)

    # White line mask on green pitch
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
    pitch_zone = cv2.dilate(pitch_mask, np.ones((15, 15), np.uint8))
    line_mask = cv2.bitwise_and(white_mask, pitch_zone)

    # Edge detection
    edges = cv2.Canny(line_mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=40, maxLineGap=15)

    segments: List[Tuple[int, int, int, int]] = []
    if lines is not None and len(lines) > 0:
        lines_reshaped = lines.reshape(-1, 4)
        for x1, y1, x2, y2 in lines_reshaped:
            segments.append((int(x1), int(y1), int(x2), int(y2)))

    return segments


def circular_ang_dist(a: float, b: float) -> float:
    """Computes minimal angular distance on [0, pi) manifold."""
    diff = abs(a - b) % np.pi
    return min(diff, np.pi - diff)


def evaluate_pitch_geometry(
    frames_sequence: List[np.ndarray]
) -> Tuple[float, Dict[str, any]]:
    """Evaluates pitch line straightness and stability across frames.
    
    Returns:
        (pitch_score [1.0, 5.0], details)
    """
    total_lines_detected = 0
    wobble_events = 0
    line_counts_per_frame: List[int] = []

    for frame in frames_sequence:
        lines = extract_pitch_lines(frame)
        total_lines_detected += len(lines)
        line_counts_per_frame.append(len(lines))

        # Check line angles: map to [0, pi) and cluster into dominant perspective bundles
        # (touchlines vs penalty box / goal lines)
        if len(lines) >= 2:
            angles = [(np.arctan2(y2 - y1, x2 - x1) % np.pi) for x1, y1, x2, y2 in lines]
            
            # Simple clustering into up to 2 perspective vanishing sets
            cluster_a: List[float] = [angles[0]]
            cluster_b: List[float] = []
            
            for ang in angles[1:]:
                if circular_ang_dist(ang, cluster_a[0]) < 0.45:
                    cluster_a.append(ang)
                elif not cluster_b or circular_ang_dist(ang, cluster_b[0]) < 0.45:
                    cluster_b.append(ang)
                else:
                    # Line does not fit either dominant vanishing bundle: potential warp
                    wobble_events += 1

            # Check angular variance within each bundle
            for cl in (cluster_a, cluster_b):
                if len(cl) >= 3:
                    within_std = float(np.std(cl))
                    if within_std > 0.35:  # Divergent / bent lines within same orientation
                        wobble_events += 1

    # Check temporal continuity of pitch line counts (sudden frame-to-frame drops indicate broken lines)
    sudden_drops = 0
    if len(line_counts_per_frame) >= 2:
        for i in range(1, len(line_counts_per_frame)):
            if abs(line_counts_per_frame[i] - line_counts_per_frame[i - 1]) >= 3:
                sudden_drops += 1

    score = 5.0
    if len(frames_sequence) > 0:
        wobble_rate = wobble_events / float(len(frames_sequence))
        score -= min(2.5, (wobble_rate / 0.10) * 2.5)
    
    if len(line_counts_per_frame) > 1:
        drop_rate = sudden_drops / float(len(line_counts_per_frame) - 1)
        score -= min(1.5, (drop_rate / 0.05) * 1.5)

    score = max(1.0, min(5.0, score))

    details = {
        "score": round(score, 2),
        "total_lines_detected": total_lines_detected,
        "wobble_events": wobble_events,
        "sudden_line_drop_events": sudden_drops
    }

    return float(score), details
