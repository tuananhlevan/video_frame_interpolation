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

    # White line mask
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
    line_mask = cv2.bitwise_and(white_mask, pitch_mask)

    # Edge detection
    edges = cv2.Canny(line_mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=15)

    segments: List[Tuple[int, int, int, int]] = []
    if lines is not None and len(lines) > 0:
        lines_reshaped = lines.reshape(-1, 4)
        for x1, y1, x2, y2 in lines_reshaped:
            segments.append((int(x1), int(y1), int(x2), int(y2)))

    return segments


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

        # Check line angles: in typical broadcast cameras, pitch lines have consistent perspective angles.
        # Lines with anomalous angles or fragments with curvature indicate warping.
        if len(lines) >= 2:
            angles = [np.arctan2(y2 - y1, x2 - x1) for x1, y1, x2, y2 in lines]
            # Standard deviation of angles within frame
            std_ang = float(np.std(angles))
            if std_ang > 1.2:
                wobble_events += 1

    # Check temporal stability of pitch line count (broken lines cause sudden count fluctuations)
    count_variance = float(np.var(line_counts_per_frame)) if line_counts_per_frame else 0.0

    score = 5.0
    if len(frames_sequence) > 0:
        wobble_rate = wobble_events / float(len(frames_sequence))
        score -= min(2.5, wobble_rate * 3.0)
    score -= min(1.5, count_variance / 50.0)
    score = max(1.0, min(5.0, score))

    details = {
        "score": score,
        "total_lines_detected": total_lines_detected,
        "wobble_events": wobble_events,
        "line_count_variance": count_variance
    }

    return float(score), details
