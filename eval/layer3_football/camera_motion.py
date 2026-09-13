"""Global Camera Motion continuity evaluation."""

from typing import Dict, List, Tuple
import cv2
import numpy as np


def evaluate_camera_motion_continuity(
    frames_sequence: List[np.ndarray]
) -> Tuple[float, Dict[str, any]]:
    """Evaluates continuity of global camera translation/panning across frames.
    
    Returns:
        (camera_score [1.0, 5.0], details)
    """
    if len(frames_sequence) < 3:
        return 4.5, {"score": 4.5, "note": "Insufficient frames"}

    shifts: List[Tuple[float, float]] = []

    for i in range(len(frames_sequence) - 1):
        g1 = cv2.cvtColor(frames_sequence[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(frames_sequence[i + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Global displacement via phase correlation
        shift, _ = cv2.phaseCorrelate(g1, g2)
        shifts.append((float(shift[0]), float(shift[1])))

    # Measure continuity: delta between consecutive pan vectors
    jitter_events = 0
    pan_magnitudes = [np.hypot(dx, dy) for dx, dy in shifts]
    mean_pan = float(np.mean(pan_magnitudes)) if pan_magnitudes else 0.0

    for i in range(len(shifts) - 1):
        dx1, dy1 = shifts[i]
        dx2, dy2 = shifts[i + 1]
        delta = np.hypot(dx2 - dx1, dy2 - dy1)
        # If pan direction abruptly reverses or jumps by > 2.0x mean pan
        if delta > max(2.5, mean_pan * 1.5):
            jitter_events += 1

    score = 5.0
    if len(shifts) > 1:
        jitter_rate = jitter_events / float(len(shifts))
        score -= min(3.0, jitter_rate * 8.0)
    score = max(1.0, min(5.0, score))

    details = {
        "score": score,
        "mean_camera_pan_speed": mean_pan,
        "camera_jitter_events": jitter_events
    }

    return float(score), details
