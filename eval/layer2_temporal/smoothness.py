"""Motion Smoothness and Trajectory Continuity evaluation."""

from typing import Dict, List, Tuple
import cv2
import numpy as np


def compute_frame_motion_vector(frame1: np.ndarray, frame2: np.ndarray) -> Tuple[float, float, float]:
    """Computes global translation (dx, dy) and mean motion magnitude between two frames.
    
    Uses phase correlation for global translation and optical flow magnitude.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if frame1.ndim == 3 else frame1
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if frame2.ndim == 3 else frame2

    # Phase correlation for global displacement
    shift, response = cv2.phaseCorrelate(gray1.astype(np.float32), gray2.astype(np.float32))
    dx, dy = shift
    global_mag = float(np.sqrt(dx ** 2 + dy ** 2))
    return float(dx), float(dy), global_mag


def evaluate_motion_smoothness(
    frames_sequence: List[np.ndarray],
    discontinuity_ratio_threshold: float = 2.5
) -> Tuple[float, int, List[int], Dict[str, float]]:
    """Evaluates motion continuity across consecutive frames (e.g. 50 fps output).
    
    Checks for sudden jerk / acceleration anomalies between t-1, t, and t+1.
    
    Returns:
        (smoothness_score [0, 1], discontinuity_count, anomaly_indices, stats)
    """
    if len(frames_sequence) < 3:
        return 1.0, 0, [], {"mean_acceleration": 0.0, "max_acceleration": 0.0}

    motion_vectors: List[Tuple[float, float, float]] = []
    for i in range(len(frames_sequence) - 1):
        vec = compute_frame_motion_vector(frames_sequence[i], frames_sequence[i + 1])
        motion_vectors.append(vec)

    # Compute acceleration (change in motion vector)
    accelerations: List[float] = []
    discontinuity_indices: List[int] = []

    for i in range(len(motion_vectors) - 1):
        dx1, dy1, mag1 = motion_vectors[i]
        dx2, dy2, mag2 = motion_vectors[i + 1]

        # Delta in motion vector
        acc = float(np.sqrt((dx2 - dx1) ** 2 + (dy2 - dy1) ** 2))
        accelerations.append(acc)

        # Flag sudden jerk where motion jumps abruptly compared to neighbors
        base_mag = max(1.0, (mag1 + mag2) / 2.0)
        if acc > discontinuity_ratio_threshold * base_mag and acc > 3.0:
            discontinuity_indices.append(i + 1)

    mean_acc = float(np.mean(accelerations)) if accelerations else 0.0
    max_acc = float(np.max(accelerations)) if accelerations else 0.0

    # Smoothness score in [0.0, 1.0]: 1.0 is smooth, decreases with discontinuities and high jerk
    discontinuity_rate = len(discontinuity_indices) / float(max(1, len(motion_vectors)))
    discontinuity_penalty = min(0.6, (discontinuity_rate / 0.02) * 0.6)
    jerk_penalty = min(0.4, mean_acc / 10.0)
    smoothness_score = max(0.0, 1.0 - discontinuity_penalty - jerk_penalty)

    stats = {
        "mean_acceleration": mean_acc,
        "max_acceleration": max_acc,
        "discontinuity_count": len(discontinuity_indices)
    }

    return float(smoothness_score), len(discontinuity_indices), discontinuity_indices, stats
