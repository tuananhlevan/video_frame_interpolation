"""Goal and Net Integrity evaluation."""

from typing import Dict, List, Tuple
import cv2
import numpy as np


def evaluate_goal_net_integrity(
    frames_sequence: List[np.ndarray]
) -> Tuple[float, Dict[str, any]]:
    """Evaluates high-contrast thin structures (goalposts, crossbar, goal net mesh).
    
    Returns:
        (goal_net_score [1.0, 5.0], details)
    """
    if not frames_sequence:
        return 4.5, {"score": 4.5, "note": "No frames to evaluate"}

    # Evaluate high-frequency structural preservation in thin edge regions
    laplacian_variances: List[float] = []

    for frame in frames_sequence:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        # Filter for high-contrast white structures (goalposts / net mesh)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        # Structural crispness
        lap = cv2.Laplacian(thresh, cv2.CV_32F)
        lap_var = float(lap.var())
        laplacian_variances.append(lap_var)

    mean_lap = float(np.mean(laplacian_variances)) if laplacian_variances else 0.0
    var_lap = float(np.var(laplacian_variances)) if laplacian_variances else 0.0

    # Clean VFI maintains consistent mesh energy without severe frame-to-frame drops
    score = 4.8
    # If high-frequency variance fluctuates wildly between even and odd frames, net is fluttering
    if len(laplacian_variances) >= 4:
        even_mean = float(np.mean(laplacian_variances[0::2]))
        odd_mean = float(np.mean(laplacian_variances[1::2]))
        diff_ratio = abs(even_mean - odd_mean) / (even_mean + 1e-5)
        if diff_ratio > 0.20:
            score -= min(2.5, diff_ratio * 4.0)

    score = max(1.0, min(5.0, score))

    details = {
        "score": score,
        "mean_mesh_crispness": mean_lap,
        "mesh_energy_variance": var_lap
    }

    return float(score), details
