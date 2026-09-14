"""Temporal Flicker and Odd-Even Oscillation evaluation."""

from typing import Dict, List, Tuple
import cv2
import numpy as np


def evaluate_temporal_flicker(
    frames_sequence: List[np.ndarray]
) -> Tuple[float, float, Dict[str, float]]:
    """Measures temporal intensity oscillation and high-frequency flicker.
    
    Returns:
        (flicker_score, odd_even_oscillation_index, details)
        flicker_score: higher = more temporal flicker/instability
        odd_even_oscillation_index: higher = strong odd-frame popping artifact
    """
    if len(frames_sequence) < 4:
        return 0.0, 0.0, {}

    # Convert to grayscale float
    grays = [
        cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) if f.ndim == 3 else f.astype(np.float32)
        for f in frames_sequence
    ]

    # Mean intensity differences across consecutive frames
    consecutive_diffs: List[float] = []
    lag2_diffs: List[float] = []

    for i in range(1, len(grays)):
        d1 = float(np.mean(np.abs(grays[i] - grays[i - 1])))
        consecutive_diffs.append(d1)

    for i in range(2, len(grays)):
        d2 = float(np.mean(np.abs(grays[i] - grays[i - 2])))
        lag2_diffs.append(d2)

    mean_d1 = float(np.mean(consecutive_diffs))
    mean_d2 = float(np.mean(lag2_diffs)) if lag2_diffs else mean_d1

    # In continuous motion, lag-2 difference is usually comparable or larger than lag-1 difference.
    # If lag-1 >> lag-2, consecutive frames oscillate back and forth (odd-even shimmer).
    oscillation_ratios: List[float] = []
    for i in range(len(lag2_diffs)):
        # d1_cur = |F_{i+2} - F_{i+1}|, d2 = |F_{i+2} - F_i|
        d1_cur = consecutive_diffs[i + 1]
        d2_cur = lag2_diffs[i]
        ratio = d1_cur / (d2_cur + 1.0)
        oscillation_ratios.append(ratio)

    mean_oscillation = float(np.mean(oscillation_ratios)) if oscillation_ratios else 1.0

    # Local texture variance flicker (e.g. grass texture shimmer)
    # Downsampled local variance
    var_flickers: List[float] = []
    for i in range(len(grays) - 1):
        # Laplacian variance measures texture crispness
        lap1 = cv2.Laplacian(grays[i], cv2.CV_32F).var()
        lap2 = cv2.Laplacian(grays[i + 1], cv2.CV_32F).var()
        var_flickers.append(abs(lap1 - lap2) / (lap1 + lap2 + 1e-5))

    # 2nd-order temporal residual: isolates true temporal flicker / pulsation from linear camera pan motion
    temporal_residuals: List[float] = []
    for i in range(1, len(grays) - 1):
        pred_linear = 0.5 * (grays[i - 1] + grays[i + 1])
        temporal_residuals.append(float(np.mean(np.abs(grays[i] - pred_linear))))

    mean_res = float(np.mean(temporal_residuals)) if temporal_residuals else mean_d1

    # Composite flicker score [0.0, 10.0], lower is better
    flicker_score = float(mean_res * 0.15 + mean_var_flicker * 3.5)

    details = {
        "mean_consecutive_diff": mean_d1,
        "mean_lag2_diff": mean_d2,
        "mean_temporal_residual": mean_res,
        "odd_even_oscillation_index": mean_oscillation,
        "texture_flicker_index": mean_var_flicker,
        "flicker_score": flicker_score
    }

    return flicker_score, mean_oscillation, details
