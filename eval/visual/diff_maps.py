"""Difference and warping residual heatmap generator."""

import os
from typing import List
import cv2
import numpy as np


def generate_difference_heatmap(
    img1: np.ndarray,
    img2: np.ndarray,
    save_path: str,
    amplification: float = 3.0
) -> str:
    """Generates and saves an amplified JET colorized difference heatmap between two frames."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    diff_gray = np.mean(diff, axis=2)

    # Amplify difference for human visualization
    amplified = np.clip(diff_gray * amplification, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(amplified, cv2.COLORMAP_JET)

    cv2.imwrite(save_path, heatmap)
    return save_path
