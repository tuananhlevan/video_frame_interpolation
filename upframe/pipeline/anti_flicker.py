"""Temporal Anti-Flicker and Firefly Suppressor for video frame interpolation.

Eliminates 1-frame flickering white/black firefly pixels and boundary synthesis
overshoots caused by optical flow discontinuity, attention activation spikes,
or residual overshoot, while preserving genuine high-speed football ball motion.
"""

import logging
from typing import Optional
import cv2
import numpy as np

from eval.layer3_football.ball import detect_ball_candidates

logger = logging.getLogger(__name__)


class TemporalAntiFlicker:
    """Applies temporal envelope bounding box clamping to interpolated frames.
    
    Guarantees that no pixel at t=0.5 can violate the physical spatial-temporal
    envelope defined by source frames F0 and F1, eliminating temporal impulse
    noise (white fireflies / flickering sparks).
    """

    def __init__(
        self,
        radius: int = 3,
        margin: int = 15,
        protect_ball: bool = True,
        clamp_undershoot: bool = True
    ) -> None:
        self.radius = radius
        self.margin = margin
        self.protect_ball = protect_ball
        self.clamp_undershoot = clamp_undershoot
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (2 * radius + 1, 2 * radius + 1)
        )

    def process(
        self,
        frame_0: np.ndarray,
        frame_inter: np.ndarray,
        frame_1: np.ndarray,
        ball_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Clamps frame_inter within the local spatial-temporal envelope of frame_0 and frame_1.
        
        Args:
            frame_0: Previous source frame (H, W, C) uint8
            frame_inter: Synthesized intermediate frame (H, W, C) uint8
            frame_1: Next source frame (H, W, C) uint8
            ball_mask: Optional boolean or uint8 mask where fast ball is present to protect.
            
        Returns:
            Cleaned intermediate frame with fireflies removed.
        """
        # 1. Compute spatial-temporal upper bound (dilated envelope)
        max_0 = cv2.dilate(frame_0, self.kernel)
        max_1 = cv2.dilate(frame_1, self.kernel)
        temporal_max = np.maximum(max_0, max_1)
        upper_bound = np.clip(temporal_max.astype(np.int16) + self.margin, 0, 255).astype(np.uint8)

        # 2. Compute spatial-temporal lower bound (eroded envelope) if clamp_undershoot enabled
        if self.clamp_undershoot:
            min_0 = cv2.erode(frame_0, self.kernel)
            min_1 = cv2.erode(frame_1, self.kernel)
            temporal_min = np.minimum(min_0, min_1)
            lower_bound = np.clip(temporal_min.astype(np.int16) - self.margin, 0, 255).astype(np.uint8)
            clamped = np.clip(frame_inter, lower_bound, upper_bound)
        else:
            clamped = np.minimum(frame_inter, upper_bound)

        # 3. Protect ball region from envelope clamping
        if ball_mask is not None and np.any(ball_mask):
            clamped[ball_mask > 0] = frame_inter[ball_mask > 0]
        elif self.protect_ball:
            cands_0 = detect_ball_candidates(frame_0, min_circularity=0.45)
            cands_1 = detect_ball_candidates(frame_1, min_circularity=0.45)
            if cands_0 and cands_1:
                b0 = max(cands_0, key=lambda x: x[3])
                b1 = max(cands_1, key=lambda x: x[3])
                disp = float(np.hypot(b1[0] - b0[0], b1[1] - b0[1]))
                if 25.0 <= disp <= 160.0:
                    mx = int(round((b0[0] + b1[0]) / 2.0))
                    my = int(round((b0[1] + b1[1]) / 2.0))
                    r = int(round(max(b0[2], b1[2]) + 15))
                    mask = np.zeros(frame_inter.shape[:2], dtype=np.uint8)
                    cv2.circle(mask, (mx, my), r, 1, thickness=-1)
                    clamped[mask == 1] = frame_inter[mask == 1]

        return clamped
