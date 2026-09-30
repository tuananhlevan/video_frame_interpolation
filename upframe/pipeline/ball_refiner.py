"""Ball Trajectory Refiner and Ghosting Inpainter for football broadcast VFI."""

import logging
import math
from typing import Optional, Tuple
import cv2
import numpy as np

from eval.layer3_football.ball import detect_ball_candidates

logger = logging.getLogger(__name__)


class BallRefiner:
    """Detects and repairs fast-moving ball artifacts (ghosting, disappearance, deformation)
    in interpolated frames using physical trajectory continuity.
    """

    def __init__(
        self,
        min_motion_threshold: float = 25.0,  # Minimum pixel jump between F0 and F1 to trigger refinement
        max_motion_threshold: float = 160.0, # Maximum plausible 40ms displacement (~144 km/h)
        blend_radius_mult: float = 1.6
    ) -> None:
        self.min_motion_threshold = min_motion_threshold
        self.max_motion_threshold = max_motion_threshold
        self.blend_radius_mult = blend_radius_mult
        self._last_f1: Optional[np.ndarray] = None
        self._last_cands_1: Optional[list] = None

    def refine(
        self,
        frame_0: np.ndarray,
        frame_inter: np.ndarray,
        frame_1: np.ndarray,
        timestep: float = 0.5
    ) -> np.ndarray:
        """Inspects and refines the ball region in frame_inter.
        
        If both frame_0 and frame_1 contain a fast-moving ball:
        1. Calculates expected physical midpoint (pos_mid) based on trajectory.
        2. Inspects frame_inter for ball defects:
           - Ball missing (dissolved) at pos_mid.
           - Dual ghost balls appearing at pos_0 or pos_1.
        3. Inpaints / synthesizes the crisp ball at pos_mid and cleans residual ghost halos at pos_0/pos_1.
        
        Returns:
            Refined frame_inter (same shape and dtype as input).
        """
        # Fast candidate detection on F0 and F1 with sequential cache
        if self._last_f1 is frame_0 and self._last_cands_1 is not None:
            cands_0 = list(self._last_cands_1)
        else:
            cands_0 = detect_ball_candidates(frame_0, min_circularity=0.45)

        cands_1 = detect_ball_candidates(frame_1, min_circularity=0.45)
        self._last_f1 = frame_1
        self._last_cands_1 = list(cands_1)

        if not cands_0 or not cands_1:
            return frame_inter

        # Select highest circularity candidates
        cands_0.sort(key=lambda c: c[3], reverse=True)
        cands_1.sort(key=lambda c: c[3], reverse=True)

        ball_0 = cands_0[0]
        ball_1 = cands_1[0]

        x0, y0, r0, circ0 = ball_0
        x1, y1, r1, circ1 = ball_1

        disp = math.hypot(x1 - x0, y1 - y0)
        if disp < self.min_motion_threshold or disp > self.max_motion_threshold:
            # Small displacement (handled well by standard optical flow) or implausible jump
            return frame_inter

        # Expected position at timestep (default 0.5)
        exp_x = x0 + (x1 - x0) * timestep
        exp_y = y0 + (y1 - y0) * timestep
        avg_r = (r0 + r1) / 2.0
        patch_r = int(math.ceil(avg_r * self.blend_radius_mult))
        patch_r = max(patch_r, 8)

        # Check if frame_inter already has a sharp, well-formed ball near exp_x, exp_y
        cands_inter = detect_ball_candidates(frame_inter, min_circularity=0.40)
        has_good_ball = False
        ghost_at_0 = False
        ghost_at_1 = False

        for cx, cy, cr, c_circ in cands_inter:
            d_mid = math.hypot(cx - exp_x, cy - exp_y)
            d_0 = math.hypot(cx - x0, cy - y0)
            d_1 = math.hypot(cx - x1, cy - y1)

            if d_mid <= avg_r * 1.5 and c_circ >= 0.55:
                has_good_ball = True
            if d_0 <= avg_r * 1.5 and d_mid > avg_r * 2.0:
                ghost_at_0 = True
            if d_1 <= avg_r * 1.5 and d_mid > avg_r * 2.0:
                ghost_at_1 = True

        # If ball is already crisp at midpoint and no dual ghosts, keep original frame
        if has_good_ball and not (ghost_at_0 or ghost_at_1):
            return frame_inter

        # Apply seamless ball restoration
        output_frame = frame_inter.copy()
        h, w = output_frame.shape[:2]

        # 1. Erase ghost remnants at (x0, y0) and (x1, y1) if present
        def clean_ghost_halo(gx: float, gy: float):
            ix, iy = int(round(gx)), int(round(gy))
            radius = int(math.ceil(avg_r * 1.4))
            x_min, x_max = max(0, ix - radius), min(w, ix + radius)
            y_min, y_max = max(0, iy - radius), min(h, iy + radius)
            if x_max > x_min and y_max > y_min:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (ix, iy), radius, 255, -1)
                inpainted = cv2.inpaint(output_frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                output_frame[y_min:y_max, x_min:x_max] = inpainted[y_min:y_max, x_min:x_max]

        if ghost_at_0:
            clean_ghost_halo(x0, y0)
        if ghost_at_1:
            clean_ghost_halo(x1, y1)

        # 2. Extract ball appearances from F0 and F1
        def get_ball_patch(src: np.ndarray, bx: float, by: float) -> Optional[np.ndarray]:
            ix, iy = int(round(bx)), int(round(by))
            if ix - patch_r < 0 or ix + patch_r >= w or iy - patch_r < 0 or iy + patch_r >= h:
                return None
            return src[iy - patch_r:iy + patch_r, ix - patch_r:ix + patch_r]

        patch_0 = get_ball_patch(frame_0, x0, y0)
        patch_1 = get_ball_patch(frame_1, x1, y1)

        if patch_0 is not None and patch_1 is not None and patch_0.shape == patch_1.shape:
            ball_patch = cv2.addWeighted(patch_0, 1.0 - timestep, patch_1, timestep, 0)
        elif patch_0 is not None:
            ball_patch = patch_0
        elif patch_1 is not None:
            ball_patch = patch_1
        else:
            return output_frame

        # 3. Create circular alpha feather mask for natural insertion
        ph, pw = ball_patch.shape[:2]
        center = (pw // 2, ph // 2)
        alpha_mask = np.zeros((ph, pw), dtype=np.float32)
        cv2.circle(alpha_mask, center, int(avg_r * 1.1), 1.0, -1)
        alpha_mask = cv2.GaussianBlur(alpha_mask, (5, 5), 1.5)
        alpha_3c = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)

        # 4. Composite onto expected midpoint
        mid_ix, mid_iy = int(round(exp_x)), int(round(exp_y))
        x_min, x_max = mid_ix - pw // 2, mid_ix - pw // 2 + pw
        y_min, y_max = mid_iy - ph // 2, mid_iy - ph // 2 + ph

        if 0 <= x_min and x_max <= w and 0 <= y_min and y_max <= h:
            bg_region = output_frame[y_min:y_max, x_min:x_max].astype(np.float32)
            fg_region = ball_patch.astype(np.float32)
            blended = fg_region * alpha_3c + bg_region * (1.0 - alpha_3c)
            output_frame[y_min:y_max, x_min:x_max] = np.clip(blended, 0, 255).astype(np.uint8)

        return output_frame
