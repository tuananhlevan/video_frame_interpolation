"""Pitch Geometry evaluation.

Detects pitch markings (touchlines, penalty boxes, center circle, halfway line)
and tests for bending, wobbling, double lines, or broken lines in generated frames.
Outputs a 1.0 to 5.0 pitch geometry score.
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np

from eval.layer3_football.ball import get_pitch_coverage
from upframe.pipeline.scene_detect import compute_frame_difference


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


def extract_pitch_line_mask(frame: np.ndarray) -> np.ndarray:
    """Extracts binary mask of white pitch markings on the green field."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    pitch_mask = cv2.inRange(hsv, lower_green, upper_green)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
    pitch_zone = cv2.dilate(pitch_mask, np.ones((15, 15), np.uint8))
    return cv2.bitwise_and(white_mask, pitch_zone)


def evaluate_pitch_geometry_triplet(
    frame_a: np.ndarray,
    frame_x: np.ndarray,
    frame_b: np.ndarray
) -> Dict[str, any]:
    """Evaluates pitch line geometry on interpolated frame F_X against reference frames F_A and F_B.
    
    Computes:
    1. Differential Pitch Line Profile Residual:
       R_pitch = mean(|M_X - 0.5*(M_A + M_B)|) over pitch region.
       Zero false positives on natural pitch circles/arcs, flags synthetic wavy/bent/ghost lines.
    2. Differential Line Straightness / Curvature:
       Fits linear model to long line contours and computes differential orthogonal deviation.
    
    Returns:
        Dict with keys:
        - pitch_active: bool (True if green pitch >= 15%)
        - line_residual: float (mask deviation)
        - warp_events: int (number of lines with synthetic curvature)
        - max_warp_deviation: float (px)
        - lines_detected_x: int
    """
    res: Dict[str, any] = {
        "pitch_active": False,
        "line_residual": 0.0,
        "warp_events": 0,
        "max_warp_deviation": 0.0,
        "lines_detected_x": 0
    }
    
    # Check pitch coverage on all 3 frames and check for scene cut
    cov_a = get_pitch_coverage(frame_a)
    cov_b = get_pitch_coverage(frame_b)
    
    hsv_x = cv2.cvtColor(frame_x, cv2.COLOR_BGR2HSV)
    pitch_mask_x = cv2.inRange(hsv_x, np.array([30, 40, 40]), np.array([90, 255, 255]))
    pitch_coverage_x = float(np.count_nonzero(pitch_mask_x)) / float(frame_x.shape[0] * frame_x.shape[1])
    
    if cov_a < 0.15 or cov_b < 0.15 or pitch_coverage_x < 0.15:
        return res
    
    # Check if transition between frame_a and frame_b is a camera cut
    if compute_frame_difference(frame_a, frame_b) >= 0.35:
        return res
    
    res["pitch_active"] = True
    
    # 1. Differential Mask Residual
    mask_a = extract_pitch_line_mask(frame_a).astype(np.float32) / 255.0
    mask_b = extract_pitch_line_mask(frame_b).astype(np.float32) / 255.0
    mask_x = extract_pitch_line_mask(frame_x).astype(np.float32) / 255.0
    
    expected_mask = 0.5 * (mask_a + mask_b)
    pitch_zone = (pitch_mask_x > 0)
    if np.any(pitch_zone):
        res["line_residual"] = float(np.mean(np.abs(mask_x[pitch_zone] - expected_mask[pitch_zone])))
    
    # 2. Contour straightness on F_X with reference differential
    line_mask_x_uint8 = (mask_x * 255).astype(np.uint8)
    contours_x, _ = cv2.findContours(line_mask_x_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    line_mask_a_uint8 = (mask_a * 255).astype(np.uint8)
    line_mask_b_uint8 = (mask_b * 255).astype(np.uint8)
    
    warp_events = 0
    max_warp = 0.0
    lines_x_count = 0
    
    for c in contours_x:
        if cv2.arcLength(c, False) >= 80:
            lines_x_count += 1
            pts = c.reshape(-1, 2).astype(np.float32)
            line_params = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(line_params[0][0]), float(line_params[1][0])
            x0, y0 = float(line_params[2][0]), float(line_params[3][0])
            dists_x = np.abs(vy * (pts[:, 0] - x0) - vx * (pts[:, 1] - y0))
            p95_dev_x = float(np.percentile(dists_x, 95))
            
            if p95_dev_x > 2.2:
                rect = cv2.boundingRect(c)
                rx, ry, rw, rh = rect
                sub_a = line_mask_a_uint8[ry:ry+rh, rx:rx+rw]
                sub_b = line_mask_b_uint8[ry:ry+rh, rx:rx+rw]
                
                ref_curved = False
                for sub_ref in (sub_a, sub_b):
                    if sub_ref.size == 0:
                        continue
                    cnts_ref, _ = cv2.findContours(sub_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    for cr in cnts_ref:
                        if cv2.arcLength(cr, False) >= 60:
                            pts_r = cr.reshape(-1, 2).astype(np.float32)
                            lp_r = cv2.fitLine(pts_r, cv2.DIST_L2, 0, 0.01, 0.01)
                            vrx, vry = float(lp_r[0][0]), float(lp_r[1][0])
                            rx0, ry0 = float(lp_r[2][0]), float(lp_r[3][0])
                            dr = np.abs(vry * (pts_r[:, 0] - rx0) - vrx * (pts_r[:, 1] - ry0))
                            if float(np.percentile(dr, 95)) > 1.8:
                                ref_curved = True
                                break
                    if ref_curved:
                        break
                
                if not ref_curved:
                    warp_events += 1
                    max_warp = max(max_warp, p95_dev_x)
    
    res["warp_events"] = warp_events
    res["max_warp_deviation"] = max_warp
    res["lines_detected_x"] = lines_x_count
    return res


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
        # Filter for structural straight lines (length >= 70px) to prevent
        # circular pitch markings (center circle, penalty arcs) from triggering false wobble
        lines_straight = [l for l in lines if np.hypot(l[2] - l[0], l[3] - l[1]) >= 70]
        if len(lines_straight) >= 2:
            angles = [(np.arctan2(y2 - y1, x2 - x1) % np.pi) for x1, y1, x2, y2 in lines_straight]
            
            # Simple clustering into up to 2 perspective vanishing sets
            cluster_a: List[float] = [angles[0]]
            cluster_b: List[float] = []
            
            for ang in angles[1:]:
                d_a = circular_ang_dist(ang, cluster_a[0])
                if d_a < 0.45:
                    cluster_a.append(ang)
                elif abs(d_a - np.pi / 2) < 0.30:
                    cluster_b.append(ang)
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
