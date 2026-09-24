"""Ball Integrity evaluation for football broadcast video.

Detects and tracks football candidates on the pitch, checking for:
- single ball constraint (no ghost ball / duplicate balls) with velocity-adaptive window
- trajectory continuity (no teleportation / sudden jumps) with outlier gating
- shape integrity (penalizes stretched, blurred, or deformed balls)
- track presence (penalizes ball disappearance / dissolution)
- outputs 1.0 to 5.0 ball score with rate-normalized soft-saturation curve.
"""

import math
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from eval.layer3_football.player_occlusion import detect_player_blobs


def detect_ball_candidates(
    frame: np.ndarray,
    players: Optional[List[Dict[str, any]]] = None,
    pitch_lines: Optional[List[Tuple[int, int, int, int]]] = None,
    min_circularity: float = 0.35
) -> List[Tuple[float, float, float, float]]:
    """Detects potential football candidates on the pitch.
    
    Constrains search to the main pitch polygon (eliminating billboards and crowd)
    and suppresses player bounding boxes (eliminating cleats and socks) and pitch markings.
    Permits deformed balls down to min_circularity (0.35) so shape degradation
    can be detected and penalized rather than silently discarded.
    
    Returns:
        List of (cx, cy, radius, circularity)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Green pitch mask
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    pitch_mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(pitch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest_c = max(contours, key=cv2.contourArea)

    # 1. Pitch field polygon constraint (eroded to eliminate LED boards & crowd)
    clean_pitch = np.zeros_like(pitch_mask)
    cv2.drawContours(clean_pitch, [largest_c], -1, 255, thickness=cv2.FILLED)
    clean_pitch = cv2.erode(clean_pitch, np.ones((7, 7), np.uint8))

    # 2. Player silhouette suppression (eliminates cleats & socks in walls/scrums)
    if players is None:
        players = detect_player_blobs(frame)
    for p in players:
        px, py, pw, ph = p["bbox"]
        cv2.rectangle(clean_pitch, (px - 10, py - 5), (px + pw + 10, py + ph + 15), 0, thickness=cv2.FILLED)

    # 3. Suppress known pitch markings (touchlines, penalty boxes)
    if pitch_lines is not None:
        for lx1, ly1, lx2, ly2 in pitch_lines:
            cv2.line(clean_pitch, (lx1, ly1), (lx2, ly2), 0, thickness=14)

    # 4. Find white/high-contrast objects inside clean pitch
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    ball_search_mask = cv2.bitwise_and(bright_mask, clean_pitch)

    contours, _ = cv2.findContours(ball_search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Tuple[float, float, float, float]] = []

    for c in contours:
        area = cv2.contourArea(c)
        # Expected ball size in 1080p: area ~ 20 to 600 px
        if 20 <= area <= 600:
            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                circularity = (4.0 * math.pi * area) / (perimeter ** 2)
                if circularity >= min_circularity:
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    candidates.append((float(x), float(y), float(radius), float(circularity)))

    return candidates


def get_pitch_coverage(frame: np.ndarray) -> float:
    """Calculates the proportion of the frame occupied by the green pitch field."""
    small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([90, 255, 255]))
    return float(np.count_nonzero(mask)) / float(320 * 180)


def evaluate_ball_triplet(
    frame_a: np.ndarray,
    frame_x: np.ndarray,
    frame_b: np.ndarray,
    players_x: Optional[List[Dict[str, any]]] = None,
    pitch_lines_x: Optional[List[Tuple[int, int, int, int]]] = None
) -> Dict[str, any]:
    """Evaluates ball integrity on an interpolated frame F_X using ground-truth frames (F_A, F_B).
    
    If both F_A and F_B have a confirmed ball at (x_A, y_A) and (x_B, y_B) with plausible velocity,
    F_X is verified for:
    - Ball in-betweening presence (flags dissolution if missing)
    - Trajectory collinearity (flags trajectory wobble/bending)
    - Ball deformation (flags circularity drop)
    - Ghost duplicates (flags extra balls along motion trajectory)
    
    If either F_A or F_B does NOT have a ball, F_X is not penalized and non-ball blobs are ignored.
    
    Returns:
        Dict with keys: ball_active, ball_matched, ball_dissolved, ball_wobble, ball_deformed, ball_duplicate,
        pos_a, pos_b, pos_x, expected_pos, collinear_dist, circularity_x
    """
    cands_a = detect_ball_candidates(frame_a, min_circularity=0.50)
    cands_b = detect_ball_candidates(frame_b, min_circularity=0.50)
    
    res: Dict[str, any] = {
        "ball_active": False,
        "ball_matched": False,
        "ball_dissolved": False,
        "ball_occluded": False,
        "ball_wobble": False,
        "ball_deformed": False,
        "ball_duplicate": False,
        "pos_a": None,
        "pos_b": None,
        "pos_x": None,
        "expected_pos": None,
        "collinear_dist": 0.0,
        "circularity_x": None
    }
    
    if not cands_a or not cands_b:
        return res
    
    cands_a.sort(key=lambda x: x[3], reverse=True)
    cands_b.sort(key=lambda x: x[3], reverse=True)
    
    ball_a = cands_a[0]
    ball_b = cands_b[0]
    
    dist_ab = math.hypot(ball_b[0] - ball_a[0], ball_b[1] - ball_a[1])
    # Max physical displacement over 2 source frames (40ms = 25fps) is ~160px (144 km/h at 1080p)
    if dist_ab > 160.0:
        return res
    
    res["ball_active"] = True
    res["pos_a"] = (ball_a[0], ball_a[1])
    res["pos_b"] = (ball_b[0], ball_b[1])
    
    expected_x = (ball_a[0] + ball_b[0]) / 2.0
    expected_y = (ball_a[1] + ball_b[1]) / 2.0
    res["expected_pos"] = (expected_x, expected_y)
    
    cands_x = detect_ball_candidates(frame_x, players=players_x, pitch_lines=pitch_lines_x, min_circularity=0.35)
    
    search_radius = max(20.0, dist_ab * 0.55 + 10.0)
    matched_x = None
    min_d = float('inf')
    for c in cands_x:
        d = math.hypot(c[0] - expected_x, c[1] - expected_y)
        if d <= search_radius and d < min_d:
            min_d = d
            matched_x = c
    
    if matched_x is None:
        # Check if ball is naturally occluded by a player in frame_x
        is_occluded = False
        if players_x is not None:
            for p in players_x:
                px, py, pw, ph = p["bbox"]
                if (px - 10 <= expected_x <= px + pw + 10) and (py - 10 <= expected_y <= py + ph + 10):
                    is_occluded = True
                    break
        if is_occluded:
            res["ball_occluded"] = True
            res["ball_dissolved"] = False
        else:
            res["ball_dissolved"] = True
        return res
    
    res["ball_matched"] = True
    res["pos_x"] = (matched_x[0], matched_x[1])
    res["circularity_x"] = matched_x[3]
    
    if matched_x[3] < 0.60 and (ball_a[3] >= 0.65 or ball_b[3] >= 0.65):
        res["ball_deformed"] = True
    
    if dist_ab > 8.0:
        p1 = np.array([ball_a[0], ball_a[1]], dtype=np.float32)
        p2 = np.array([ball_b[0], ball_b[1]], dtype=np.float32)
        px = np.array([matched_x[0], matched_x[1]], dtype=np.float32)
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        line_unit = line_vec / (line_len + 1e-6)
        proj = np.dot(px - p1, line_unit)
        perp_vec = (px - p1) - proj * line_unit
        perp_dist = float(np.linalg.norm(perp_vec))
        res["collinear_dist"] = perp_dist
        if perp_dist > 8.0:
            res["ball_wobble"] = True
    
    if dist_ab >= 8.0:
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        line_unit = line_vec / (line_len + 1e-6)
        for c in cands_x:
            if (c[0], c[1]) != (matched_x[0], matched_x[1]) and c[3] >= 0.55:
                pc = np.array([c[0], c[1]], dtype=np.float32)
                proj_c = float(np.dot(pc - p1, line_unit))
                perp_vec_c = (pc - p1) - proj_c * line_unit
                perp_dist_c = float(np.linalg.norm(perp_vec_c))
                d_to_primary = math.hypot(c[0] - matched_x[0], c[1] - matched_x[1])
                
                # Ghost ball is along the motion trajectory between pA and pB
                if (-10.0 <= proj_c <= line_len + 10.0) and perp_dist_c <= 12.0 and d_to_primary >= 8.0:
                    res["ball_duplicate"] = True
                    break
    
    return res


def evaluate_ball_integrity(
    frames_sequence: List[np.ndarray],
    fps: float = 50.0,
    max_ball_velocity_px: Optional[float] = None,
    max_lost_gap_frames: int = 6,
    source_has_ball: Optional[bool] = None,
    source_tracked_count: Optional[int] = None,
    min_pitch_pct: float = 0.15
) -> Tuple[float, Dict[str, any]]:
    """Evaluates ball integrity across a sequence of frames with outlier gating and presence checks.
    
    Returns:
        (ball_score [1.0, 5.0], details)
    """
    total_frames = len(frames_sequence)
    if total_frames == 0:
        return 5.0, {
            "score": 5.0,
            "ball_present_in_scene": False,
            "duplicate_ball_events": 0,
            "teleportation_count": 0,
            "deformed_frames_count": 0,
            "tracked_positions_count": 0,
            "dup_rate_pct": 0.0,
            "tel_rate_pct": 0.0,
            "deform_rate_pct": 0.0,
            "track_presence_pct": 0.0,
            "avg_pitch_pct": 0.0
        }

    base_velocity_limit = max_ball_velocity_px if max_ball_velocity_px is not None else 130.0 * (50.0 / max(1.0, fps))

    teleportation_count = 0
    duplicate_frames_count = 0
    deformed_frames_count = 0
    trajectories: List[Tuple[float, float]] = []
    pitch_coverage_samples: List[float] = []

    prev_pos: Optional[Tuple[float, float]] = None
    prev_prev_pos: Optional[Tuple[float, float]] = None
    frames_since_last_seen = 0

    for idx, frame in enumerate(frames_sequence):
        if idx % 5 == 0 or idx == total_frames - 1:
            pitch_coverage_samples.append(get_pitch_coverage(frame))
        players = detect_player_blobs(frame)
        candidates = detect_ball_candidates(frame, players=players, min_circularity=0.50)

        # 1. Candidate tracking with Outlier Gating (Flaw 3) & Shape Accounting (Flaw 6)
        best_candidate: Optional[Tuple[float, float, float, float]] = None
        if candidates:
            if prev_pos is None:
                # Pick most circular candidate to initiate track reliably
                candidates.sort(key=lambda x: x[3], reverse=True)
                if candidates[0][3] >= 0.55:
                    best_candidate = candidates[0]
                    trajectories.append((best_candidate[0], best_candidate[1]))
                    prev_prev_pos = prev_pos
                    prev_pos = (best_candidate[0], best_candidate[1])
                    frames_since_last_seen = 0
                    if candidates[0][3] < 0.65:
                        deformed_frames_count += 1
            else:
                allowed_displacement = base_velocity_limit * (frames_since_last_seen + 1)
                candidates.sort(key=lambda x: math.hypot(x[0] - prev_pos[0], x[1] - prev_pos[1]))
                closest = candidates[0]
                dist = math.hypot(closest[0] - prev_pos[0], closest[1] - prev_pos[1])

                if dist > allowed_displacement:
                    # Only penalize teleportation if ball was actively tracked on immediate prior frame
                    if frames_since_last_seen == 0:
                        teleportation_count += 1
                    frames_since_last_seen += 1
                    if frames_since_last_seen > 3:
                        prev_pos = None
                        prev_prev_pos = None
                else:
                    best_candidate = closest
                    trajectories.append((best_candidate[0], best_candidate[1]))
                    prev_prev_pos = prev_pos
                    prev_pos = (best_candidate[0], best_candidate[1])
                    frames_since_last_seen = 0
                    if closest[3] < 0.65:
                        deformed_frames_count += 1
        else:
            frames_since_last_seen += 1
            if frames_since_last_seen > 3:
                prev_pos = None
                prev_prev_pos = None

        # 2. Velocity-adaptive duplicate window (Flaw 7) & frame-level accounting (Flaw 4)
        # Duplicate ball is an artifact of the ball itself (ghosting)
        current_vel = 0.0
        if prev_pos is not None and prev_prev_pos is not None:
            current_vel = math.hypot(prev_pos[0] - prev_prev_pos[0], prev_pos[1] - prev_prev_pos[1])
        max_dup_dist = max(60.0, min(150.0, current_vel * 1.5))
        min_dup_dist = 8.0

        has_duplicate = False
        if best_candidate is not None and len(candidates) >= 2:
            for c in candidates:
                if c is not best_candidate:
                    dist = math.hypot(best_candidate[0] - c[0], best_candidate[1] - c[1])
                    if min_dup_dist <= dist <= max_dup_dist:
                        has_duplicate = True
                        break
        if has_duplicate:
            duplicate_frames_count += 1

    tracked_count = len(trajectories)
    avg_pitch_pct = float(np.mean(pitch_coverage_samples)) if pitch_coverage_samples else 0.0

    # 3. Context & presence determination
    if source_has_ball is False:
        scene_has_ball = False
    elif source_has_ball is True:
        scene_has_ball = True
    else:
        # Standalone evaluation: infer presence from active pitch play coverage
        scene_has_ball = (avg_pitch_pct >= min_pitch_pct)

    # If the scene does not have an active pitch or ball was not present in source, having 0 ball candidates is expected
    if not scene_has_ball:
        return 5.0, {
            "score": 5.0,
            "ball_present_in_scene": False,
            "duplicate_ball_events": duplicate_frames_count,
            "teleportation_count": teleportation_count,
            "deformed_frames_count": deformed_frames_count,
            "tracked_positions_count": tracked_count,
            "dup_rate_pct": 0.0,
            "tel_rate_pct": 0.0,
            "deform_rate_pct": 0.0,
            "track_presence_pct": 0.0,
            "avg_pitch_pct": round(avg_pitch_pct * 100, 2)
        }

    # True active football scene where ball is expected:
    if tracked_count == 0:
        return 1.0, {
            "score": 1.0,
            "ball_present_in_scene": True,
            "duplicate_ball_events": duplicate_frames_count,
            "teleportation_count": teleportation_count,
            "deformed_frames_count": deformed_frames_count,
            "tracked_positions_count": 0,
            "dup_rate_pct": 0.0,
            "tel_rate_pct": 0.0,
            "deform_rate_pct": 0.0,
            "track_presence_pct": 0.0,
            "avg_pitch_pct": round(avg_pitch_pct * 100, 2)
        }

    # 4. Soft Saturation Curve
    dup_rate = duplicate_frames_count / max(1, tracked_count)
    tel_rate = teleportation_count / max(1, tracked_count)
    deform_rate = deformed_frames_count / max(1, tracked_count)

    score = 5.0
    score -= min(2.0, (dup_rate / 0.15) * 2.0)
    score -= min(1.5, (tel_rate / 0.15) * 1.5)
    score -= min(1.0, (deform_rate / 0.15) * 1.0)

    # Relative presence penalty: only penalize if source video had a tracked ball and output lost it,
    # or if standalone run on wide pitch (>40% pitch) with zero or near-zero tracking
    track_presence = tracked_count / float(total_frames)
    if source_tracked_count is not None:
        expected_tracked = max(1, source_tracked_count * 2)
        retention = tracked_count / float(expected_tracked)
        if retention < 0.70:
            deficit = (0.70 - retention) / 0.70
            score -= min(2.0, deficit * 2.0)
    elif avg_pitch_pct >= 0.40 and total_frames >= 60 and track_presence < 0.08:
        presence_deficit = (0.08 - track_presence) / 0.08
        score -= min(1.5, presence_deficit * 1.5)

    score = max(1.0, min(5.0, score))

    details = {
        "score": round(score, 2),
        "ball_present_in_scene": True,
        "duplicate_ball_events": duplicate_frames_count,
        "teleportation_count": teleportation_count,
        "deformed_frames_count": deformed_frames_count,
        "tracked_positions_count": tracked_count,
        "dup_rate_pct": round(dup_rate * 100, 2),
        "tel_rate_pct": round(tel_rate * 100, 2),
        "deform_rate_pct": round(deform_rate * 100, 2),
        "track_presence_pct": round(track_presence * 100, 2),
        "avg_pitch_pct": round(avg_pitch_pct * 100, 2)
    }

    return float(score), details
