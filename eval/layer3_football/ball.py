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
    min_circularity: float = 0.35
) -> List[Tuple[float, float, float, float]]:
    """Detects potential football candidates on the pitch.
    
    Constrains search to the main pitch polygon (eliminating billboards and crowd)
    and suppresses player bounding boxes (eliminating cleats and socks).
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

    # 3. Find white/high-contrast objects inside clean pitch
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


def evaluate_ball_integrity(
    frames_sequence: List[np.ndarray],
    fps: float = 50.0,
    max_ball_velocity_px: Optional[float] = None,
    max_lost_gap_frames: int = 6,
    source_has_ball: Optional[bool] = None,
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
                    # OUTLIER GATING: Penalize teleportation, but DO NOT snap prev_pos to the outlier!
                    teleportation_count += 1
                    frames_since_last_seen += 1
                    if frames_since_last_seen > max_lost_gap_frames:
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
            if frames_since_last_seen > max_lost_gap_frames:
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
        elif best_candidate is None and len(candidates) >= 2:
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    dist = math.hypot(candidates[i][0] - candidates[j][0], candidates[i][1] - candidates[j][1])
                    if min_dup_dist <= dist <= max_dup_dist:
                        has_duplicate = True
                        break
                if has_duplicate:
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

    # 4. Soft Saturation Curve (Flaw 2)
    dup_rate = duplicate_frames_count / max(1, tracked_count)
    tel_rate = teleportation_count / max(1, tracked_count)
    deform_rate = deformed_frames_count / max(1, tracked_count)

    score = 5.0
    score -= min(2.0, (dup_rate / 0.10) * 2.0)
    score -= min(1.5, (tel_rate / 0.10) * 1.5)
    score -= min(1.0, (deform_rate / 0.10) * 1.0)

    # Presence penalty: If video has significant length (>= 30 frames) but tracked ball is < 15% of frames
    track_presence = tracked_count / float(total_frames)
    if total_frames >= 30 and track_presence < 0.15:
        presence_deficit = (0.15 - track_presence) / 0.15
        score -= min(2.0, presence_deficit * 2.0)

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
