"""Player Integrity and Player-Player Occlusion evaluation."""

from typing import Dict, List, Tuple
import cv2
import numpy as np


def detect_player_blobs(frame: np.ndarray) -> List[Dict[str, any]]:
    """Detects player foreground blobs on the football pitch."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Green pitch mask
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    pitch_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Invert pitch mask: players are non-pitch objects on or near the field
    non_pitch = cv2.bitwise_not(pitch_mask)
    # Clean up with opening/closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    non_pitch = cv2.morphologyEx(non_pitch, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(non_pitch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    players = []

    for c in contours:
        area = cv2.contourArea(c)
        # Filter for typical player bounding boxes (height 30-300, width 15-150)
        x, y, w, h = cv2.boundingRect(c)
        aspect = h / float(w) if w > 0 else 0
        if 200 <= area <= 25000 and 25 <= h <= 300 and 10 <= w <= 180 and 1.2 <= aspect <= 5.0:
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area / hull_area) if hull_area > 0 else 1.0
            players.append({
                "bbox": (x, y, w, h),
                "area": area,
                "solidity": solidity,
                "contour": c
            })

    return players


def evaluate_player_and_occlusion_integrity(
    frames_sequence: List[np.ndarray]
) -> Tuple[float, float, Dict[str, any]]:
    """Evaluates player silhouette integrity and occlusion handling across frames.
    
    Returns:
        (player_score [1.0, 5.0], occlusion_score [1.0, 5.0], details)
    """
    solidity_anomalies = 0
    occlusion_events = 0
    occlusion_failures = 0
    total_players_checked = 0

    for frame in frames_sequence:
        players = detect_player_blobs(frame)
        total_players_checked += len(players)

        # Check player silhouette solidity: warped / melted limbs drop solidity significantly
        for p in players:
            if p["solidity"] < 0.45:  # severed/melted silhouette
                solidity_anomalies += 1

        # Check occlusion events (intersecting bboxes)
        for i in range(len(players)):
            x1, y1, w1, h1 = players[i]["bbox"]
            for j in range(i + 1, len(players)):
                x2, y2, w2, h2 = players[j]["bbox"]
                # Intersection
                ix = max(x1, x2)
                iy = max(y1, y2)
                iw = min(x1 + w1, x2 + w2) - ix
                ih = min(y1 + h1, y2 + h2) - iy

                if iw > 5 and ih > 10:
                    occlusion_events += 1
                    # Inspect overlap region for ghosting/transparency
                    roi = frame[iy:iy + ih, ix:ix + iw]
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    lap_var = cv2.Laplacian(gray_roi, cv2.CV_32F).var()
                    # If overlap region has abnormally low texture / blurry wash, flag occlusion artifact
                    if lap_var < 50.0:
                        occlusion_failures += 1

    # Player Score: 1.0 to 5.0
    player_score = 5.0
    if total_players_checked > 0:
        anomaly_rate = solidity_anomalies / float(total_players_checked)
        player_score -= min(3.5, anomaly_rate * 25.0)
    player_score = max(1.0, min(5.0, player_score))

    # Occlusion Score: 1.0 to 5.0
    occlusion_score = 5.0
    if occlusion_events > 0:
        fail_rate = occlusion_failures / float(occlusion_events)
        occlusion_score -= min(3.5, fail_rate * 5.0)
    occlusion_score = max(1.0, min(5.0, occlusion_score))

    details = {
        "player_score": player_score,
        "occlusion_score": occlusion_score,
        "total_players_checked": total_players_checked,
        "solidity_anomalies": solidity_anomalies,
        "occlusion_events": occlusion_events,
        "occlusion_failures": occlusion_failures
    }

    return float(player_score), float(occlusion_score), details
