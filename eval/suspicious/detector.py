"""Suspicious moment detector and timestamp ranker."""

import csv
import os
from typing import Any, Dict, List, Optional
import numpy as np
from eval.types import SuspiciousMoment


def format_timestamp(seconds: float) -> str:
    """Formats seconds as HH:MM:SS.mmm."""
    total_sec = max(0.0, seconds)
    hrs = int(total_sec // 3600)
    mins = int((total_sec % 3600) // 60)
    secs = int(total_sec % 60)
    millis = int(round((total_sec - int(total_sec)) * 1000))
    if millis >= 1000:
        secs += 1
        millis -= 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def detect_suspicious_moments(
    frame_metrics: List[Dict[str, Any]],
    fps: float = 50.0,
    top_n: int = 25
) -> List[SuspiciousMoment]:
    """Ranks and flags the most suspicious frames/timestamps from frame metrics.
    
    frame_metrics contains dicts with:
        frame_index: int
        warping_error: float
        flicker_error: float
        ball_anomaly: bool / float
        occlusion_anomaly: bool / float
        line_distortion: bool / float
    """
    if not frame_metrics:
        return []

    moments: List[SuspiciousMoment] = []

    # Extract arrays for z-score anomaly detection
    warping_errors = np.array([m.get("warping_error", 0.0) for m in frame_metrics])
    mean_w = np.mean(warping_errors) if len(warping_errors) > 0 else 0.0
    std_w = np.std(warping_errors) if len(warping_errors) > 0 else 1.0
    if std_w < 1e-4:
        std_w = 1.0

    for m in frame_metrics:
        f_idx = m["frame_index"]
        t_sec = f_idx / fps if fps > 0 else 0.0
        t_str = format_timestamp(t_sec)

        w_err = m.get("warping_error", 0.0)
        z_w = (w_err - mean_w) / std_w

        # Specific domain anomalies
        ball_issue = m.get("ball_anomaly", False)
        occlusion_issue = m.get("occlusion_anomaly", False)
        line_issue = m.get("line_distortion", False)
        cut_issue = m.get("cut_hybrid", False)

        anomaly_type: Optional[str] = None
        severity = 1
        score = float(max(0.0, z_w))
        desc = ""

        if cut_issue:
            anomaly_type = "cut_hybrid"
            severity = 5
            score += 15.0
            desc = "Possible synthetic hybrid frame across scene cut"
        elif ball_issue:
            anomaly_type = "ball_artifact"
            severity = 4 if score > 3.0 else 3
            score += 8.0
            desc = "Possible ball artifact (ghosting / teleportation)"
        elif occlusion_issue:
            anomaly_type = "occlusion_artifact"
            severity = 4 if score > 2.5 else 3
            score += 6.0
            desc = "Possible player occlusion artifact (merged/ghost limbs)"
        elif line_issue:
            anomaly_type = "pitch_distortion"
            severity = 3
            score += 4.0
            desc = "Possible pitch-line distortion / curvature wobble"
        elif z_w >= 2.5:
            anomaly_type = "temporal_warping_spike"
            severity = 3 if z_w < 4.0 else 4
            desc = f"Unusually large temporal residual (z-score: {z_w:.1f})"

        if anomaly_type is not None:
            moments.append(
                SuspiciousMoment(
                    timestamp_sec=round(t_sec, 3),
                    timestamp_str=t_str,
                    frame_index=f_idx,
                    anomaly_type=anomaly_type,
                    severity=min(5, severity),
                    anomaly_score=round(score, 2),
                    description=desc
                )
            )

    # Sort descending by anomaly score
    moments.sort(key=lambda x: x.anomaly_score, reverse=True)
    return moments[:top_n]


def export_suspicious_moments_csv(moments: List[SuspiciousMoment], output_path: str) -> None:
    """Exports suspicious moments to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Seconds", "Frame_Index", "Anomaly_Type", "Severity", "Anomaly_Score", "Description"])
        for m in moments:
            writer.writerow([
                m.timestamp_str,
                m.timestamp_sec,
                m.frame_index,
                m.anomaly_type,
                m.severity,
                m.anomaly_score,
                m.description
            ])
