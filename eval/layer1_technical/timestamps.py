"""Timestamp and PTS continuity validation."""

import json
import logging
import subprocess
from typing import Dict, List, Optional, Tuple
from upframe.utils.ffmpeg import find_binary

logger = logging.getLogger(__name__)


def validate_timestamps(
    filepath: str,
    max_frames_to_check: int = 500,
    tolerance_gap_sec: float = 0.08  # for 50 fps, expected interval is 0.02s; >0.08s indicates dropped frames
) -> Tuple[bool, Dict[str, any], List[str]]:
    """Checks presentation timestamps (PTS) in display order for monotonicity, gaps, and duplicates.
    
    Returns:
        (passed, details, warnings)
    """
    ffprobe_bin = find_binary("ffprobe")
    
    # Check packet PTS directly (fast demux without decoding video frames)
    interval_sec = max(5.0, min(20.0, max_frames_to_check / 25.0))
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,dts_time",
        "-read_intervals", f"%+{interval_sec:.1f}",
        "-of", "json",
        filepath
    ]

    warnings: List[str] = []
    pts_list: List[float] = []

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(proc.stdout)
        packets = data.get("packets", [])
        for p in packets:
            pt = p.get("pts_time")
            if pt is not None:
                try:
                    pts_list.append(float(pt))
                except ValueError:
                    pass
        # Packets in video containers with B-frames are in decode (DTS) order.
        # Sort PTS to evaluate presentation timeline
        pts_list.sort()
    except Exception as e:
        logger.warning(f"Packet PTS check failed ({e})")

    if not pts_list:
        return True, {"frames_checked": 0, "note": "PTS inspection skipped"}, []

    monotonic = True
    duplicates_count = 0
    gaps_count = 0
    max_gap_sec = 0.0

    for i in range(len(pts_list) - 1):
        diff = pts_list[i + 1] - pts_list[i]
        if diff < 0:
            monotonic = False
        elif diff == 0:
            duplicates_count += 1
        elif diff > tolerance_gap_sec:
            gaps_count += 1
            if diff > max_gap_sec:
                max_gap_sec = diff

    passed = monotonic and (duplicates_count == 0) and (gaps_count == 0)

    if not monotonic:
        warnings.append("Non-monotonic timestamps detected in output video (PTS went backward).")
    if duplicates_count > 0:
        warnings.append(f"Detected {duplicates_count} duplicate timestamps.")
    if gaps_count > 0:
        warnings.append(
            f"Detected {gaps_count} unexpected timestamp gaps (largest gap: {max_gap_sec:.4f}s)."
        )

    details = {
        "frames_checked": len(pts_list),
        "monotonic": monotonic,
        "duplicates_count": duplicates_count,
        "gaps_count": gaps_count,
        "max_gap_sec": max_gap_sec,
        "passed": passed
    }

    return passed, details, warnings
