"""Slow-motion inspection video generator (2x, 4x, 8x)."""

import os
import subprocess
from typing import Optional
from upframe.utils.ffmpeg import find_binary


def generate_slowmo_clip(
    video_path: str,
    output_path: str,
    factor: int = 2,
    start_sec: float = 0.0,
    duration_sec: float = 5.0
) -> bool:
    """Generates a slow-motion video segment (2x, 4x, 8x) for visual artifact scrutiny.
    
    factor: 2 = 2x slower, 4 = 4x slower, 8 = 8x slower.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ffmpeg_bin = find_binary("ffmpeg")

    # setpts filter: setpts=2.0*PTS slows down video by factor of 2
    pts_multiplier = float(factor)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", video_path,
        "-filter:v", f"setpts={pts_multiplier}*PTS",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-an",
        output_path
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception:
        return False
