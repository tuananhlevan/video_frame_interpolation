"""FFmpeg and FFprobe system utilities."""

import os
import shutil
import subprocess
from typing import Optional


def find_binary(binary_name: str) -> str:
    """Finds path to ffmpeg or ffprobe executable."""
    path = shutil.which(binary_name)
    if path is not None:
        return path
    # Common fallbacks
    for candidate in [f"/usr/bin/{binary_name}", f"/usr/local/bin/{binary_name}"]:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return binary_name


def is_nvenc_available(ffmpeg_bin: str = "ffmpeg") -> bool:
    """Checks if h264_nvenc hardware encoder is functional with current GPU and driver."""
    resolved_bin = find_binary(ffmpeg_bin)
    try:
        proc = subprocess.run(
            [
                resolved_bin,
                "-v", "error",
                "-f", "lavfi",
                "-i", "color=s=64x64",
                "-c:v", "h264_nvenc",
                "-frames:v", "1",
                "-f", "null",
                "-"
            ],
            capture_output=True,
            text=True
        )
        return proc.returncode == 0
    except Exception:
        return False


def format_duration(seconds: float) -> str:
    """Formats duration in seconds to HH:MM:SS or MM:SS."""
    total_sec = max(0, int(round(seconds)))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_fraction(val_str: str, default: float = 25.0) -> float:
    """Parses fraction strings like '25/1' or '6480/259' into float."""
    try:
        if "/" in val_str:
            num, den = val_str.split("/")
            den_val = float(den)
            return float(num) / den_val if den_val != 0 else default
        return float(val_str)
    except Exception:
        return default
