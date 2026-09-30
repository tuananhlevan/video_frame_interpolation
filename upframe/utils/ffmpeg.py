"""FFmpeg and FFprobe system utilities."""

import functools
import os
import shutil
import subprocess
from typing import Optional


@functools.lru_cache(maxsize=16)
def _test_nvenc_detail(resolved_bin: str) -> tuple:
    """Tests if a specific ffmpeg binary can encode with h264_nvenc on current GPU hardware."""
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
            text=True,
            timeout=6
        )
        if proc.returncode == 0:
            return True, ""
        err_msg = proc.stderr.strip() if proc.stderr else f"exit code {proc.returncode}"
        return False, err_msg
    except Exception as e:
        return False, str(e)


def _test_nvenc(resolved_bin: str) -> bool:
    ok, _ = _test_nvenc_detail(resolved_bin)
    return ok


def check_nvenc_support(ffmpeg_bin: str = "ffmpeg") -> tuple:
    """Returns (is_available, error_reason) for NVENC hardware encoding."""
    resolved = find_binary(ffmpeg_bin, prefer_nvenc=True)
    return _test_nvenc_detail(resolved)


def find_binary(binary_name: str, prefer_nvenc: bool = False) -> str:
    """Finds path to ffmpeg or ffprobe executable.
    
    If prefer_nvenc is True and binary_name is 'ffmpeg', scans all candidate
    paths and returns an NVENC-capable binary if available.
    """
    import sys
    candidates = []

    which_path = shutil.which(binary_name)
    if which_path:
        candidates.append(which_path)

    py_bin = os.path.join(sys.prefix, "bin", binary_name)
    if os.path.exists(py_bin) and os.access(py_bin, os.X_OK):
        candidates.append(py_bin)

    for p in [f"/usr/local/bin/{binary_name}", os.path.expanduser(f"~/.local/bin/{binary_name}"), f"/usr/bin/{binary_name}"]:
        if os.path.exists(p) and os.access(p, os.X_OK):
            candidates.append(p)

    seen = set()
    unique = []
    for c in candidates:
        r = os.path.realpath(c)
        if r not in seen:
            seen.add(r)
            unique.append(c)

    if prefer_nvenc and binary_name == "ffmpeg":
        for cand in unique:
            if _test_nvenc(cand):
                return cand

    return unique[0] if unique else binary_name


@functools.lru_cache(maxsize=8)
def is_nvenc_available(ffmpeg_bin: str = "ffmpeg") -> bool:
    """Checks if h264_nvenc hardware encoder is functional with current GPU and driver."""
    resolved_bin = find_binary(ffmpeg_bin, prefer_nvenc=True)
    return _test_nvenc(resolved_bin)


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
