"""Side-by-side comparison video generator (e.g. Source vs VFI or Model A vs Model B)."""

import os
import subprocess
from upframe.utils.ffmpeg import find_binary


def generate_side_by_side_video(
    video1_path: str,
    video2_path: str,
    output_path: str,
    label1: str = "Source 25fps",
    label2: str = "VFI 50fps",
    duration_sec: float = 10.0
) -> bool:
    """Combines two videos into a side-by-side labeled comparison video."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ffmpeg_bin = find_binary("ffmpeg")

    filter_complex = (
        f"[0:v]scale=960:540,drawtext=text='{label1}':x=20:y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6[v0];"
        f"[1:v]scale=960:540,drawtext=text='{label2}':x=20:y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6[v1];"
        f"[v0][v1]hstack[v]"
    )

    cmd = [
        ffmpeg_bin,
        "-y",
        "-t", str(duration_sec),
        "-i", video1_path,
        "-t", str(duration_sec),
        "-i", video2_path,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        output_path
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception:
        return False
