"""FFprobe wrapper for comprehensive video and audio stream inspection."""

import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from upframe.core.types import AudioMetadata, VideoMetadata
from upframe.utils.ffmpeg import find_binary, parse_fraction

logger = logging.getLogger(__name__)


def detect_interlacing(
    filepath: str,
    max_frames: int = 50,
    ffmpeg_bin: str = "ffmpeg"
) -> Tuple[bool, str, Dict[str, Any]]:
    """Analyzes the video using FFmpeg's idet filter to detect field interleaving (1080i/576i).

    Returns:
        (is_interlaced, field_order, idet_details)
    """
    try:
        resolved_bin = find_binary(ffmpeg_bin)
        cmd = [
            resolved_bin,
            "-hide_banner",
            "-i", filepath,
            "-filter:v", "idet",
            "-frames:v", str(max_frames),
            "-an",
            "-f", "null",
            "-"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        last_idet_line = None
        for line in proc.stderr.splitlines():
            if "Multi frame detection:" in line:
                last_idet_line = line

        if last_idet_line:
            parts = last_idet_line.split("Multi frame detection:")[-1].split()
            data: Dict[str, int] = {}
            for i in range(0, len(parts), 2):
                key = parts[i].rstrip(":")
                try:
                    data[key] = int(parts[i + 1])
                except (ValueError, IndexError):
                    pass
            tff = data.get("TFF", 0)
            bff = data.get("BFF", 0)
            prog = data.get("Progressive", 0)

            interlaced_frames = tff + bff
            is_interlaced = (interlaced_frames > prog and interlaced_frames >= 10)
            detected_field = "tff" if tff >= bff else "bff"
            order = detected_field if is_interlaced else "progressive"
            return is_interlaced, order, data
    except Exception as e:
        logger.debug(f"Interlace detection failed on {filepath}: {e}")

    return False, "progressive", {}


def probe_video(filepath: str, ffprobe_bin: str = "ffprobe", check_interlace: bool = True) -> VideoMetadata:
    """Probes video and audio streams using ffprobe."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input video not found: {filepath}")

    resolved_bin = find_binary(ffprobe_bin)
    cmd = [
        resolved_bin,
        "-v", "error",
        "-show_format",
        "-show_streams",
        "-of", "json",
        filepath
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed on {filepath}: {e.stderr}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse ffprobe json output: {e}")

    format_info = data.get("format", {})
    streams = data.get("streams", [])

    video_stream: Optional[Dict[str, Any]] = None
    audio_streams: List[AudioMetadata] = []

    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type == "video" and video_stream is None:
            video_stream = stream
        elif codec_type == "audio":
            try:
                sample_rate = int(stream.get("sample_rate", 48000))
            except ValueError:
                sample_rate = 48000
            try:
                channels = int(stream.get("channels", 2))
            except ValueError:
                channels = 2
            try:
                a_duration = float(stream.get("duration", 0.0))
            except (ValueError, TypeError):
                a_duration = None
            try:
                a_bitrate = int(stream.get("bit_rate", 0))
            except (ValueError, TypeError):
                a_bitrate = None
            try:
                a_nb_frames = int(stream.get("nb_frames", 0))
            except (ValueError, TypeError):
                a_nb_frames = None

            audio_streams.append(
                AudioMetadata(
                    index=int(stream.get("index", 0)),
                    codec_name=stream.get("codec_name", "aac"),
                    sample_rate=sample_rate,
                    channels=channels,
                    channel_layout=stream.get("channel_layout"),
                    duration=a_duration,
                    bit_rate=a_bitrate,
                    nb_frames=a_nb_frames
                )
            )

    if video_stream is None:
        raise ValueError(f"No video stream found in {filepath}")

    width = int(video_stream.get("width", 1920))
    height = int(video_stream.get("height", 1080))
    codec_name = video_stream.get("codec_name", "h264")
    pix_fmt = video_stream.get("pix_fmt", "yuv420p")
    color_space = video_stream.get("color_space")
    color_primaries = video_stream.get("color_primaries")
    color_transfer = video_stream.get("color_transfer")
    color_range = video_stream.get("color_range")

    raw_field_order = video_stream.get("field_order")
    is_interlaced = False
    field_order = "progressive"
    interlace_details: Dict[str, Any] = {}

    if raw_field_order in ("tt", "bb", "tb", "bt"):
        is_interlaced = True
        field_order = "tff" if raw_field_order in ("tt", "tb") else "bff"
        interlace_details = {"container_field_order": raw_field_order}
    elif check_interlace:
        is_interlaced, field_order, interlace_details = detect_interlacing(filepath)

    r_frame_rate = video_stream.get("r_frame_rate", "25/1")
    avg_frame_rate = video_stream.get("avg_frame_rate", "25/1")
    nominal_fps = parse_fraction(r_frame_rate, 25.0)
    avg_fps = parse_fraction(avg_frame_rate, nominal_fps)

    # Duration parsing
    try:
        duration = float(video_stream.get("duration", 0.0))
    except (ValueError, TypeError):
        try:
            duration = float(format_info.get("duration", 0.0))
        except (ValueError, TypeError):
            duration = 0.0

    # Frame count parsing
    nb_frames = 0
    if "nb_frames" in video_stream:
        try:
            nb_frames = int(video_stream["nb_frames"])
        except ValueError:
            pass

    # Validate container nb_frames against duration * fps (handles clipped videos with stale atom headers)
    fps_for_est = nominal_fps if nominal_fps > 0 else avg_fps
    if duration > 0 and fps_for_est > 0:
        duration_est_frames = int(round(duration * fps_for_est))
        if nb_frames == 0:
            nb_frames = duration_est_frames
        elif abs(nb_frames - duration_est_frames) > max(3, int(fps_for_est * 0.2)):
            logger.warning(
                f"Container metadata nb_frames ({nb_frames}) differs significantly from "
                f"duration-based frame count ({duration_est_frames} frames for {duration:.2f}s @ {fps_for_est:.2f} fps). "
                f"Reconciling to {duration_est_frames}."
            )
            nb_frames = duration_est_frames

    time_base = video_stream.get("time_base", "1/1000")
    try:
        bit_rate = int(format_info.get("bit_rate", 0))
    except (ValueError, TypeError):
        bit_rate = None

    try:
        file_size_bytes = int(format_info.get("size", os.path.getsize(filepath)))
    except (ValueError, TypeError):
        file_size_bytes = os.path.getsize(filepath)

    return VideoMetadata(
        filepath=os.path.abspath(filepath),
        width=width,
        height=height,
        codec_name=codec_name,
        pix_fmt=pix_fmt,
        color_space=color_space,
        color_primaries=color_primaries,
        color_transfer=color_transfer,
        color_range=color_range,
        r_frame_rate=r_frame_rate,
        avg_frame_rate=avg_frame_rate,
        nominal_fps=nominal_fps,
        avg_fps=avg_fps,
        duration=duration,
        nb_frames=nb_frames,
        time_base=time_base,
        bit_rate=bit_rate,
        file_size_bytes=file_size_bytes,
        audio_streams=audio_streams,
        field_order=field_order,
        is_interlaced=is_interlaced,
        interlace_details=interlace_details
    )
