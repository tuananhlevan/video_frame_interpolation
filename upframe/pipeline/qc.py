"""Quality Control, post-processing validation, and UPFRAME REPORT generation."""

import json
import logging
import os
from typing import List
from upframe.core.types import QCReport, VideoMetadata
from upframe.pipeline.probe import probe_video
from upframe.utils.ffmpeg import format_duration

logger = logging.getLogger(__name__)


def validate_output(
    input_meta: VideoMetadata,
    output_path: str,
    tolerance_frames: int = 5
) -> QCReport:
    """Validates that the output file exists, matches expected 50 fps timing, and has audio."""
    warnings: List[str] = []
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"Output video file is missing or empty: {output_path}")

    out_meta = probe_video(output_path)

    # Check FPS is close to 50.0
    if abs(out_meta.nominal_fps - 50.0) > 0.5 and abs(out_meta.avg_fps - 50.0) > 0.5:
        warnings.append(
            f"Output FPS ({out_meta.nominal_fps:.2f} / {out_meta.avg_fps:.2f}) differs from expected 50.0 fps"
        )

    # Check audio stream preservation
    if input_meta.has_audio and not out_meta.has_audio:
        warnings.append("Original audio was present, but output has no audio stream")

    # Check frame count
    expected_frames = input_meta.target_nb_frames
    if out_meta.nb_frames > 0 and abs(out_meta.nb_frames - expected_frames) > tolerance_frames:
        warnings.append(
            f"Frame count mismatch: expected ~{expected_frames}, got {out_meta.nb_frames}"
        )

    return QCReport(
        input_path=input_meta.filepath,
        output_path=output_path,
        resolution=f"{out_meta.width}x{out_meta.height}",
        input_fps=input_meta.nominal_fps,
        output_fps=out_meta.nominal_fps,
        duration_sec=out_meta.duration,
        duration_str=format_duration(out_meta.duration),
        model_name="unknown",
        gpus_used="0",
        source_frames=input_meta.nb_frames,
        generated_frames=max(0, out_meta.nb_frames - input_meta.nb_frames),
        total_output_frames=out_meta.nb_frames,
        scene_cuts_count=0,
        processing_time_sec=0.0,
        realtime_factor=0.0,
        audio_status="stream copied" if out_meta.has_audio else "none",
        encoding_codec=out_meta.codec_name.upper(),
        status="SUCCESS" if not warnings else "SUCCESS_WITH_WARNINGS",
        validation_passed=len(warnings) == 0,
        warnings=warnings
    )
