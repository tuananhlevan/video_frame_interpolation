"""FPS and Frame Count validation for 25 FPS -> 50 FPS upframing."""

from typing import Dict, List, Tuple
from upframe.core.types import VideoMetadata


def validate_fps_and_frames(
    source_meta: VideoMetadata,
    output_meta: VideoMetadata,
    tolerance_frames: int = 5,
    target_fps: float = 50.0
) -> Tuple[bool, bool, Dict[str, any], List[str]]:
    """Validates FPS transition (~25 to ~50 fps) and expected frame counts.

    Returns:
        (fps_passed, frame_count_passed, details, warnings)
    """
    warnings: List[str] = []
    
    # 1. FPS validation: output should be close to 50.0 fps (or ~2x source nominal)
    expected_fps = source_meta.nominal_fps * 2.0 if abs(source_meta.nominal_fps - 25.0) < 1.0 else target_fps
    fps_delta = abs(output_meta.nominal_fps - expected_fps)
    avg_fps_delta = abs(output_meta.avg_fps - expected_fps)
    
    fps_passed = (fps_delta <= 1.0) or (avg_fps_delta <= 1.0)
    if not fps_passed:
        warnings.append(
            f"Output FPS ({output_meta.nominal_fps:.2f} nominal, {output_meta.avg_fps:.2f} avg) "
            f"deviates from expected {expected_fps:.2f} FPS (delta: {fps_delta:.2f})"
        )

    # 2. Frame count validation: for 2x interpolation, expected = 2 * N_src - 1 (or 2 * N_src depending on container)
    expected_frames = source_meta.target_nb_frames
    if expected_frames <= 0 and source_meta.duration > 0:
        expected_frames = int(round(source_meta.duration * expected_fps))

    frame_count_diff = output_meta.nb_frames - expected_frames
    frame_count_passed = abs(frame_count_diff) <= tolerance_frames

    if not frame_count_passed:
        if frame_count_diff < 0:
            warnings.append(
                f"Missing frames detected: expected ~{expected_frames:,}, got {output_meta.nb_frames:,} "
                f"({abs(frame_count_diff)} frames dropped/missing)"
            )
        else:
            warnings.append(
                f"Extra frames detected: expected ~{expected_frames:,}, got {output_meta.nb_frames:,} "
                f"({frame_count_diff} duplicate/extra frames)"
            )

    details = {
        "source_nominal_fps": source_meta.nominal_fps,
        "source_avg_fps": source_meta.avg_fps,
        "output_nominal_fps": output_meta.nominal_fps,
        "output_avg_fps": output_meta.avg_fps,
        "source_nb_frames": source_meta.nb_frames,
        "output_nb_frames": output_meta.nb_frames,
        "expected_nb_frames": expected_frames,
        "frame_count_diff": frame_count_diff,
        "fps_passed": fps_passed,
        "frame_count_passed": frame_count_passed
    }

    return fps_passed, frame_count_passed, details, warnings
