"""Scene cut validation: ensures no synthetic hybrid frames across camera cuts."""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from upframe.pipeline.decode import VideoDecoder
from upframe.pipeline.probe import probe_video
from upframe.pipeline.scene_detect import compute_frame_difference


def validate_scene_cut_handling(
    source_path: str,
    output_path: str,
    cut_threshold: float = 0.35,
    max_frames_to_scan: Optional[int] = 500
) -> Tuple[bool, Dict[str, any], List[str]]:
    """Detects hard cuts in source and verifies intermediate frames in output are not blended hybrids.
    
    Returns:
        (properly_handled, details, warnings)
    """
    src_meta = probe_video(source_path)
    out_meta = probe_video(output_path)

    cuts_detected: List[int] = []
    hybrid_failures: List[Dict[str, any]] = []
    warnings: List[str] = []

    dec_src = VideoDecoder(source_path, src_meta.width, src_meta.height)
    dec_out = VideoDecoder(output_path, out_meta.width, out_meta.height)

    stream_src = dec_src.stream_frames()
    stream_out = dec_out.stream_frames()

    prev_frame_src: Optional[np.ndarray] = None
    prev_out_odd: Optional[np.ndarray] = None
    src_idx = 0

    try:
        while True:
            try:
                frame_src = next(stream_src)
            except StopIteration:
                break

            # Output has 2 frames: even (source-aligned) and odd (intermediate)
            try:
                frame_out_even = next(stream_out)
            except StopIteration:
                frame_out_even = None

            try:
                frame_out_odd = next(stream_out)
            except StopIteration:
                frame_out_odd = None

            if prev_frame_src is not None:
                # Check if transition between src_idx - 1 and src_idx is a cut
                diff = compute_frame_difference(prev_frame_src, frame_src)
                if diff >= cut_threshold:
                    cut_idx = src_idx - 1
                    cuts_detected.append(cut_idx)

                    # The intermediate generated frame between cut_idx and src_idx is prev_out_odd
                    # (at output frame index 2 * cut_idx + 1)
                    if prev_out_odd is not None:
                        frame_inter = prev_out_odd
                        if frame_inter.shape != frame_src.shape:
                            frame_inter = cv2.resize(frame_inter, (frame_src.shape[1], frame_src.shape[0]))

                        diff_to_a = compute_frame_difference(frame_inter, prev_frame_src)
                        diff_to_b = compute_frame_difference(frame_inter, frame_src)

                        ideal_synthetic_blend = (prev_frame_src.astype(np.float32) * 0.5 + frame_src.astype(np.float32) * 0.5).astype(np.uint8)
                        diff_to_blend = compute_frame_difference(frame_inter, ideal_synthetic_blend)

                        is_hybrid = (diff_to_a > 0.20 and diff_to_b > 0.20) or (diff_to_blend < 0.12 and diff >= cut_threshold)
                        if is_hybrid:
                            target_intermediate_idx = 2 * cut_idx + 1
                            hybrid_failures.append({
                                "cut_source_frame": cut_idx,
                                "output_frame": target_intermediate_idx,
                                "diff_ab": diff,
                                "diff_inter_to_a": diff_to_a,
                                "diff_inter_to_b": diff_to_b
                            })
                            warnings.append(
                                f"Interpolation across hard cut detected at source frame {cut_idx} -> {cut_idx + 1} "
                                f"(output frame {target_intermediate_idx} is a synthetic hybrid frame)."
                            )

            prev_frame_src = frame_src.copy()
            prev_out_odd = frame_out_odd.copy() if frame_out_odd is not None else None
            src_idx += 1

            if max_frames_to_scan and src_idx >= max_frames_to_scan:
                break
    except Exception as e:
        warnings.append(f"Scene cut scanning interrupted by error: {e}")

    properly_handled = len(hybrid_failures) == 0
    details = {
        "source_cuts_detected": len(cuts_detected),
        "cut_indices": cuts_detected,
        "hybrid_failures_count": len(hybrid_failures),
        "hybrid_failures": hybrid_failures,
        "properly_handled": properly_handled
    }

    return properly_handled, details, warnings
