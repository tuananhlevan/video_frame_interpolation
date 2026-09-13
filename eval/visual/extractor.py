"""Extracts newly generated intermediate frames (odd frames) to disk."""

import os
from typing import List
import cv2
from upframe.pipeline.decode import VideoDecoder
from upframe.pipeline.probe import probe_video


def extract_injected_frames(
    output_video_path: str,
    save_dir: str,
    max_frames: int = 50,
    stride: int = 1
) -> List[str]:
    """Extracts injected intermediate frames (at indices 1, 3, 5...) for visual inspection."""
    os.makedirs(save_dir, exist_ok=True)
    meta = probe_video(output_video_path)

    saved_paths: List[str] = []
    frame_idx = 0
    saved_count = 0

    try:
        decoder = VideoDecoder(output_video_path, meta.width, meta.height)
        for frame in decoder.stream_frames():
            # Injected frames are at odd indices in 50 FPS sequence
            if frame_idx % 2 == 1:
                if (frame_idx // 2) % stride == 0:
                    filename = f"injected_frame_{frame_idx:06d}.jpg"
                    out_file = os.path.join(save_dir, filename)
                    # Convert RGB to BGR for cv2.imwrite
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(out_file, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_paths.append(out_file)
                    saved_count += 1
                    if saved_count >= max_frames:
                        break
            frame_idx += 1
    except Exception:
        pass

    return saved_paths
