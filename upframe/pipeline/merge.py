"""Ordered chunk merging and boundary overlap deduplication."""

import logging
import os
from typing import Generator, List
import numpy as np
from upframe.pipeline.decode import VideoDecoder
from upframe.pipeline.encode import VideoEncoder

logger = logging.getLogger(__name__)


class ChunkMerger:
    """Merges chunk frame outputs in sequential order, removing boundary duplicate frames."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def merge_chunk_frames(
        self,
        chunk_idx: int,
        frames: List[np.ndarray],
        encoder: VideoEncoder
    ) -> int:
        """Pipes frames from a completed chunk into the encoder.
        
        For chunk_idx == 0: writes all frames.
        For chunk_idx > 0: drops frames[0] (which is the duplicate overlap frame
        matching the last frame of chunk_idx - 1), then writes remaining frames.
        
        Returns count of frames written.
        """
        if not frames:
            return 0

        start_offset = 1 if chunk_idx > 0 else 0
        written = 0
        for frame in frames[start_offset:]:
            encoder.write_frame(frame)
            written += 1
        return written

    def stream_merge_chunk_files(
        self,
        chunk_files: List[str],
        encoder: VideoEncoder,
        fps: float = 50.0
    ) -> int:
        """Sequentially decodes and merges chunk video files produced by workers."""
        total_written = 0
        for idx, chunk_file in enumerate(chunk_files):
            if not os.path.exists(chunk_file):
                raise FileNotFoundError(f"Missing chunk file: {chunk_file}")
            
            decoder = VideoDecoder(chunk_file, width=self.width, height=self.height)
            frame_gen = decoder.stream_frames(start_frame=0, fps=fps)

            # Skip first frame for subsequent chunks to eliminate boundary overlap
            if idx > 0:
                try:
                    next(frame_gen)
                except StopIteration:
                    continue

            for frame in frame_gen:
                encoder.write_frame(frame)
                total_written += 1

        return total_written
