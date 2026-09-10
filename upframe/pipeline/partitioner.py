"""Chunk partitioning and overlap calculation for parallel processing."""

from typing import List
from upframe.core.types import ChunkRange, VideoMetadata


class ChunkPartitioner:
    """Partitions a video sequence into overlapping chunks for worker processing."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 1) -> None:
        if chunk_size < 2:
            raise ValueError(f"chunk_size must be >= 2, got {chunk_size}")
        if overlap < 1:
            raise ValueError(f"overlap must be >= 1, got {overlap}")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def partition(self, total_frames: int) -> List[ChunkRange]:
        """Generates a list of ChunkRange objects covering [0, total_frames - 1].
        
        Chunk i starts at start_frame and ends at end_frame.
        Neighboring chunks share `overlap` frames (default 1) at their boundary:
        Chunk i ends at frame E, Chunk i+1 starts at frame E.
        """
        if total_frames <= 0:
            return []

        chunks: List[ChunkRange] = []
        chunk_id = 0
        start = 0

        while start < total_frames - 1:
            end = min(start + self.chunk_size, total_frames - 1)
            chunks.append(ChunkRange(chunk_id=chunk_id, start_frame=start, end_frame=end))
            if end >= total_frames - 1:
                break
            # Overlap: next chunk starts at the previous end frame
            start = end
            chunk_id += 1

        if not chunks:
            chunks.append(ChunkRange(chunk_id=0, start_frame=0, end_frame=max(0, total_frames - 1)))

        return chunks

    def partition_video(self, metadata: VideoMetadata) -> List[ChunkRange]:
        """Partitions frames based on VideoMetadata."""
        total = metadata.nb_frames
        if total <= 0:
            total = int(round(metadata.duration * metadata.nominal_fps))
        return self.partition(total)
