"""FFmpeg-based frame decoder streaming raw RGB frames directly to memory/workers."""

import logging
import os
import subprocess
from typing import Generator, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class VideoDecoder:
    """Streams decoded RGB24 frames from video using FFmpeg rawvideo pipes."""

    def __init__(self, filepath: str, width: int, height: int, ffmpeg_bin: str = "ffmpeg") -> None:
        self.filepath = os.path.abspath(filepath)
        self.width = width
        self.height = height
        self.ffmpeg_bin = ffmpeg_bin
        self.frame_bytes = width * height * 3

    def decode_chunk(
        self,
        start_frame: int,
        count: int,
        fps: float = 25.0
    ) -> List[np.ndarray]:
        """Decodes an exact sequence of frames for a chunk starting at start_frame."""
        frames: List[np.ndarray] = []
        for frame in self.stream_frames(start_frame=start_frame, count=count, fps=fps):
            frames.append(frame)
        return frames

    def stream_frames(
        self,
        start_frame: int = 0,
        count: Optional[int] = None,
        fps: float = 25.0
    ) -> Generator[np.ndarray, None, None]:
        """Streams decoded raw RGB24 frames as (H, W, 3) uint8 numpy arrays."""
        cmd = [self.ffmpeg_bin, "-v", "error"]

        if start_frame > 0:
            start_sec = max(0.0, start_frame / fps)
            if start_sec > 10.0:
                # Two-stage seek: fast coarse seek, then frame-accurate fine seek
                cmd.extend(["-ss", f"{max(0.0, start_sec - 3.0):.4f}"])
                cmd.extend(["-i", self.filepath])
                cmd.extend(["-ss", f"{min(start_sec, 3.0):.4f}"])
            else:
                # Precise seek after -i
                cmd.extend(["-i", self.filepath])
                cmd.extend(["-ss", f"{start_sec:.4f}"])
        else:
            cmd.extend(["-i", self.filepath])

        if count is not None and count > 0:
            cmd.extend(["-frames:v", str(count)])

        cmd.extend([
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1"
        ])

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10 * self.frame_bytes
        )

        try:
            read_count = 0
            while True:
                if count is not None and read_count >= count:
                    break
                raw = proc.stdout.read(self.frame_bytes)
                if len(raw) < self.frame_bytes:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
                yield frame
                read_count += 1
        finally:
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
            proc.kill()
            proc.wait()
