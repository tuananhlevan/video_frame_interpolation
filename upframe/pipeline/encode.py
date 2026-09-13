"""FFmpeg video encoder for 50 fps output with audio stream-copy and NVENC/x264 support."""

import logging
import os
import subprocess
from typing import Optional
import numpy as np
from upframe.utils.ffmpeg import find_binary, is_nvenc_available

logger = logging.getLogger(__name__)


class VideoEncoder:
    """Pipes raw 50 fps frames into FFmpeg for high-fidelity encoding and audio stream-copy."""

    def __init__(
        self,
        output_filepath: str,
        width: int,
        height: int,
        fps: float = 50.0,
        source_audio_path: Optional[str] = None,
        crf: int = 18,
        preset: str = "medium",
        use_nvenc: Optional[bool] = None,
        bitrate: Optional[str] = None,
        ffmpeg_bin: str = "ffmpeg"
    ) -> None:
        self.output_filepath = os.path.abspath(output_filepath)
        self.width = width
        self.height = height
        self.fps = fps
        self.source_audio_path = os.path.abspath(source_audio_path) if source_audio_path else None
        self.crf = crf
        self.preset = preset
        self.ffmpeg_bin = find_binary(ffmpeg_bin)
        self.frame_bytes = width * height * 3
        self.bitrate = bitrate

        if use_nvenc is True:
            if not is_nvenc_available(self.ffmpeg_bin):
                logger.warning(
                    f"NVIDIA NVENC requested, but 'h264_nvenc' is not supported by '{self.ffmpeg_bin}'. "
                    "Falling back to software encoder (libx264)."
                )
                self.use_nvenc = False
            else:
                self.use_nvenc = True
        elif use_nvenc is None:
            self.use_nvenc = is_nvenc_available(self.ffmpeg_bin)
        else:
            self.use_nvenc = False

        self.proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        """Starts the FFmpeg encoding subprocess."""
        os.makedirs(os.path.dirname(self.output_filepath), exist_ok=True)

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-v", "warning",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-i", "pipe:0"
        ]

        if self.source_audio_path and os.path.exists(self.source_audio_path):
            cmd.extend(["-i", self.source_audio_path])

        if self.use_nvenc:
            logger.info("Using NVIDIA NVENC hardware encoder (h264_nvenc)")
            cmd.extend([
                "-c:v", "h264_nvenc",
                "-preset", "p5",
                "-cq", str(self.crf),
                "-b:v", self.bitrate if self.bitrate else "0"
            ])
        else:
            logger.info("Using software encoder (libx264)")
            cmd.extend([
                "-c:v", "libx264",
                "-preset", self.preset,
                "-crf", str(self.crf)
            ])

        # Standard broadcast video formatting: yuv420p + BT.709 color tags
        cmd.extend([
            "-pix_fmt", "yuv420p",
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709"
        ])

        # Audio stream-copy
        if self.source_audio_path and os.path.exists(self.source_audio_path):
            cmd.extend([
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:a", "copy"
            ])

        cmd.append(self.output_filepath)

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    def write_frame(self, frame: np.ndarray) -> None:
        """Writes a single (H, W, 3) RGB uint8 frame to the encoder pipe."""
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("Encoder has not been started.")

        if self.proc.poll() is not None:
            stderr_msg = ""
            if self.proc.stderr:
                try:
                    stderr_msg = self.proc.stderr.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
            raise RuntimeError(
                f"FFmpeg process terminated unexpectedly with exit code {self.proc.returncode}: {stderr_msg.strip()}"
            )

        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)

        try:
            self.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, IOError, OSError) as e:
            stderr_msg = ""
            if self.proc.stderr:
                try:
                    stderr_msg = self.proc.stderr.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
            ret = self.proc.poll()
            raise RuntimeError(
                f"FFmpeg encoding pipe broken (exit code {ret}): {stderr_msg.strip() or str(e)}"
            ) from e

    def finish(self) -> None:
        """Closes the encoder pipe and waits for FFmpeg to finalize the container."""
        if self.proc is not None:
            stdout, stderr = self.proc.communicate()
            if self.proc.returncode != 0:
                logger.error(f"FFmpeg encoding error: {stderr.decode('utf-8', errors='replace')}")
                raise RuntimeError(f"FFmpeg encoding failed with exit code {self.proc.returncode}")
            self.proc = None
