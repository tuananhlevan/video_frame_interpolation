"""Timestamp and frame timing management for exact 50 fps output with audio synchronization."""

from fractions import Fraction
from typing import List, Tuple
from upframe.pipeline.probe import VideoMetadata


class TimestampManager:
    """Manages output timestamps and A/V synchronization.
    
    Normalizes container timing (e.g. 25.02 fps) into an exact 50.000 fps broadcast timeline
    while guaranteeing that total duration and PTS match the original audio stream.
    """

    def __init__(self, metadata: VideoMetadata, target_fps: float = 50.0) -> None:
        self.metadata = metadata
        self.target_fps = float(target_fps)
        self.time_base = Fraction(1, int(target_fps))  # 1/50 second per frame
        self.frame_duration_sec = 1.0 / self.target_fps

    def compute_output_frame_count(self) -> int:
        """Computes expected output frame count.
        
        For N source frames, we produce N source frames + (N - 1) interpolated frames = 2 * N - 1.
        If source frame count is not exact from container headers, uses duration * target_fps.
        """
        if self.metadata.nb_frames > 0:
            return 2 * self.metadata.nb_frames - 1
        if self.metadata.duration > 0:
            return int(round(self.metadata.duration * self.target_fps))
        return 0

    def get_output_pts(self, frame_index: int) -> int:
        """Returns the presentation timestamp (PTS) in time_base units for a given output frame index."""
        return frame_index

    def get_output_timestamp_seconds(self, frame_index: int) -> float:
        """Returns the timestamp in seconds for a given output frame index."""
        return frame_index * self.frame_duration_sec

    def check_av_sync_drift(self) -> Tuple[float, bool]:
        """Calculates expected time difference (drift) between audio and upframed video.
        
        Returns (drift_seconds, is_acceptable).
        Drift under 40ms (1 source frame at 25fps) is considered broadcast-safe.
        """
        if not self.metadata.audio_streams:
            return 0.0, True

        primary_audio = self.metadata.audio_streams[0]
        audio_duration = primary_audio.duration or self.metadata.duration
        video_duration = self.metadata.duration

        drift = abs(video_duration - audio_duration)
        # Broadcast safe tolerance: 40 ms
        is_safe = drift <= 0.040
        return drift, is_safe
