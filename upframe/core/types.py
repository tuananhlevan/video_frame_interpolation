"""Unified domain types and dataclasses for the upframe pipeline."""

from dataclasses import asdict, dataclass, field
import os
from typing import Any, Dict, List, Optional


@dataclass
class AudioMetadata:
    """Metadata for an audio stream within a video container."""
    index: int
    codec_name: str
    sample_rate: int
    channels: int
    channel_layout: Optional[str] = None
    duration: Optional[float] = None
    bit_rate: Optional[int] = None
    nb_frames: Optional[int] = None


@dataclass
class VideoMetadata:
    """Metadata for a video file and its streams."""
    filepath: str
    width: int
    height: int
    codec_name: str
    pix_fmt: str
    color_space: Optional[str] = None
    color_primaries: Optional[str] = None
    color_transfer: Optional[str] = None
    color_range: Optional[str] = None
    r_frame_rate: str = "25/1"
    avg_frame_rate: str = "25/1"
    nominal_fps: float = 25.0
    avg_fps: float = 25.0
    duration: float = 0.0
    nb_frames: int = 0
    time_base: str = "1/1000"
    bit_rate: Optional[int] = None
    file_size_bytes: int = 0
    audio_streams: List[AudioMetadata] = field(default_factory=list)
    field_order: Optional[str] = None
    is_interlaced: bool = False
    interlace_details: Optional[Dict[str, Any]] = None

    @property
    def has_audio(self) -> bool:
        return len(self.audio_streams) > 0

    @property
    def target_fps(self) -> float:
        """Target output frame rate (nominally 50.0 fps)."""
        return 50.0

    @property
    def target_nb_frames(self) -> int:
        """Expected output frame count for 2x upframing (2 * N - 1)."""
        if self.nb_frames <= 0:
            return 0
        return 2 * self.nb_frames - 1


@dataclass(frozen=True)
class ChunkRange:
    """Defines a slice of frames for a chunk, including boundary overlap."""
    chunk_id: int
    start_frame: int
    end_frame: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass
class ChunkTask:
    """Task specification dispatched to a worker."""
    chunk_id: int
    start_frame: int
    end_frame: int
    input_path: str
    output_dir: str
    width: int
    height: int
    fps: float
    model_name: str = "rife"
    device: str = "cuda:0"
    checkpoint_path: Optional[str] = None
    scene_threshold: float = 0.35
    fp16: bool = True
    color_space: Optional[str] = None
    color_primaries: Optional[str] = None
    color_transfer: Optional[str] = None
    color_range: Optional[str] = None


@dataclass
class ChunkResult:
    """Execution result returned by a worker after processing a chunk."""
    chunk_id: int
    status: str  # "COMPLETED" or "FAILED"
    output_file: Optional[str]
    source_frames_count: int
    output_frames_count: int
    scene_cuts_count: int
    elapsed_seconds: float
    error_message: Optional[str] = None


@dataclass
class QCReport:
    """Quality control verification and final processing summary."""
    input_path: str
    output_path: str
    resolution: str
    input_fps: float
    output_fps: float
    duration_sec: float
    duration_str: str
    model_name: str
    gpus_used: str
    source_frames: int
    generated_frames: int
    total_output_frames: int
    scene_cuts_count: int
    processing_time_sec: float
    realtime_factor: float
    audio_status: str = "stream copied"
    encoding_codec: str = "H.264"
    status: str = "SUCCESS"
    validation_passed: bool = True
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def render_text(self) -> str:
        lines = [
            "=" * 50,
            "               UPFRAME REPORT",
            "=" * 50,
            f"Input:             {os.path.basename(self.input_path)}",
            f"Output:            {os.path.basename(self.output_path)}",
            "",
            f"Resolution:        {self.resolution}",
            f"Input FPS:         ~{self.input_fps:.2f} fps",
            f"Output FPS:        {self.output_fps:.2f} fps",
            f"Duration:          {self.duration_str}",
            "",
            f"Model:             {self.model_name.upper()}",
            f"GPUs:              {self.gpus_used}",
            "",
            f"Source frames:     {self.source_frames:,}",
            f"Generated frames:  {self.generated_frames:,}",
            f"Total out frames:  {self.total_output_frames:,}",
            f"Scene cuts:        {self.scene_cuts_count}",
            "",
            f"Processing time:   {self.duration_str if hasattr(self, '_fmt_time') else f'{self.processing_time_sec:.1f}s'}",
            f"Realtime factor:   {self.realtime_factor:.2f}x",
            "",
            f"Audio:             {self.audio_status}",
            f"Encoding:          {self.encoding_codec}",
            f"Status:            {self.status}",
            "=" * 50
        ]
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
            lines.append("=" * 50)
        return "\n".join(lines)
