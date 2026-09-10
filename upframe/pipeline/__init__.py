"""Pipeline components for probing, decoding, timestamping, scheduling, and encoding."""

from upframe.core.types import AudioMetadata, VideoMetadata, ChunkRange, QCReport
from upframe.pipeline.probe import probe_video
from upframe.pipeline.timestamps import TimestampManager
from upframe.pipeline.scene_detect import compute_frame_difference, detect_cuts, is_scene_cut
from upframe.pipeline.decode import VideoDecoder
from upframe.pipeline.encode import VideoEncoder
from upframe.pipeline.partitioner import ChunkPartitioner
from upframe.pipeline.state import StateStore
from upframe.pipeline.merge import ChunkMerger
from upframe.pipeline.qc import validate_output
from upframe.pipeline.scheduler import PipelineScheduler
from upframe.utils.ffmpeg import format_duration, is_nvenc_available

__all__ = [
    "AudioMetadata",
    "VideoMetadata",
    "ChunkRange",
    "probe_video",
    "TimestampManager",
    "compute_frame_difference",
    "detect_cuts",
    "is_scene_cut",
    "VideoDecoder",
    "VideoEncoder",
    "is_nvenc_available",
    "ChunkPartitioner",
    "StateStore",
    "ChunkMerger",
    "QCReport",
    "validate_output",
    "format_duration",
    "PipelineScheduler",
]
