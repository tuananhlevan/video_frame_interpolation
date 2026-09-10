"""Core domain types and configuration for upframe."""

from upframe.core.types import (
    AudioMetadata,
    VideoMetadata,
    ChunkRange,
    ChunkTask,
    ChunkResult,
    QCReport,
)
from upframe.core.config import (
    PipelineConfig,
    FOOTBALL_PRESET,
)

__all__ = [
    "AudioMetadata",
    "VideoMetadata",
    "ChunkRange",
    "ChunkTask",
    "ChunkResult",
    "QCReport",
    "PipelineConfig",
    "FOOTBALL_PRESET",
]
