"""UPFRAME Video Frame Interpolation Evaluation System.

Faithfully implements the production evaluation strategy for 25 FPS -> 50 FPS
football broadcast video outlined in upframe_video_evaluation_strategy.txt and
upframe_complete_pipeline_and_evaluation_spec.txt.
"""

from eval.config import EvaluationConfig, FootballScoreWeights, TechnicalThresholds, PerformanceThresholds
from eval.pipeline import EvaluationPipeline
from eval.types import (
    ArtifactDifferentialResult,
    ArtifactMetric,
    EvaluationReport,
    FootballQCResult,
    GroundTruthQCResult,
    PerceptualQCResult,
    PerformanceQCResult,
    ScorecardResult,
    SuspiciousMoment,
    TechnicalQCResult,
    TemporalQCResult
)

__all__ = [
    "EvaluationPipeline",
    "EvaluationConfig",
    "EvaluationReport",
    "FootballScoreWeights",
    "TechnicalThresholds",
    "PerformanceThresholds",
    "TechnicalQCResult",
    "TemporalQCResult",
    "FootballQCResult",
    "ArtifactDifferentialResult",
    "ArtifactMetric",
    "PerceptualQCResult",
    "PerformanceQCResult",
    "ScorecardResult",
    "SuspiciousMoment",
    "GroundTruthQCResult"
]
