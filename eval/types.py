"""Domain types and dataclasses for the upframe evaluation pipeline."""

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Dict, List, Optional


@dataclass
class TechnicalQCResult:
    """Layer 1: Technical & Pipeline Integrity validation results."""
    source_fps: float
    output_fps: float
    source_nb_frames: int
    output_nb_frames: int
    expected_nb_frames: int
    frame_count_diff: int
    fps_check_passed: bool
    frame_count_passed: bool
    source_preservation_psnr: float
    source_preservation_ssim: float
    source_preservation_mae: float
    source_preservation_max_diff: float
    source_preservation_passed: bool
    audio_present: bool
    audio_duration_diff_sec: float
    audio_sync_passed: bool
    pts_monotonic: bool
    pts_gaps_detected: int
    pts_duplicates_detected: int
    pts_check_passed: bool
    scene_cuts_detected_source: int
    scene_cuts_properly_handled: bool
    status: str  # PASS, WARN, FAIL
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status in ("PASS", "WARN")


@dataclass
class TemporalQCResult:
    """Layer 2: Temporal / VFI Quality metrics."""
    motion_smoothness_score: float  # [0.0, 1.0], higher = smoother/more continuous
    motion_discontinuity_count: int
    mean_warping_error: float  # [0.0, 255.0]
    p95_warping_error: float
    motion_boundary_error: float
    temporal_flicker_score: float  # lower = less flicker
    odd_even_oscillation_index: float  # lower = less shimmer
    status: str  # EXCELLENT, GOOD, FAIR, POOR
    warnings: List[str] = field(default_factory=list)


@dataclass
class ArtifactMetric:
    """Metrics for an individual artifact class."""
    source_level: float  # [0.0, 1.0] or percentage
    output_level: float
    added_level: float  # max(0.0, output - source)
    severity_level: int  # 1 (invisible) to 5 (critical)


@dataclass
class ArtifactDifferentialResult:
    """Source vs output artifact differential breakdown."""
    ghosting: ArtifactMetric
    double_contour: ArtifactMetric
    edge_tearing: ArtifactMetric
    deformation: ArtifactMetric
    temporal_flicker: ArtifactMetric
    motion_regime_breakdown: Dict[str, float] = field(default_factory=dict)
    # e.g. {"low_motion_added": 0.002, "med_motion_added": 0.008, "high_motion_added": 0.035}


@dataclass
class FootballQCResult:
    """Layer 3: Football-Specific Quality evaluations (1-5 scale)."""
    ball_integrity_score: float  # 1 (severe fail) to 5 (clean)
    player_integrity_score: float  # 1 to 5
    occlusion_handling_score: float  # 1 to 5
    pitch_geometry_score: float  # 1 to 5
    goal_net_score: float  # 1 to 5
    broadcast_graphics_score: float  # 1 to 5
    camera_motion_score: float  # 1 to 5
    detected_ball_teleportations: int = 0
    detected_duplicate_balls: int = 0
    detected_bent_lines_count: int = 0
    graphics_jitter_detected: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerceptualQCResult:
    """Layer 4: Human Perceptual MOS and survey results (1-5 scale)."""
    overall_quality_mos: float  # 1 to 5
    motion_naturalness_mos: float  # 1 to 5
    artifact_free_mos: float  # 1 to 5
    pairwise_preference_pct: Optional[float] = None  # % preferred vs baseline
    survey_responses_count: int = 0


@dataclass
class PerformanceQCResult:
    """Production Performance measurements."""
    total_processing_time_sec: Optional[float] = None
    source_duration_sec: float = 0.0
    realtime_factor: Optional[float] = None  # RTF = processing_time / source_duration
    target_rtf_achieved: Optional[bool] = None  # 1.0 <= RTF <= 1.3
    interpolations_per_sec: Optional[float] = None
    frames_per_sec: Optional[float] = None
    peak_gpu_memory_mb: Optional[float] = None
    avg_gpu_utilization_pct: Optional[float] = None
    cpu_utilization_pct: Optional[float] = None
    performance_score: Optional[float] = None  # 0 to 10 scale
    is_measured: bool = False


@dataclass
class SuspiciousMoment:
    """A detected anomaly or bad moment with timestamp."""
    timestamp_sec: float
    timestamp_str: str  # HH:MM:SS.mmm
    frame_index: int
    anomaly_type: str  # "ball_artifact", "occlusion_artifact", "pitch_distortion", "flow_discontinuity", "cut_hybrid"
    severity: int  # 1 to 5
    anomaly_score: float
    description: str


@dataclass
class ScorecardResult:
    """Final high-level production scorecard."""
    model_name: str
    quality_score: float  # 0 to 10 scale
    performance_score: Optional[float] = None  # 0 to 10 scale, or None if unmeasured
    realtime_factor: Optional[float] = None
    human_mos: float = 4.0
    mos_is_surveyed: bool = False
    technical_status: str = "PASS"  # PASS, WARN, FAIL
    quality_status: str = "GOOD"  # EXCELLENT, GOOD, ACCEPTABLE, POOR, REJECTED
    performance_status: str = "NOT MEASURED"  # PASS, WARN, FAIL, NOT MEASURED
    recommendation: str = ""  # PRODUCTION CANDIDATE, ACCEPTABLE WITH RESERVATIONS, REJECTED


@dataclass
class GroundTruthQCResult:
    """Metrics calculated when true high-FPS ground truth is available."""
    psnr: float
    ssim: float
    ms_ssim: Optional[float] = None
    lpips: Optional[float] = None
    mae: float = 0.0
    mse: float = 0.0
    warping_error: float = 0.0


@dataclass
class EvaluationReport:
    """Complete, self-contained evaluation report for a video or model."""
    source_file: str
    output_file: str
    model_name: str
    resolution: str
    source_duration_str: str
    output_duration_str: str
    technical_qc: TechnicalQCResult
    temporal_qc: TemporalQCResult
    artifact_diff: ArtifactDifferentialResult
    football_qc: FootballQCResult
    perceptual_qc: PerceptualQCResult
    performance_qc: PerformanceQCResult
    scorecard: ScorecardResult
    suspicious_moments: List[SuspiciousMoment] = field(default_factory=list)
    ground_truth_qc: Optional[GroundTruthQCResult] = None
    eval_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def render_summary_text(self) -> str:
        perf_score_str = f"{self.scorecard.performance_score:.1f} / 10.0 ({self.scorecard.performance_status})" if self.scorecard.performance_score is not None else f"N/A ({self.scorecard.performance_status})"
        rtf_str = f"{self.scorecard.realtime_factor:.2f}x (Target: 1.0x - 1.3x)" if self.scorecard.realtime_factor is not None else "N/A (Offline eval; pass --processing-time for RTF)"
        mos_label = "Human MOS (Surveyed):" if self.scorecard.mos_is_surveyed else "Predicted MOS (Proxy):"

        lines = [
            "=" * 60,
            "                   UPFRAME EVALUATION REPORT",
            "=" * 60,
            f"Source Video:        {os.path.basename(self.source_file)}",
            f"Output Video:        {os.path.basename(self.output_file)}",
            f"Model:               {self.model_name.upper()}",
            f"Resolution:          {self.resolution}",
            f"Duration:            {self.output_duration_str}",
            "-" * 60,
            "SCORECARD",
            f"  Quality Score:     {self.scorecard.quality_score:.1f} / 10.0 ({self.scorecard.quality_status})",
            f"  Performance Score: {perf_score_str}",
            f"  Realtime Factor:   {rtf_str}",
            f"  {mos_label:<19} {self.scorecard.human_mos:.2f} / 5.0",
            f"  Technical Status:  {self.scorecard.technical_status}",
            f"  Recommendation:    {self.scorecard.recommendation}",
            "-" * 60,
            "TECHNICAL INTEGRITY",
            f"  Source FPS:        {self.technical_qc.source_fps:.2f} -> Output: {self.technical_qc.output_fps:.2f}",
            f"  Frame Counts:      Expected ~{self.technical_qc.expected_nb_frames:,}, Actual {self.technical_qc.output_nb_frames:,}",
            f"  Source Preserved:  {'PASS' if self.technical_qc.source_preservation_passed else 'FAIL'} (PSNR: {self.technical_qc.source_preservation_psnr:.2f} dB, SSIM: {self.technical_qc.source_preservation_ssim:.4f})",
            f"  Audio Sync:        {'PASS' if self.technical_qc.audio_sync_passed else 'FAIL'}",
            f"  PTS Monotonic:     {'PASS' if self.technical_qc.pts_check_passed else 'FAIL'}",
            f"  Scene Cut Guard:   {'PASS' if self.technical_qc.scene_cuts_properly_handled else 'FAIL'}",
            "-" * 60,
            "TEMPORAL & VFI QUALITY",
            f"  Motion Smoothness: {self.temporal_qc.motion_smoothness_score:.3f} ({self.temporal_qc.status})",
            f"  Mean Warping Err:  {self.temporal_qc.mean_warping_error:.2f}",
            f"  Motion Boundary:   {self.temporal_qc.motion_boundary_error:.2f}",
            f"  Temporal Flicker:  {self.temporal_qc.temporal_flicker_score:.3f}",
            "-" * 60,
            "FOOTBALL QUALITY (1-5)",
            f"  Player Integrity:  {self.football_qc.player_integrity_score:.1f} / 5.0",
            f"  Ball Integrity:    {self.football_qc.ball_integrity_score:.1f} / 5.0",
            f"  Occlusion:         {self.football_qc.occlusion_handling_score:.1f} / 5.0",
            f"  Pitch Geometry:    {self.football_qc.pitch_geometry_score:.1f} / 5.0",
            f"  Goal & Net:        {self.football_qc.goal_net_score:.1f} / 5.0",
            f"  Graphics & Logo:   {self.football_qc.broadcast_graphics_score:.1f} / 5.0",
            f"  Camera Motion:     {self.football_qc.camera_motion_score:.1f} / 5.0",
            "-" * 60,
            "ADDED ARTIFACT DIFFERENTIAL (Delta = Output - Source)",
            f"  Added Ghosting:    +{self.artifact_diff.ghosting.added_level * 100:.2f}%",
            f"  Added Double Edge: +{self.artifact_diff.double_contour.added_level * 100:.2f}%",
            f"  Added Tearing:     +{self.artifact_diff.edge_tearing.added_level * 100:.2f}%",
            f"  Added Deformation: +{self.artifact_diff.deformation.added_level * 100:.2f}%",
            "-" * 60,
            f"SUSPICIOUS MOMENTS:  {len(self.suspicious_moments)} flagged"
        ]
        if self.suspicious_moments:
            lines.append("  Top Suspicious Moments:")
            for sm in self.suspicious_moments[:5]:
                lines.append(f"    - [{sm.timestamp_str}] Severity {sm.severity}: {sm.description}")
        lines.append("=" * 60)
        return "\n".join(lines)
