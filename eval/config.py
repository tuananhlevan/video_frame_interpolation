"""Configuration and scoring weights for the upframe evaluation pipeline."""

from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional
import torch


@dataclass
class FootballScoreWeights:
    """Weights for composite football quality scoring based on spec Section 12 & 49."""
    player_integrity: float = 0.15
    ball_integrity: float = 0.15
    temporal_stability: float = 0.15
    occlusion_handling: float = 0.10
    camera_motion: float = 0.10
    pitch_geometry: float = 0.05
    broadcast_graphics: float = 0.10
    goal_net: float = 0.05
    human_mos: float = 0.15

    def validate(self) -> None:
        total = sum([
            self.player_integrity,
            self.ball_integrity,
            self.temporal_stability,
            self.occlusion_handling,
            self.camera_motion,
            self.pitch_geometry,
            self.broadcast_graphics,
            self.goal_net,
            self.human_mos
        ])
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Score weights must sum to 1.0, got {total:.4f}")


@dataclass
class TechnicalThresholds:
    """Pass/warn/fail thresholds for Layer 1 Technical Integrity."""
    min_psnr_source_preservation: float = 30.0  # dB (<30 fails, 30-35 warns, >35 passes)
    min_ssim_source_preservation: float = 0.90  # (<0.90 fails, 0.90-0.96 warns, >0.96 passes)
    max_mae_source_preservation: float = 8.0  # [0, 255]
    max_frame_count_diff: int = 5
    max_audio_duration_diff_sec: float = 0.25
    scene_cut_threshold: float = 0.35


@dataclass
class PerformanceThresholds:
    """Target thresholds for production performance."""
    target_rtf_min: float = 1.0
    target_rtf_max: float = 1.3
    min_fps_target: float = 45.0  # nominal 50 fps target


@dataclass
class EvaluationConfig:
    """Master evaluation configuration."""
    eval_dir: str = "evaluation"
    sample_stride: int = 1  # 1 = full evaluation, N = subsample every Nth interval
    max_frames: Optional[int] = None  # limit frames evaluated (for quick benchmarking)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Visualization toggles
    generate_visuals: bool = True
    extract_injected_frames: bool = True
    max_injected_frames_to_save: int = 50
    generate_slowmo: bool = True
    slowmo_rates: List[int] = field(default_factory=lambda: [2, 4])
    slowmo_duration_sec: float = 5.0
    generate_diff_maps: bool = True
    max_diff_maps_to_save: int = 20
    generate_side_by_side: bool = False
    
    # Reporting
    generate_html: bool = True
    generate_json: bool = True
    generate_csv: bool = True
    
    # Thresholds & Weights
    weights: FootballScoreWeights = field(default_factory=FootballScoreWeights)
    technical_thresholds: TechnicalThresholds = field(default_factory=TechnicalThresholds)
    perf_thresholds: PerformanceThresholds = field(default_factory=PerformanceThresholds)
    
    # Human MOS external data (if available)
    human_mos_file: Optional[str] = None
    default_mos_fallback: float = 4.0  # fallback MOS if no human survey provided
    
    # Ground truth comparison video (optional)
    ground_truth_path: Optional[str] = None
