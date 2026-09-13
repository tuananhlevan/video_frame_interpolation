"""Production Scorecard and Model Recommendation decision engine."""

from typing import Dict
from eval.config import FootballScoreWeights, PerformanceThresholds
from eval.types import (
    FootballQCResult,
    PerceptualQCResult,
    PerformanceQCResult,
    ScorecardResult,
    TechnicalQCResult,
    TemporalQCResult
)


def compute_production_scorecard(
    model_name: str,
    technical_qc: TechnicalQCResult,
    temporal_qc: TemporalQCResult,
    football_qc: FootballQCResult,
    perceptual_qc: PerceptualQCResult,
    performance_qc: PerformanceQCResult,
    weights: FootballScoreWeights = FootballScoreWeights(),
    perf_thresholds: PerformanceThresholds = PerformanceThresholds()
) -> ScorecardResult:
    """Calculates weighted 0-10 Quality Score, 0-10 Performance Score, and recommendation."""
    weights.validate()

    # Temporal stability normalized to 1-5 scale (from [0, 1] smoothness)
    temporal_stability_1_to_5 = 1.0 + 4.0 * temporal_qc.motion_smoothness_score
    # Penalize if high temporal flicker
    temporal_stability_1_to_5 = max(1.0, temporal_stability_1_to_5 - min(2.0, temporal_qc.temporal_flicker_score * 0.5))

    # Weighted sum on 1-5 scale
    weighted_1_to_5 = (
        weights.player_integrity * football_qc.player_integrity_score +
        weights.ball_integrity * football_qc.ball_integrity_score +
        weights.temporal_stability * temporal_stability_1_to_5 +
        weights.occlusion_handling * football_qc.occlusion_handling_score +
        weights.camera_motion * football_qc.camera_motion_score +
        weights.pitch_geometry * football_qc.pitch_geometry_score +
        weights.broadcast_graphics * football_qc.broadcast_graphics_score +
        weights.goal_net * football_qc.goal_net_score +
        weights.human_mos * perceptual_qc.overall_quality_mos
    )

    # Scale 1-5 to 0-10 Quality Score
    quality_score = round(float(weighted_1_to_5 * 2.0), 2)

    # Performance Score (0-10 scale)
    rtf = performance_qc.realtime_factor
    is_measured = performance_qc.is_measured or (performance_qc.total_processing_time_sec is not None)
    if rtf is None or not is_measured:
        perf_score = None
        perf_status = "NOT MEASURED"
    elif rtf <= 1.0:
        perf_score = 10.0
        perf_status = "PASS"
    elif rtf <= perf_thresholds.target_rtf_max:  # <= 1.3
        perf_score = 10.0 - ((rtf - 1.0) / 0.3) * 1.5
        perf_status = "PASS"
    elif rtf <= 1.6:
        perf_score = 8.5 - ((rtf - 1.3) / 0.3) * 2.5
        perf_status = "WARN"
    else:
        perf_score = max(1.0, 6.0 - (rtf - 1.6) * 2.0)
        perf_status = "FAIL"

    if perf_score is not None:
        perf_score = round(float(perf_score), 2)

    # Quality status
    if quality_score >= 8.5:
        quality_status = "EXCELLENT"
    elif quality_score >= 7.5:
        quality_status = "GOOD"
    elif quality_score >= 6.0:
        quality_status = "ACCEPTABLE"
    else:
        quality_status = "POOR"

    # Recommendation logic based on Decision Process (Section 16 & 51)
    if not technical_qc.passed:
        recommendation = "REJECTED (Technical integrity check failed)"
    elif quality_score >= 7.8 and perf_status in ("PASS", "NOT MEASURED"):
        recommendation = "PRODUCTION CANDIDATE" if perf_status == "PASS" else "PRODUCTION CANDIDATE (Quality: EXCELLENT; measure on 3x L40S to verify RTF)"
    elif quality_score >= 6.5 and perf_status in ("PASS", "WARN", "NOT MEASURED"):
        recommendation = "ACCEPTABLE WITH RESERVATIONS"
    elif perf_status == "FAIL":
        recommendation = "REJECTED (Performance below 1.0-1.3x realtime target)"
    else:
        recommendation = "REJECTED (Visual quality below broadcast thresholds)"

    mos_is_surveyed = (perceptual_qc.survey_responses_count > 0)

    return ScorecardResult(
        model_name=model_name,
        quality_score=quality_score,
        performance_score=perf_score,
        realtime_factor=rtf,
        human_mos=round(perceptual_qc.overall_quality_mos, 2),
        mos_is_surveyed=mos_is_surveyed,
        technical_status=technical_qc.status,
        quality_status=quality_status,
        performance_status=perf_status,
        recommendation=recommendation
    )
