import pytest
from eval.config import FootballScoreWeights
from eval.types import (
    FootballQCResult,
    PerceptualQCResult,
    PerformanceQCResult,
    TechnicalQCResult,
    TemporalQCResult,
)
from eval.layer4_perceptual.scorecard import compute_production_scorecard


def test_scorecard_renormalization_without_human_survey():
    """Verify that when human survey is absent, objective weights re-normalize to 1.0."""
    technical_qc = TechnicalQCResult(
        source_fps=25.0,
        output_fps=50.0,
        source_nb_frames=100,
        output_nb_frames=200,
        expected_nb_frames=200,
        frame_count_diff=0,
        fps_check_passed=True,
        frame_count_passed=True,
        source_preservation_psnr=40.0,
        source_preservation_ssim=0.98,
        source_preservation_mae=1.5,
        source_preservation_max_diff=10.0,
        source_preservation_passed=True,
        audio_present=True,
        audio_duration_diff_sec=0.0,
        audio_sync_passed=True,
        pts_monotonic=True,
        pts_gaps_detected=0,
        pts_duplicates_detected=0,
        pts_check_passed=True,
        scene_cuts_detected_source=0,
        scene_cuts_properly_handled=True,
        status="PASS",
    )
    temporal_qc = TemporalQCResult(
        motion_smoothness_score=1.0,
        motion_discontinuity_count=0,
        mean_warping_error=5.0,
        p95_warping_error=8.0,
        motion_boundary_error=4.0,
        temporal_flicker_score=0.0,
        odd_even_oscillation_index=1.0,
        status="EXCELLENT",
    )
    football_qc = FootballQCResult(
        ball_integrity_score=5.0,
        player_integrity_score=5.0,
        occlusion_handling_score=5.0,
        pitch_geometry_score=5.0,
        goal_net_score=5.0,
        broadcast_graphics_score=5.0,
        camera_motion_score=5.0,
    )
    # Perceptual result with NO survey responses (automated pipeline run)
    perceptual_qc = PerceptualQCResult(
        overall_quality_mos=0.0,
        motion_naturalness_mos=0.0,
        artifact_free_mos=0.0,
        survey_responses_count=0,
    )
    performance_qc = PerformanceQCResult(
        total_processing_time_sec=10.0,
        source_duration_sec=10.0,
        realtime_factor=1.0,
        target_rtf_achieved=True,
    )

    # When all metrics are perfect (5.0 / 5.0) and smoothness=1.0, flicker=0:
    # temporal_stability_1_to_5 = 1.0 + 4.0 * 1.0 = 5.0.
    # Quality score should be exactly 5.0 * 2.0 = 10.0!
    card = compute_production_scorecard(
        model_name="test_model",
        technical_qc=technical_qc,
        temporal_qc=temporal_qc,
        football_qc=football_qc,
        perceptual_qc=perceptual_qc,
        performance_qc=performance_qc,
    )

    assert card.quality_score == 10.0
    assert card.recommendation == "PRODUCTION CANDIDATE"
    assert card.quality_status == "EXCELLENT"
