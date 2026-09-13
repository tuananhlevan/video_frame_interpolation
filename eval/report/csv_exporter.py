"""Exports evaluation metrics to CSV."""

import csv
import os
from typing import Any, Dict, List, Optional
from eval.types import EvaluationReport


def export_report_csv(
    report: EvaluationReport,
    output_path: str,
    frame_metrics: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Exports key metrics and optional per-frame traces to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Metric", "Value", "Target_or_Unit"])
        
        # Summary metrics
        writer.writerow(["Scorecard", "Model", report.model_name, "name"])
        writer.writerow(["Scorecard", "Quality_Score", report.scorecard.quality_score, "0-10"])
        writer.writerow(["Scorecard", "Performance_Score", report.scorecard.performance_score, "0-10"])
        writer.writerow(["Scorecard", "Realtime_Factor", report.scorecard.realtime_factor, "x (target 1.0-1.3)"])
        writer.writerow(["Scorecard", "Human_MOS", report.scorecard.human_mos, "1-5"])
        writer.writerow(["Scorecard", "Recommendation", report.scorecard.recommendation, "status"])

        # Technical QC
        writer.writerow(["Technical", "Source_FPS", report.technical_qc.source_fps, "fps"])
        writer.writerow(["Technical", "Output_FPS", report.technical_qc.output_fps, "fps"])
        writer.writerow(["Technical", "Source_Frames", report.technical_qc.source_nb_frames, "frames"])
        writer.writerow(["Technical", "Output_Frames", report.technical_qc.output_nb_frames, "frames"])
        writer.writerow(["Technical", "Source_Preservation_PSNR", report.technical_qc.source_preservation_psnr, "dB"])
        writer.writerow(["Technical", "Source_Preservation_SSIM", report.technical_qc.source_preservation_ssim, "score"])
        writer.writerow(["Technical", "Audio_Sync_Passed", report.technical_qc.audio_sync_passed, "bool"])
        writer.writerow(["Technical", "PTS_Monotonic", report.technical_qc.pts_monotonic, "bool"])
        writer.writerow(["Technical", "Scene_Cuts_Handled", report.technical_qc.scene_cuts_properly_handled, "bool"])

        # Temporal QC
        writer.writerow(["Temporal", "Motion_Smoothness", report.temporal_qc.motion_smoothness_score, "0-1"])
        writer.writerow(["Temporal", "Mean_Warping_Error", report.temporal_qc.mean_warping_error, "px"])
        writer.writerow(["Temporal", "Motion_Boundary_Error", report.temporal_qc.motion_boundary_error, "px"])
        writer.writerow(["Temporal", "Flicker_Score", report.temporal_qc.temporal_flicker_score, "score"])

        # Football QC
        writer.writerow(["Football", "Player_Integrity", report.football_qc.player_integrity_score, "1-5"])
        writer.writerow(["Football", "Ball_Integrity", report.football_qc.ball_integrity_score, "1-5"])
        writer.writerow(["Football", "Occlusion_Handling", report.football_qc.occlusion_handling_score, "1-5"])
        writer.writerow(["Football", "Pitch_Geometry", report.football_qc.pitch_geometry_score, "1-5"])
        writer.writerow(["Football", "Goal_Net", report.football_qc.goal_net_score, "1-5"])
        writer.writerow(["Football", "Broadcast_Graphics", report.football_qc.broadcast_graphics_score, "1-5"])
        writer.writerow(["Football", "Camera_Motion", report.football_qc.camera_motion_score, "1-5"])

        # Artifacts
        writer.writerow(["Artifacts", "Added_Ghosting", report.artifact_diff.ghosting.added_level, "delta"])
        writer.writerow(["Artifacts", "Added_Double_Contour", report.artifact_diff.double_contour.added_level, "delta"])
        writer.writerow(["Artifacts", "Added_Edge_Tearing", report.artifact_diff.edge_tearing.added_level, "delta"])
        writer.writerow(["Artifacts", "Added_Deformation", report.artifact_diff.deformation.added_level, "delta"])

        # Ground truth if available
        if report.ground_truth_qc:
            writer.writerow(["GroundTruth", "GT_PSNR", report.ground_truth_qc.psnr, "dB"])
            writer.writerow(["GroundTruth", "GT_SSIM", report.ground_truth_qc.ssim, "score"])
            if report.ground_truth_qc.lpips is not None:
                writer.writerow(["GroundTruth", "GT_LPIPS", report.ground_truth_qc.lpips, "score"])

    return output_path
