from collections import deque
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from upframe.pipeline.probe import probe_video
from upframe.utils.ffmpeg import format_duration

from eval.config import EvaluationConfig
from eval.ground_truth.benchmark import evaluate_against_ground_truth
from eval.layer1_technical import (
    validate_audio_integrity,
    validate_fps_and_frames,
    validate_scene_cut_handling,
    validate_source_preservation,
    validate_timestamps
)
from eval.layer2_temporal import (
    evaluate_motion_smoothness,
    evaluate_optical_flow_consistency,
    evaluate_temporal_flicker
)
from eval.layer2_temporal.smoothness import compute_frame_motion_vector
from eval.layer3_football import (
    evaluate_artifact_differential,
    evaluate_ball_integrity,
    evaluate_broadcast_graphics,
    evaluate_camera_motion_continuity,
    evaluate_goal_net_integrity,
    evaluate_pitch_geometry,
    evaluate_player_and_occlusion_integrity
)
from eval.layer3_football.artifacts import build_artifact_differential_result, measure_frame_artifacts
from eval.layer3_football.ball import detect_ball_candidates
from eval.layer3_football.pitch_geometry import extract_pitch_lines
from eval.layer3_football.player_occlusion import detect_player_blobs

from eval.layer4_perceptual import compute_production_scorecard, evaluate_perceptual_quality
from eval.performance.profiler import evaluate_performance
from eval.report.generator import generate_evaluation_reports
from eval.suspicious.detector import detect_suspicious_moments
from eval.types import (
    ArtifactDifferentialResult,
    EvaluationReport,
    FootballQCResult,
    GroundTruthQCResult,
    PerformanceQCResult,
    TechnicalQCResult,
    TemporalQCResult
)
from eval.visual import (
    extract_injected_frames,
    generate_difference_heatmap,
    generate_side_by_side_video,
    generate_slowmo_clip
)

logger = logging.getLogger("eval.pipeline")


class EvaluationPipeline:
    """Master evaluator orchestrating the 4-layer VFI evaluation strategy."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()

    def evaluate(
        self,
        source_path: str,
        output_path: str,
        eval_dir: Optional[str] = None,
        model_name: str = "vfi",
        processing_time_sec: Optional[float] = None
    ) -> EvaluationReport:
        """Executes full evaluation between source and output video."""
        if eval_dir is None or eval_dir in ("evaluation", "evaluation_log"):
            output_stem = os.path.splitext(os.path.basename(output_path))[0]
            eval_dir = os.path.join(self.config.eval_dir, output_stem)
        elif not os.path.isabs(eval_dir) and not eval_dir.startswith(self.config.eval_dir) and not eval_dir.startswith("./"):
            eval_dir = os.path.join(self.config.eval_dir, eval_dir)

        os.makedirs(eval_dir, exist_ok=True)
        t_start = time.perf_counter()

        # Attach file logging to evaluation.log inside the target evaluation folder
        log_file = os.path.join(eval_dir, "evaluation.log")
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

        try:
            return self._do_evaluate(source_path, output_path, eval_dir, model_name, processing_time_sec, t_start)
        finally:
            root_logger.removeHandler(file_handler)
            file_handler.close()

    def _do_evaluate(
        self,
        source_path: str,
        output_path: str,
        eval_dir: str,
        model_name: str,
        processing_time_sec: Optional[float],
        t_start: float
    ) -> EvaluationReport:
        logger.info(f"Starting UPFRAME evaluation: {source_path} vs {output_path}")

        # 1. Probing videos
        source_meta = probe_video(source_path)
        output_meta = probe_video(output_path)

        resolution_str = f"{output_meta.width}x{output_meta.height}"
        src_dur_str = format_duration(source_meta.duration)
        out_dur_str = format_duration(output_meta.duration)

        # -------------------------------------------------------------
        # Layer 1: Technical & Pipeline Integrity
        # -------------------------------------------------------------
        logger.info("Executing Layer 1: Technical Integrity checks...")
        fps_ok, frame_ok, fps_details, fps_warns = validate_fps_and_frames(
            source_meta, output_meta, tolerance_frames=self.config.technical_thresholds.max_frame_count_diff
        )

        pres_ok, pres_metrics, pres_warns = validate_source_preservation(
            source_path=source_path,
            output_path=output_path,
            max_frames=self.config.max_frames,
            stride=self.config.sample_stride,
            min_psnr=self.config.technical_thresholds.min_psnr_source_preservation,
            min_ssim=self.config.technical_thresholds.min_ssim_source_preservation,
            max_mae=self.config.technical_thresholds.max_mae_source_preservation
        )

        audio_ok, audio_details, audio_warns = validate_audio_integrity(
            source_meta, output_meta, max_duration_diff_sec=self.config.technical_thresholds.max_audio_duration_diff_sec
        )

        pts_ok, pts_details, pts_warns = validate_timestamps(output_path)

        cut_ok, cut_details, cut_warns = validate_scene_cut_handling(
            source_path, output_path, cut_threshold=self.config.technical_thresholds.scene_cut_threshold,
            max_frames_to_scan=self.config.max_frames
        )

        all_warns = fps_warns + pres_warns + audio_warns + pts_warns + cut_warns
        tech_status = "PASS"
        if not pres_ok or not fps_ok or not audio_ok or not pts_ok or not cut_ok:
            tech_status = "FAIL"
        elif all_warns:
            tech_status = "WARN"

        technical_qc = TechnicalQCResult(
            source_fps=source_meta.nominal_fps,
            output_fps=output_meta.nominal_fps,
            source_nb_frames=source_meta.nb_frames,
            output_nb_frames=output_meta.nb_frames,
            expected_nb_frames=source_meta.target_nb_frames,
            frame_count_diff=output_meta.nb_frames - source_meta.target_nb_frames,
            fps_check_passed=fps_ok,
            frame_count_passed=frame_ok,
            source_preservation_psnr=pres_metrics.get("psnr", 0.0),
            source_preservation_ssim=pres_metrics.get("ssim", 0.0),
            source_preservation_mae=pres_metrics.get("mae", 0.0),
            source_preservation_max_diff=pres_metrics.get("max_diff", 0.0),
            source_preservation_passed=pres_ok,
            audio_present=output_meta.has_audio,
            audio_duration_diff_sec=audio_details.get("audio_duration_diff_sec", 0.0),
            audio_sync_passed=audio_ok,
            pts_monotonic=pts_details.get("monotonic", True),
            pts_gaps_detected=pts_details.get("gaps_count", 0),
            pts_duplicates_detected=pts_details.get("duplicates_count", 0),
            pts_check_passed=pts_ok,
            scene_cuts_detected_source=cut_details.get("source_cuts_detected", 0),
            scene_cuts_properly_handled=cut_ok,
            status=tech_status,
            warnings=all_warns
        )

        # -------------------------------------------------------------
        # Source Baseline Compression Artifacts (Single-Pass Stream, 1-frame memory)
        # -------------------------------------------------------------
        logger.info("Streaming source frames for baseline artifact differential...")
        src_artifacts = self._evaluate_source_artifacts(
            source_path=source_path,
            width=source_meta.width,
            height=source_meta.height,
            max_frames=self.config.max_frames,
            stride=self.config.sample_stride
        )

        # -------------------------------------------------------------
        # Single-Pass Output Video Stream (Layers 2 & 3, 3-frame window)
        # -------------------------------------------------------------
        logger.info("Executing Layer 2 & 3 streaming analysis (sliding window)...")
        temporal_qc, artifact_diff, football_qc, frame_metrics = self._stream_evaluate_output(
            output_path=output_path,
            width=output_meta.width,
            height=output_meta.height,
            total_frames_hint=output_meta.nb_frames,
            src_artifact_tuple=src_artifacts,
            eval_dir=eval_dir,
            max_frames=self.config.max_frames,
            stride=self.config.sample_stride
        )

        # -------------------------------------------------------------
        # Suspicious Moments Detection
        # -------------------------------------------------------------
        logger.info("Detecting suspicious moments and anomalies...")
        suspicious_moments = detect_suspicious_moments(frame_metrics, fps=output_meta.nominal_fps)

        # -------------------------------------------------------------
        # Layer 4: Perceptual MOS & Scorecard
        # -------------------------------------------------------------
        logger.info("Computing Layer 4: Perceptual MOS and Scorecard...")
        auto_quality_estimate = float(np.mean([
            football_qc.player_integrity_score,
            football_qc.ball_integrity_score,
            football_qc.occlusion_handling_score,
            football_qc.pitch_geometry_score,
            football_qc.broadcast_graphics_score,
            football_qc.camera_motion_score
        ]))
        perceptual_qc = evaluate_perceptual_quality(
            auto_quality_estimate, human_mos_file=self.config.human_mos_file
        )

        # Performance evaluation
        actual_proc_time = processing_time_sec or (time.perf_counter() - t_start)
        performance_qc = evaluate_performance(
            processing_time_sec=actual_proc_time,
            source_duration_sec=source_meta.duration,
            output_frames_count=output_meta.nb_frames,
            source_frames_count=source_meta.nb_frames,
            thresholds=self.config.perf_thresholds
        )

        scorecard = compute_production_scorecard(
            model_name=model_name,
            technical_qc=technical_qc,
            temporal_qc=temporal_qc,
            football_qc=football_qc,
            perceptual_qc=perceptual_qc,
            performance_qc=performance_qc,
            weights=self.config.weights,
            perf_thresholds=self.config.perf_thresholds
        )

        # Optional Ground Truth Evaluation
        ground_truth_qc: Optional[GroundTruthQCResult] = None
        if self.config.ground_truth_path and os.path.exists(self.config.ground_truth_path):
            logger.info("Running Ground Truth benchmark comparison...")
            ground_truth_qc = evaluate_against_ground_truth(
                self.config.ground_truth_path, output_path, max_frames=self.config.max_frames
            )

        # -------------------------------------------------------------
        # Visual Materials Generation (Slowmo & Side-by-Side)
        # Note: Injected frames and diff heatmaps are already extracted
        # during the streaming pass without redundant decoding.
        # -------------------------------------------------------------
        if self.config.generate_visuals:
            if self.config.generate_slowmo:
                logger.info("Generating slow-motion review clips...")
                slowmo_dir = os.path.join(eval_dir, "slowmo")
                for rate in self.config.slowmo_rates:
                    slowmo_out = os.path.join(slowmo_dir, f"slowmo_{rate}x.mp4")
                    generate_slowmo_clip(output_path, slowmo_out, factor=rate, duration_sec=self.config.slowmo_duration_sec)

            if self.config.generate_side_by_side:
                logger.info("Generating side-by-side comparison clip...")
                sbs_out = os.path.join(eval_dir, "side_by_side.mp4")
                generate_side_by_side_video(source_path, output_path, sbs_out)


        # -------------------------------------------------------------
        # Build Report & Generate Exports
        # -------------------------------------------------------------
        report = EvaluationReport(
            source_file=os.path.abspath(source_path),
            output_file=os.path.abspath(output_path),
            model_name=model_name,
            resolution=resolution_str,
            source_duration_str=src_dur_str,
            output_duration_str=out_dur_str,
            technical_qc=technical_qc,
            temporal_qc=temporal_qc,
            artifact_diff=artifact_diff,
            football_qc=football_qc,
            perceptual_qc=perceptual_qc,
            performance_qc=performance_qc,
            scorecard=scorecard,
            suspicious_moments=suspicious_moments,
            ground_truth_qc=ground_truth_qc,
            eval_dir=eval_dir
        )

        logger.info("Generating report files (JSON, CSV, HTML, TXT)...")
        generate_evaluation_reports(report, eval_dir, frame_metrics=frame_metrics)

        logger.info(f"UPFRAME evaluation completed successfully! Reports saved in: {eval_dir}")
        return report

    def _evaluate_source_artifacts(
        self,
        source_path: str,
        width: int,
        height: int,
        max_frames: Optional[int] = None,
        stride: int = 1
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Single-pass streaming evaluation of baseline compression artifacts in source video."""
        from upframe.pipeline.decode import VideoDecoder
        ghosting: List[float] = []
        double: List[float] = []
        tearing: List[float] = []
        deform: List[float] = []

        try:
            decoder = VideoDecoder(source_path, width, height)
            count = 0
            eval_count = 0
            for frame in decoder.stream_frames():
                if count % stride == 0:
                    m = measure_frame_artifacts(frame)
                    ghosting.append(m["ghosting"])
                    double.append(m["double_contour"])
                    tearing.append(m["edge_tearing"])
                    deform.append(m["deformation"])
                    eval_count += 1
                    if max_frames and eval_count >= max_frames:
                        break
                count += 1
        except Exception as e:
            logger.warning(f"Error streaming source artifacts: {e}")

        return ghosting, double, tearing, deform

    def _stream_evaluate_output(
        self,
        output_path: str,
        width: int,
        height: int,
        total_frames_hint: int,
        src_artifact_tuple: Tuple[List[float], List[float], List[float], List[float]],
        eval_dir: str,
        max_frames: Optional[int] = None,
        stride: int = 1
    ) -> Tuple[TemporalQCResult, ArtifactDifferentialResult, FootballQCResult, List[Dict[str, Any]]]:
        """Single-pass streaming evaluation over output frames using a 3-frame sliding window.
        
        Evaluates Layer 2 (temporal stability, optical flow consistency, flicker) and
        Layer 3 (football geometry, ball/player integrity, static graphics, camera continuity)
        with bounded O(1) memory footprint (< 150 MB).
        """
        from upframe.pipeline.decode import VideoDecoder

        max_eval_output = (max_frames * 2) if max_frames else None
        target_eval_total = min(total_frames_hint, max_eval_output) if max_eval_output else total_frames_hint

        # Visual directories
        injected_dir = os.path.join(eval_dir, "injected_frames")
        diff_dir = os.path.join(eval_dir, "diff_maps")
        if self.config.generate_visuals:
            if self.config.extract_injected_frames:
                os.makedirs(injected_dir, exist_ok=True)
            if self.config.generate_diff_maps:
                os.makedirs(diff_dir, exist_ok=True)

        saved_injected = 0
        saved_diff = 0

        # Layer 2 accumulators
        motion_vectors: List[Tuple[float, float, float]] = []
        shifts: List[Tuple[float, float]] = []
        consecutive_diffs: List[float] = []
        lag2_diffs: List[float] = []
        var_flickers: List[float] = []
        warping_errors: List[float] = []
        motion_boundary_errors: List[float] = []
        motion_magnitudes: List[float] = []
        frame_metrics: List[Dict[str, Any]] = []

        # Layer 3 output artifact metrics (on intermediate frames)
        out_ghosting: List[float] = []
        out_double: List[float] = []
        out_tearing: List[float] = []
        out_deform: List[float] = []

        # Ball tracking
        ball_trajectories: List[Tuple[float, float]] = []
        prev_ball_pos: Optional[Tuple[float, float]] = None
        teleportation_count = 0
        duplicate_ball_events = 0

        # Player & occlusion tracking
        total_players_checked = 0
        solidity_anomalies = 0
        occlusion_events = 0
        occlusion_failures = 0

        # Pitch geometry tracking
        total_lines_detected = 0
        wobble_events = 0
        line_counts_per_frame: List[int] = []

        # Goal net tracking
        laplacian_variances: List[float] = []

        # Broadcast graphics tracking
        roi_top_left = (slice(0, int(height * 0.18)), slice(0, int(width * 0.30)))
        roi_top_right = (slice(0, int(height * 0.15)), slice(int(width * 0.75), int(width * 0.98)))
        diffs_tl: List[float] = []
        diffs_tr: List[float] = []

        # 3-frame sliding window
        window: deque = deque(maxlen=3)
        gray_window: deque = deque(maxlen=3)
        indices_window: deque = deque(maxlen=3)

        raw_idx = 0
        eval_idx = 0

        decoder = VideoDecoder(output_path, width, height)
        for frame in decoder.stream_frames():
            if raw_idx % stride != 0:
                raw_idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

            # -------------------------------------------------------------
            # Pairwise checks with immediate previous frame
            # -------------------------------------------------------------
            if len(window) >= 1:
                prev_frame = window[-1]
                prev_gray = gray_window[-1]

                # Motion vector (Smoothness) & Camera pan
                dx, dy, mag = compute_frame_motion_vector(prev_frame, frame)
                motion_vectors.append((dx, dy, mag))
                shifts.append((dx, dy))

                # Static graphics jitter
                for (ys, xs), diff_list in [(roi_top_left, diffs_tl), (roi_top_right, diffs_tr)]:
                    patch1 = prev_gray[ys, xs]
                    patch2 = gray[ys, xs]
                    edges1 = cv2.Canny(patch1, 80, 180)
                    if np.sum(edges1) > 100:
                        diff_val = float(np.mean(np.abs(patch1[edges1 > 0].astype(np.float32) - patch2[edges1 > 0].astype(np.float32))))
                        diff_list.append(diff_val)

                # Temporal flicker consecutive diff & laplacian variance diff
                g_curr_f = gray.astype(np.float32)
                g_prev_f = prev_gray.astype(np.float32)
                consecutive_diffs.append(float(np.mean(np.abs(g_curr_f - g_prev_f))))
                lap_curr = cv2.Laplacian(g_curr_f, cv2.CV_32F).var()
                lap_prev = cv2.Laplacian(g_prev_f, cv2.CV_32F).var()
                var_flickers.append(abs(lap_curr - lap_prev) / (lap_curr + lap_prev + 1e-5))

            # -------------------------------------------------------------
            # Lag-2 checks with frame at t-2
            # -------------------------------------------------------------
            if len(window) >= 2:
                prev2_gray = gray_window[-2]
                g_curr_f = gray.astype(np.float32)
                g_prev2_f = prev2_gray.astype(np.float32)
                lag2_diffs.append(float(np.mean(np.abs(g_curr_f - g_prev2_f))))

            # -------------------------------------------------------------
            # Triplet check (A, X, B) for interpolated intermediate frame
            # -------------------------------------------------------------
            if len(window) >= 2 and (indices_window[-1] % 2 == 1):
                fa = window[-2]
                fx = window[-1]
                fb = frame
                odd_idx = indices_window[-1]

                # Artifact measurements on intermediate frame fx
                m = measure_frame_artifacts(fx)
                out_ghosting.append(m["ghosting"])
                out_double.append(m["double_contour"])
                out_tearing.append(m["edge_tearing"])
                out_deform.append(m["deformation"])

                # Optical flow consistency
                flow_res = evaluate_optical_flow_consistency(fa, fx, fb)
                w_err = flow_res["mean_warping_error"]
                mb_err = flow_res["motion_boundary_error"]
                mag_val = flow_res["motion_magnitude"]
                warping_errors.append(w_err)
                motion_boundary_errors.append(mb_err)
                motion_magnitudes.append(mag_val)

                frame_metrics.append({
                    "frame_index": odd_idx,
                    "warping_error": w_err,
                    "motion_boundary_error": mb_err,
                    "motion_magnitude": mag_val
                })

                # Visual assets extraction directly during stream
                if self.config.generate_visuals:
                    if self.config.extract_injected_frames and saved_injected < self.config.max_injected_frames_to_save:
                        fname = f"injected_frame_{odd_idx:06d}.jpg"
                        cv2.imwrite(os.path.join(injected_dir, fname), fx, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        saved_injected += 1
                    if self.config.generate_diff_maps and saved_diff < self.config.max_diff_maps_to_save:
                        dname = f"diff_heatmap_{odd_idx:06d}.jpg"
                        generate_difference_heatmap(fa, fx, os.path.join(diff_dir, dname))
                        saved_diff += 1

            # -------------------------------------------------------------
            # Per-frame Football checks
            # -------------------------------------------------------------
            # Ball
            candidates = detect_ball_candidates(frame)
            if len(candidates) >= 2:
                for i in range(len(candidates)):
                    for j in range(i + 1, len(candidates)):
                        dist = math.hypot(candidates[i][0] - candidates[j][0], candidates[i][1] - candidates[j][1])
                        if 10.0 <= dist <= 60.0:
                            duplicate_ball_events += 1
            best_candidate = None
            if candidates:
                if prev_ball_pos is None:
                    candidates.sort(key=lambda x: x[3], reverse=True)
                    best_candidate = (candidates[0][0], candidates[0][1])
                else:
                    candidates.sort(key=lambda x: math.hypot(x[0] - prev_ball_pos[0], x[1] - prev_ball_pos[1]))
                    closest = candidates[0]
                    dist = math.hypot(closest[0] - prev_ball_pos[0], closest[1] - prev_ball_pos[1])
                    if dist > 75.0:
                        teleportation_count += 1
                    best_candidate = (closest[0], closest[1])
            if best_candidate:
                ball_trajectories.append(best_candidate)
                prev_ball_pos = best_candidate

            # Players & Occlusion
            players = detect_player_blobs(frame)
            total_players_checked += len(players)
            for p in players:
                if p["solidity"] < 0.45:
                    solidity_anomalies += 1
            for i in range(len(players)):
                x1, y1, w1, h1 = players[i]["bbox"]
                for j in range(i + 1, len(players)):
                    x2, y2, w2, h2 = players[j]["bbox"]
                    ix = max(x1, x2)
                    iy = max(y1, y2)
                    iw = min(x1 + w1, x2 + w2) - ix
                    ih = min(y1 + h1, y2 + h2) - iy
                    if iw > 5 and ih > 10:
                        occlusion_events += 1
                        roi = frame[iy:iy + ih, ix:ix + iw]
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        if cv2.Laplacian(gray_roi, cv2.CV_32F).var() < 50.0:
                            occlusion_failures += 1

            # Pitch Geometry
            lines = extract_pitch_lines(frame)
            total_lines_detected += len(lines)
            line_counts_per_frame.append(len(lines))
            if len(lines) >= 2:
                angles = [np.arctan2(y2 - y1, x2 - x1) for x1, y1, x2, y2 in lines]
                if float(np.std(angles)) > 1.2:
                    wobble_events += 1

            # Goal Net
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            laplacian_variances.append(float(cv2.Laplacian(thresh, cv2.CV_32F).var()))

            # Slide window
            window.append(frame)
            gray_window.append(gray)
            indices_window.append(raw_idx)

            eval_idx += 1
            raw_idx += 1

            if eval_idx % 100 == 0 or (target_eval_total > 0 and eval_idx == target_eval_total):
                pct = (eval_idx / target_eval_total * 100) if target_eval_total > 0 else 0
                logger.info(f"Stream-analyzed {eval_idx}/{target_eval_total} frames ({pct:.1f}%)...")

            if max_eval_output and eval_idx >= max_eval_output:
                break

        # -------------------------------------------------------------
        # Aggregate Layer 2: Temporal QC Result
        # -------------------------------------------------------------
        accelerations: List[float] = []
        discontinuity_indices: List[int] = []
        for i in range(len(motion_vectors) - 1):
            dx1, dy1, mag1 = motion_vectors[i]
            dx2, dy2, mag2 = motion_vectors[i + 1]
            acc = float(np.sqrt((dx2 - dx1) ** 2 + (dy2 - dy1) ** 2))
            accelerations.append(acc)
            base_mag = max(1.0, (mag1 + mag2) / 2.0)
            if acc > 2.5 * base_mag and acc > 3.0:
                discontinuity_indices.append(i + 1)

        mean_acc = float(np.mean(accelerations)) if accelerations else 0.0
        discontinuity_penalty = min(0.6, len(discontinuity_indices) * 0.05)
        jerk_penalty = min(0.4, mean_acc / 10.0)
        smoothness_score = max(0.0, 1.0 - discontinuity_penalty - jerk_penalty)

        mean_d1 = float(np.mean(consecutive_diffs)) if consecutive_diffs else 0.0
        mean_d2 = float(np.mean(lag2_diffs)) if lag2_diffs else mean_d1
        oscillation_ratios: List[float] = []
        for i in range(len(lag2_diffs)):
            if i + 1 < len(consecutive_diffs):
                d1_cur = consecutive_diffs[i + 1]
                d2_cur = lag2_diffs[i]
                oscillation_ratios.append(d1_cur / (d2_cur + 1.0))
        mean_oscillation = float(np.mean(oscillation_ratios)) if oscillation_ratios else 1.0
        mean_var_flicker = float(np.mean(var_flickers)) if var_flickers else 0.0
        flicker_score = float(mean_d1 * 0.1 + mean_var_flicker * 5.0)

        mean_warp = float(np.mean(warping_errors)) if warping_errors else 0.0
        p95_warp = float(np.percentile(warping_errors, 95)) if warping_errors else 0.0
        mean_mb = float(np.mean(motion_boundary_errors)) if motion_boundary_errors else 0.0

        temp_status = "GOOD"
        if smoothness_score >= 0.85 and mean_warp < 15.0:
            temp_status = "EXCELLENT"
        elif smoothness_score < 0.65 or mean_warp > 28.0:
            temp_status = "POOR"
        elif smoothness_score < 0.75:
            temp_status = "FAIR"

        temporal_qc = TemporalQCResult(
            motion_smoothness_score=round(smoothness_score, 3),
            motion_discontinuity_count=len(discontinuity_indices),
            mean_warping_error=round(mean_warp, 2),
            p95_warping_error=round(p95_warp, 2),
            motion_boundary_error=round(mean_mb, 2),
            temporal_flicker_score=round(flicker_score, 3),
            odd_even_oscillation_index=round(mean_oscillation, 3),
            status=temp_status
        )

        # -------------------------------------------------------------
        # Aggregate Layer 3: Artifact Differential
        # -------------------------------------------------------------
        src_g, src_dc, src_et, src_df = src_artifact_tuple
        artifact_diff = build_artifact_differential_result(
            src_ghosting=src_g,
            src_double=src_dc,
            src_tearing=src_et,
            src_deform=src_df,
            out_ghosting=out_ghosting,
            out_double=out_double,
            out_tearing=out_tearing,
            out_deform=out_deform,
            motion_magnitudes=motion_magnitudes
        )

        # -------------------------------------------------------------
        # Aggregate Layer 3: Football QC Result
        # -------------------------------------------------------------
        # Ball
        ball_score = 5.0
        if len(ball_trajectories) > 1:
            teleport_rate = teleportation_count / float(len(ball_trajectories))
            ball_score -= min(3.0, teleport_rate * 10.0)
        if duplicate_ball_events > 0:
            ball_score -= min(2.0, duplicate_ball_events * 0.25)
        ball_score = max(1.0, min(5.0, ball_score))

        # Player
        player_score = 5.0
        if total_players_checked > 0:
            anomaly_rate = solidity_anomalies / float(total_players_checked)
            player_score -= min(3.5, anomaly_rate * 25.0)
        player_score = max(1.0, min(5.0, player_score))

        # Occlusion
        occlusion_score = 5.0
        if occlusion_events > 0:
            fail_rate = occlusion_failures / float(occlusion_events)
            occlusion_score -= min(3.5, fail_rate * 5.0)
        occlusion_score = max(1.0, min(5.0, occlusion_score))

        # Pitch
        pitch_score = 5.0
        if eval_idx > 0:
            wobble_rate = wobble_events / float(eval_idx)
            pitch_score -= min(2.5, wobble_rate * 3.0)
        count_variance = float(np.var(line_counts_per_frame)) if line_counts_per_frame else 0.0
        pitch_score -= min(1.5, count_variance / 50.0)
        pitch_score = max(1.0, min(5.0, pitch_score))

        # Goal net
        goal_score = 4.8
        if len(laplacian_variances) >= 4:
            even_mean = float(np.mean(laplacian_variances[0::2]))
            odd_mean = float(np.mean(laplacian_variances[1::2]))
            diff_ratio = abs(even_mean - odd_mean) / (even_mean + 1e-5)
            if diff_ratio > 0.20:
                goal_score -= min(2.5, diff_ratio * 4.0)
        goal_score = max(1.0, min(5.0, goal_score))

        # Graphics
        roi_scores = []
        if diffs_tl:
            roi_scores.append(float(np.mean(diffs_tl)))
        if diffs_tr:
            roi_scores.append(float(np.mean(diffs_tr)))
        avg_jitter = float(np.mean(roi_scores)) if roi_scores else 2.0
        graphics_score = 5.0
        if avg_jitter > 3.0:
            graphics_score -= min(3.0, (avg_jitter - 3.0) * 0.4)
        graphics_score = max(1.0, min(5.0, graphics_score))

        # Camera
        pan_magnitudes = [np.hypot(dx, dy) for dx, dy in shifts]
        mean_pan = float(np.mean(pan_magnitudes)) if pan_magnitudes else 0.0
        camera_jitter_events = 0
        for i in range(len(shifts) - 1):
            dx1, dy1 = shifts[i]
            dx2, dy2 = shifts[i + 1]
            delta = np.hypot(dx2 - dx1, dy2 - dy1)
            if delta > max(2.5, mean_pan * 1.5):
                camera_jitter_events += 1
        camera_score = 5.0
        if len(shifts) > 1:
            jitter_rate = camera_jitter_events / float(len(shifts))
            camera_score -= min(3.0, jitter_rate * 8.0)
        camera_score = max(1.0, min(5.0, camera_score))

        football_qc = FootballQCResult(
            ball_integrity_score=round(ball_score, 2),
            player_integrity_score=round(player_score, 2),
            occlusion_handling_score=round(occlusion_score, 2),
            pitch_geometry_score=round(pitch_score, 2),
            goal_net_score=round(goal_score, 2),
            broadcast_graphics_score=round(graphics_score, 2),
            camera_motion_score=round(camera_score, 2),
            detected_ball_teleportations=teleportation_count,
            detected_duplicate_balls=duplicate_ball_events,
            graphics_jitter_detected=avg_jitter > 5.0
        )

        return temporal_qc, artifact_diff, football_qc, frame_metrics

    def _load_frame_sequences(

        self,
        source_path: str,
        output_path: str,
        max_frames: Optional[int] = None,
        stride: int = 1
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Loads synchronized source and output frames for video-level metric analysis."""
        from upframe.pipeline.decode import VideoDecoder
        src_meta = probe_video(source_path)
        out_meta = probe_video(output_path)

        src_frames: List[np.ndarray] = []
        out_frames: List[np.ndarray] = []

        try:
            dec_s = VideoDecoder(source_path, src_meta.width, src_meta.height)
            s_count = 0
            for f in dec_s.stream_frames():
                if s_count % stride == 0:
                    src_frames.append(f)
                s_count += 1
                if max_frames and len(src_frames) >= max_frames:
                    break

            dec_o = VideoDecoder(output_path, out_meta.width, out_meta.height)
            o_count = 0
            for f in dec_o.stream_frames():
                if o_count % stride == 0:
                    out_frames.append(f)
                o_count += 1
                if max_frames and len(out_frames) >= (max_frames * 2):
                    break
        except Exception:
            # Fallback to OpenCV
            cap_src = cv2.VideoCapture(source_path)
            cap_out = cv2.VideoCapture(output_path)
            try:
                count = 0
                while True:
                    ret_s, fs = cap_src.read()
                    if not ret_s:
                        break
                    if count % stride == 0:
                        src_frames.append(fs)
                    count += 1
                    if max_frames and len(src_frames) >= max_frames:
                        break

                count = 0
                while True:
                    ret_o, fo = cap_out.read()
                    if not ret_o:
                        break
                    if count % stride == 0:
                        out_frames.append(fo)
                    count += 1
                    if max_frames and len(out_frames) >= (max_frames * 2):
                        break
            finally:
                cap_src.release()
                cap_out.release()

        return src_frames, out_frames
