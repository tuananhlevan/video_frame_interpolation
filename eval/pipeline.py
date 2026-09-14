from collections import deque
import logging
import math
import os
import shutil
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
from eval.layer1_technical.source_preservation import compute_psnr, compute_ssim
from upframe.pipeline.scene_detect import compute_frame_difference
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
from eval.performance.profiler import evaluate_performance, find_upframe_processing_time
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

        # Overwrite previous evaluation run if folder exists (fresh clean slate)
        if os.path.exists(eval_dir) and os.path.isdir(eval_dir):
            abs_eval = os.path.abspath(eval_dir)
            if abs_eval != "/" and abs_eval != os.path.abspath("."):
                shutil.rmtree(eval_dir, ignore_errors=True)
        os.makedirs(eval_dir, exist_ok=True)
        t_start = time.perf_counter()

        # Attach file logging to evaluation.log inside the target evaluation folder (mode="w" for fresh rerun)
        log_file = os.path.join(eval_dir, "evaluation.log")
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
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
        # -------------------------------------------------------------
        # Layer 1: Probe-based integrity checks
        # -------------------------------------------------------------
        logger.info("Executing Layer 1: Technical Integrity checks (probe metadata)...")
        fps_ok, frame_ok, fps_details, fps_warns = validate_fps_and_frames(
            source_meta, output_meta, tolerance_frames=self.config.technical_thresholds.max_frame_count_diff
        )
        audio_ok, audio_details, audio_warns = validate_audio_integrity(
            source_meta, output_meta, max_duration_diff_sec=self.config.technical_thresholds.max_audio_duration_diff_sec
        )
        pts_ok, pts_details, pts_warns = validate_timestamps(output_path)

        # -------------------------------------------------------------
        # Synchronized Dual-Stream Execution (Layers 1, 2, & 3 in single pass)
        # -------------------------------------------------------------
        if self.config.workers > 1 or self.config.gpus:
            from eval.parallel.engine import ParallelEvaluationEngine
            logger.info(f"Executing parallel multi-worker evaluation ({self.config.workers} workers, gpus={self.config.gpus})...")
            engine = ParallelEvaluationEngine(config=self.config)
            technical_qc, temporal_qc, artifact_diff, football_qc, frame_metrics = engine.evaluate_parallel(
                source_path=source_path,
                output_path=output_path,
                source_meta=source_meta,
                output_meta=output_meta,
                fps_ok=fps_ok,
                frame_ok=frame_ok,
                fps_warns=fps_warns,
                audio_ok=audio_ok,
                audio_details=audio_details,
                audio_warns=audio_warns,
                pts_ok=pts_ok,
                pts_details=pts_details,
                pts_warns=pts_warns,
                eval_dir=eval_dir,
                max_frames=self.config.max_frames
            )
        else:
            logger.info("Executing synchronized single-pass evaluation stream (Layers 1, 2, & 3)...")
            technical_qc, temporal_qc, artifact_diff, football_qc, frame_metrics = self._stream_evaluate_unified(
                source_path=source_path,
                output_path=output_path,
                source_meta=source_meta,
                output_meta=output_meta,
                fps_ok=fps_ok,
                frame_ok=frame_ok,
                fps_warns=fps_warns,
                audio_ok=audio_ok,
                audio_details=audio_details,
                audio_warns=audio_warns,
                pts_ok=pts_ok,
                pts_details=pts_details,
                pts_warns=pts_warns,
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

        # Performance evaluation:
        # Search for an upframe log matching the output video if not explicitly provided.
        # If no log is found, do NOT use evaluation runtime; leave field un-evaluated.
        actual_proc_time = processing_time_sec
        if actual_proc_time is None:
            actual_proc_time = find_upframe_processing_time(
                output_path=output_path,
                source_path=source_path,
                log_file=self.config.upframe_log_path
            )
            if actual_proc_time is not None:
                logger.info(f"Discovered upframe processing time from log: {actual_proc_time:.2f}s")
            else:
                logger.info("No upframe processing time log found for video; leaving performance un-evaluated.")

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

    def _stream_evaluate_unified(
        self,
        source_path: str,
        output_path: str,
        source_meta: Any,
        output_meta: Any,
        fps_ok: bool,
        frame_ok: bool,
        fps_warns: List[str],
        audio_ok: bool,
        audio_details: Dict[str, Any],
        audio_warns: List[str],
        pts_ok: bool,
        pts_details: Dict[str, Any],
        pts_warns: List[str],
        eval_dir: str,
        max_frames: Optional[int] = None,
        stride: int = 1
    ) -> Tuple[TechnicalQCResult, TemporalQCResult, ArtifactDifferentialResult, FootballQCResult, List[Dict[str, Any]]]:
        """Synchronized dual-stream evaluation combining Layer 1, Layer 2, and Layer 3 into a single pass.
        
        Decodes source and output video files exactly ONCE, maintaining 100% native 1080p pixel
        accuracy and bounded O(1) RAM footprint (< 150 MB).
        """
        from upframe.pipeline.decode import VideoDecoder

        width = output_meta.width
        height = output_meta.height
        src_w = source_meta.width
        src_h = source_meta.height

        max_eval_output = (max_frames * 2) if max_frames else None
        total_frames_hint = output_meta.nb_frames
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

        # Layer 1 Technical accumulators
        psnr_list: List[float] = []
        ssim_list: List[float] = []
        mae_list: List[float] = []
        max_diff_list: List[float] = []
        cuts_detected: List[int] = []
        hybrid_failures: List[Dict[str, Any]] = []
        prev_frame_src: Optional[np.ndarray] = None
        prev_out_odd: Optional[np.ndarray] = None
        cut_threshold = self.config.technical_thresholds.scene_cut_threshold

        # Source baseline artifact accumulators
        src_ghosting: List[float] = []
        src_double: List[float] = []
        src_tearing: List[float] = []
        src_deform: List[float] = []

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

        # Ball tracking (unbroken 50 FPS trajectory across all output frames)
        ball_trajectories: List[Tuple[float, float]] = []
        prev_ball_pos: Optional[Tuple[float, float]] = None
        prev_prev_ball_pos: Optional[Tuple[float, float]] = None
        teleportation_count = 0
        duplicate_ball_events = 0
        deformed_ball_events = 0
        ball_frames_since_last_seen = 0
        base_ball_velocity = 130.0 * (50.0 / max(1.0, output_meta.avg_fps or 50.0))

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

        raw_out_idx = 0
        src_idx = 0

        dec_src = VideoDecoder(source_path, src_w, src_h)
        dec_out = VideoDecoder(output_path, width, height)
        stream_src = dec_src.stream_frames()
        stream_out = dec_out.stream_frames()

        try:
            while True:
                if max_eval_output and raw_out_idx >= max_eval_output:
                    break

                try:
                    frame = next(stream_out)
                except StopIteration:
                    break

                is_even = (raw_out_idx % 2 == 0)

                # ---------------------------------------------------------
                # Layer 1 & Source Baseline: Synchronized with source stream
                # ---------------------------------------------------------
                if is_even:
                    try:
                        frame_src = next(stream_src)
                    except StopIteration:
                        frame_src = None

                    if frame_src is not None:
                        # Source baseline artifacts (sampled every N frames)
                        if src_idx % self.config.source_baseline_stride == 0:
                            m_src = measure_frame_artifacts(frame_src)
                            src_ghosting.append(m_src["ghosting"])
                            src_double.append(m_src["double_contour"])
                            src_tearing.append(m_src["edge_tearing"])
                            src_deform.append(m_src["deformation"])

                        # Scene cut detection between prev_frame_src and frame_src
                        if prev_frame_src is not None:
                            diff_cut = compute_frame_difference(prev_frame_src, frame_src)
                            if diff_cut >= cut_threshold:
                                cut_idx = src_idx - 1
                                cuts_detected.append(cut_idx)
                                if prev_out_odd is not None:
                                    f_inter = prev_out_odd
                                    if f_inter.shape != frame_src.shape:
                                        f_inter = cv2.resize(f_inter, (frame_src.shape[1], frame_src.shape[0]))
                                    diff_to_a = compute_frame_difference(f_inter, prev_frame_src)
                                    diff_to_b = compute_frame_difference(f_inter, frame_src)
                                    ideal_blend = (prev_frame_src.astype(np.float32) * 0.5 + frame_src.astype(np.float32) * 0.5).astype(np.uint8)
                                    diff_to_blend = compute_frame_difference(f_inter, ideal_blend)
                                    is_hybrid = (diff_to_a > 0.20 and diff_to_b > 0.20) or (diff_to_blend < 0.12 and diff_cut >= cut_threshold)
                                    if is_hybrid:
                                        hybrid_failures.append({
                                            "source_transition": (cut_idx, src_idx),
                                            "output_intermediate_frame": 2 * cut_idx + 1,
                                            "cut_difference": float(diff_cut),
                                            "diff_to_source_a": float(diff_to_a),
                                            "diff_to_source_b": float(diff_to_b),
                                            "diff_to_blend": float(diff_to_blend)
                                        })

                        prev_frame_src = frame_src

                        # Source frame preservation (PSNR, SSIM, MAE, max_diff)
                        if frame_src.shape != frame.shape:
                            f_out_comp = cv2.resize(frame, (frame_src.shape[1], frame_src.shape[0]))
                        else:
                            f_out_comp = frame

                        psnr_val = compute_psnr(frame_src, f_out_comp)
                        psnr_list.append(psnr_val)
                        ssim_list.append(compute_ssim(frame_src, f_out_comp))
                        mae_list.append(float(np.mean(np.abs(frame_src.astype(np.float32) - f_out_comp.astype(np.float32)))))
                        max_diff_list.append(float(np.max(np.abs(frame_src.astype(np.float32) - f_out_comp.astype(np.float32)))))
                        src_idx += 1
                else:
                    # Odd frame: track as candidate for hybrid scene cut check
                    prev_out_odd = frame

                # ---------------------------------------------------------
                # Output frame evaluation (Layers 2 & 3)
                # ---------------------------------------------------------
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

                # Pairwise checks with immediate previous frame
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

                # Lag-2 checks with frame at t-2
                if len(window) >= 2:
                    prev2_gray = gray_window[-2]
                    g_curr_f = gray.astype(np.float32)
                    g_prev2_f = prev2_gray.astype(np.float32)
                    lag2_diffs.append(float(np.mean(np.abs(g_curr_f - g_prev2_f))))

                # Triplet check (A, X, B) for interpolated intermediate frame
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

                    # Optical flow consistency (DIS Optical Flow)
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

                # Football checks on every frame
                # Players & Occlusion
                players = detect_player_blobs(frame)
                total_players_checked += len(players)

                # Ball Tracking (unbroken 50 FPS)
                candidates = detect_ball_candidates(frame, players=players, min_circularity=0.50)

                best_candidate = None
                if candidates:
                    if prev_ball_pos is None:
                        candidates.sort(key=lambda x: x[3], reverse=True)
                        if candidates[0][3] >= 0.55:
                            best_candidate = candidates[0]
                            ball_trajectories.append((best_candidate[0], best_candidate[1]))
                            prev_prev_ball_pos = prev_ball_pos
                            prev_ball_pos = (best_candidate[0], best_candidate[1])
                            ball_frames_since_last_seen = 0
                            if candidates[0][3] < 0.65:
                                deformed_ball_events += 1
                    else:
                        allowed_displacement = base_ball_velocity * (ball_frames_since_last_seen + 1)
                        candidates.sort(key=lambda x: math.hypot(x[0] - prev_ball_pos[0], x[1] - prev_ball_pos[1]))
                        closest = candidates[0]
                        dist = math.hypot(closest[0] - prev_ball_pos[0], closest[1] - prev_ball_pos[1])
                        if dist > allowed_displacement:
                            teleportation_count += 1
                            ball_frames_since_last_seen += 1
                            if ball_frames_since_last_seen > 6:
                                prev_ball_pos = None
                                prev_prev_ball_pos = None
                        else:
                            best_candidate = closest
                            ball_trajectories.append((best_candidate[0], best_candidate[1]))
                            prev_prev_ball_pos = prev_ball_pos
                            prev_ball_pos = (best_candidate[0], best_candidate[1])
                            ball_frames_since_last_seen = 0
                            if closest[3] < 0.65:
                                deformed_ball_events += 1
                else:
                    ball_frames_since_last_seen += 1
                    if ball_frames_since_last_seen > 6:
                        prev_ball_pos = None
                        prev_prev_ball_pos = None

                # Ghost ball (duplicate ball) detection
                current_vel = 0.0
                if prev_ball_pos is not None and prev_prev_ball_pos is not None:
                    current_vel = math.hypot(prev_ball_pos[0] - prev_prev_ball_pos[0], prev_ball_pos[1] - prev_prev_ball_pos[1])
                max_dup_dist = max(60.0, min(150.0, current_vel * 1.5))
                min_dup_dist = 8.0

                has_duplicate = False
                if best_candidate is not None and len(candidates) >= 2:
                    for c in candidates:
                        if c is not best_candidate:
                            dist = math.hypot(best_candidate[0] - c[0], best_candidate[1] - c[1])
                            if min_dup_dist <= dist <= max_dup_dist:
                                has_duplicate = True
                                break
                elif best_candidate is None and len(candidates) >= 2:
                    for i in range(len(candidates)):
                        for j in range(i + 1, len(candidates)):
                            dist = math.hypot(candidates[i][0] - candidates[j][0], candidates[i][1] - candidates[j][1])
                            if min_dup_dist <= dist <= max_dup_dist:
                                has_duplicate = True
                                break
                        if has_duplicate:
                            break
                if has_duplicate:
                    duplicate_ball_events += 1
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
                indices_window.append(raw_out_idx)

                raw_out_idx += 1

                if raw_out_idx % 100 == 0 or (target_eval_total > 0 and raw_out_idx == target_eval_total):
                    pct = (raw_out_idx / target_eval_total * 100) if target_eval_total > 0 else 0
                    logger.info(f"Stream-analyzed {raw_out_idx}/{target_eval_total} frames ({pct:.1f}%)...")
        finally:
            try:
                stream_src.close()
            except Exception:
                pass
            try:
                stream_out.close()
            except Exception:
                pass

        # -------------------------------------------------------------
        # Aggregate Layer 1: Technical QC Result
        # -------------------------------------------------------------
        mean_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
        mean_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
        mean_mae = float(np.mean(mae_list)) if mae_list else 0.0
        max_d = float(np.max(max_diff_list)) if max_diff_list else 0.0

        min_psnr = self.config.technical_thresholds.min_psnr_source_preservation
        min_ssim = self.config.technical_thresholds.min_ssim_source_preservation
        max_mae_thresh = self.config.technical_thresholds.max_mae_source_preservation

        pres_ok = (mean_psnr >= min_psnr and mean_ssim >= min_ssim and mean_mae <= max_mae_thresh)
        pres_warns = []
        if mean_psnr < min_psnr:
            pres_warns.append(f"Source preservation PSNR low: {mean_psnr:.2f} dB < {min_psnr:.1f} dB")
        elif mean_psnr < 35.0:
            pres_warns.append(f"Source preservation PSNR ({mean_psnr:.2f} dB) indicates minor lossy re-encoding difference.")
        if mean_ssim < min_ssim:
            pres_warns.append(f"Source preservation SSIM low: {mean_ssim:.4f} < {min_ssim:.4f}")
        if mean_mae > max_mae_thresh:
            pres_warns.append(f"Source preservation MAE high: {mean_mae:.2f} > {max_mae_thresh:.1f}")

        cut_ok = len(hybrid_failures) == 0
        cut_warns = []
        if hybrid_failures:
            cut_warns.append(f"Detected {len(hybrid_failures)} blended hybrid frames at hard scene transitions.")

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
            source_preservation_psnr=round(mean_psnr, 2),
            source_preservation_ssim=round(mean_ssim, 4),
            source_preservation_mae=round(mean_mae, 2),
            source_preservation_max_diff=round(max_d, 2),
            source_preservation_passed=pres_ok,
            audio_present=output_meta.has_audio,
            audio_duration_diff_sec=audio_details.get("audio_duration_diff_sec", 0.0),
            audio_sync_passed=audio_ok,
            pts_monotonic=pts_details.get("monotonic", True),
            pts_gaps_detected=pts_details.get("gaps_count", 0),
            pts_duplicates_detected=pts_details.get("duplicates_count", 0),
            pts_check_passed=pts_ok,
            scene_cuts_detected_source=len(cuts_detected),
            scene_cuts_properly_handled=cut_ok,
            status=tech_status,
            warnings=all_warns
        )

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
        artifact_diff = build_artifact_differential_result(
            src_ghosting=src_ghosting,
            src_double=src_double,
            src_tearing=src_tearing,
            src_deform=src_deform,
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
        tracked_count = len(ball_trajectories)
        if tracked_count == 0:
            ball_score = 1.0
        else:
            dup_rate = duplicate_ball_events / float(tracked_count)
            tel_rate = teleportation_count / float(tracked_count)
            deform_rate = deformed_ball_events / float(tracked_count)
            ball_score = 5.0 - min(2.0, (dup_rate / 0.10) * 2.0) - min(1.5, (tel_rate / 0.10) * 1.5) - min(1.0, (deform_rate / 0.10) * 1.0)
            if raw_out_idx >= 30 and (tracked_count / float(raw_out_idx)) < 0.15:
                ball_score -= min(2.0, (0.15 - (tracked_count / float(raw_out_idx))) / 0.15 * 2.0)
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
        if raw_out_idx > 0:
            wobble_rate = wobble_events / float(raw_out_idx)
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

        return technical_qc, temporal_qc, artifact_diff, football_qc, frame_metrics

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
        prev_prev_ball_pos: Optional[Tuple[float, float]] = None
        teleportation_count = 0
        duplicate_ball_events = 0
        deformed_ball_events = 0
        ball_frames_since_last_seen = 0
        base_ball_velocity = 130.0 * (50.0 / max(1.0, source_meta.avg_fps or 25.0))

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
            # Players & Occlusion
            players = detect_player_blobs(frame)
            total_players_checked += len(players)

            # Ball
            candidates = detect_ball_candidates(frame, players=players, min_circularity=0.50)

            best_candidate = None
            if candidates:
                if prev_ball_pos is None:
                    candidates.sort(key=lambda x: x[3], reverse=True)
                    if candidates[0][3] >= 0.55:
                        best_candidate = candidates[0]
                        ball_trajectories.append((best_candidate[0], best_candidate[1]))
                        prev_prev_ball_pos = prev_ball_pos
                        prev_ball_pos = (best_candidate[0], best_candidate[1])
                        ball_frames_since_last_seen = 0
                        if candidates[0][3] < 0.65:
                            deformed_ball_events += 1
                else:
                    allowed_displacement = base_ball_velocity * (ball_frames_since_last_seen + 1)
                    candidates.sort(key=lambda x: math.hypot(x[0] - prev_ball_pos[0], x[1] - prev_ball_pos[1]))
                    closest = candidates[0]
                    dist = math.hypot(closest[0] - prev_ball_pos[0], closest[1] - prev_ball_pos[1])
                    if dist > allowed_displacement:
                        teleportation_count += 1
                        ball_frames_since_last_seen += 1
                        if ball_frames_since_last_seen > 6:
                            prev_ball_pos = None
                            prev_prev_ball_pos = None
                    else:
                        best_candidate = closest
                        ball_trajectories.append((best_candidate[0], best_candidate[1]))
                        prev_prev_ball_pos = prev_ball_pos
                        prev_ball_pos = (best_candidate[0], best_candidate[1])
                        ball_frames_since_last_seen = 0
                        if closest[3] < 0.65:
                            deformed_ball_events += 1
            else:
                ball_frames_since_last_seen += 1
                if ball_frames_since_last_seen > 6:
                    prev_ball_pos = None
                    prev_prev_ball_pos = None

            # Ghost ball (duplicate ball) detection
            current_vel = 0.0
            if prev_ball_pos is not None and prev_prev_ball_pos is not None:
                current_vel = math.hypot(prev_ball_pos[0] - prev_prev_ball_pos[0], prev_ball_pos[1] - prev_prev_ball_pos[1])
            max_dup_dist = max(60.0, min(150.0, current_vel * 1.5))
            min_dup_dist = 8.0

            has_duplicate = False
            if best_candidate is not None and len(candidates) >= 2:
                for c in candidates:
                    if c is not best_candidate:
                        dist = math.hypot(best_candidate[0] - c[0], best_candidate[1] - c[1])
                        if min_dup_dist <= dist <= max_dup_dist:
                            has_duplicate = True
                            break
            elif best_candidate is None and len(candidates) >= 2:
                for i in range(len(candidates)):
                    for j in range(i + 1, len(candidates)):
                        dist = math.hypot(candidates[i][0] - candidates[j][0], candidates[i][1] - candidates[j][1])
                        if min_dup_dist <= dist <= max_dup_dist:
                            has_duplicate = True
                            break
                    if has_duplicate:
                        break
            if has_duplicate:
                duplicate_ball_events += 1
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
        tracked_count = len(ball_trajectories)
        if tracked_count == 0:
            ball_score = 1.0
        else:
            dup_rate = duplicate_ball_events / float(tracked_count)
            tel_rate = teleportation_count / float(tracked_count)
            deform_rate = deformed_ball_events / float(tracked_count)
            ball_score = 5.0 - min(2.0, (dup_rate / 0.10) * 2.0) - min(1.5, (tel_rate / 0.10) * 1.5) - min(1.0, (deform_rate / 0.10) * 1.0)
            if eval_idx >= 30 and (tracked_count / float(eval_idx)) < 0.15:
                ball_score -= min(2.0, (0.15 - (tracked_count / float(eval_idx))) / 0.15 * 2.0)
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
