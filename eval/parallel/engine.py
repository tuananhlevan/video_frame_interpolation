"""Multi-worker & Multi-GPU parallel evaluation engine.

Implements temporal chunk partitioning with 2-frame boundary overlap,
concurrent worker execution (via multiprocessing), and map-reduce aggregation.
"""

from collections import deque
from dataclasses import dataclass, field
import logging
import math
import multiprocessing as mp
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from upframe.core.types import VideoMetadata
from upframe.pipeline.decode import VideoDecoder
from eval.config import EvaluationConfig
from eval.layer1_technical.scene_cuts import compute_frame_difference
from eval.layer1_technical.source_preservation import compute_psnr, compute_ssim
from eval.layer2_temporal.optical_flow import evaluate_optical_flow_consistency
from eval.layer2_temporal.smoothness import compute_frame_motion_vector
from eval.layer3_football.artifacts import measure_frame_artifacts
from eval.layer3_football.ball import detect_ball_candidates, evaluate_ball_triplet, get_pitch_coverage
from eval.layer3_football.goal_net import detect_goal_candidate_roi
from eval.layer3_football.pitch_geometry import extract_pitch_lines, evaluate_pitch_geometry_triplet
from eval.layer3_football.player_occlusion import detect_player_blobs
from eval.types import (
    ArtifactDifferentialResult,
    FootballQCResult,
    TechnicalQCResult,
    TemporalQCResult,
)
from eval.visual.diff_maps import generate_difference_heatmap

logger = logging.getLogger("eval.parallel")


@dataclass
class ChunkEvaluationTask:
    """Specification of a video slice to evaluate."""
    chunk_id: int
    source_path: str
    output_path: str
    eval_dir: str
    out_start_frame: int       # Includes boundary overlap
    out_end_frame: int         # Exclusive
    out_active_start: int      # First frame whose metrics are recorded
    src_start_frame: int
    src_end_frame: int
    src_active_start: int
    width: int
    height: int
    src_w: int
    src_h: int
    device: str
    cut_threshold: float = 0.35
    source_baseline_stride: int = 5
    generate_visuals: bool = True
    extract_injected: bool = True
    generate_diff: bool = True
    max_injected: int = 50
    max_diff: int = 20


@dataclass
class ChunkEvaluationOutput:
    """Aggregated evaluation metrics for a single chunk."""
    chunk_id: int
    out_evaluated_count: int = 0
    psnr_list: List[float] = field(default_factory=list)
    ssim_list: List[float] = field(default_factory=list)
    mae_list: List[float] = field(default_factory=list)
    max_diff_list: List[float] = field(default_factory=list)
    cuts_detected: List[int] = field(default_factory=list)
    hybrid_failures: List[Dict[str, Any]] = field(default_factory=list)
    src_ghosting: List[float] = field(default_factory=list)
    src_double: List[float] = field(default_factory=list)
    src_tearing: List[float] = field(default_factory=list)
    src_deform: List[float] = field(default_factory=list)
    motion_vectors: List[Tuple[float, float, float]] = field(default_factory=list)
    shifts: List[Tuple[float, float]] = field(default_factory=list)
    consecutive_diffs: List[float] = field(default_factory=list)
    lag2_diffs: List[float] = field(default_factory=list)
    var_flickers: List[float] = field(default_factory=list)
    warping_errors: List[float] = field(default_factory=list)
    motion_boundary_errors: List[float] = field(default_factory=list)
    motion_magnitudes: List[float] = field(default_factory=list)
    frame_metrics: List[Dict[str, Any]] = field(default_factory=list)
    out_ghosting: List[float] = field(default_factory=list)
    out_double: List[float] = field(default_factory=list)
    out_tearing: List[float] = field(default_factory=list)
    out_deform: List[float] = field(default_factory=list)
    ball_trajectories: List[Tuple[float, float]] = field(default_factory=list)
    teleportation_count: int = 0
    duplicate_ball_events: int = 0
    deformed_ball_events: int = 0
    total_players_checked: int = 0
    solidity_anomalies: int = 0
    occlusion_events: int = 0
    occlusion_failures: int = 0
    total_lines_detected: int = 0
    wobble_events: int = 0
    line_counts_per_frame: List[int] = field(default_factory=list)
    laplacian_variances: List[float] = field(default_factory=list)
    goal_candidate_rois: List[Tuple[int, int, int, int]] = field(default_factory=list)
    diffs_tl: List[float] = field(default_factory=list)
    diffs_tr: List[float] = field(default_factory=list)
    edge_acc_tl: Optional[np.ndarray] = None
    edge_acc_tr: Optional[np.ndarray] = None
    edge_samples_tl: int = 0
    edge_samples_tr: int = 0
    temporal_residuals: List[float] = field(default_factory=list)
    pitch_coverage_samples: List[float] = field(default_factory=list)
    src_ball_detected_count: int = 0
    triplets_ball_active: int = 0
    triplets_ball_matched: int = 0
    triplets_ball_dissolved: int = 0
    triplets_ball_wobble: int = 0
    pitch_line_residuals: List[float] = field(default_factory=list)
    pitch_warp_events: int = 0
    error: Optional[str] = None


def evaluate_chunk_worker(task: ChunkEvaluationTask) -> ChunkEvaluationOutput:
    """Worker function executed in separate process to evaluate a video slice."""
    out = ChunkEvaluationOutput(chunk_id=task.chunk_id)

    # Set CUDA device affinity if assigned
    if task.device.startswith("cuda:"):
        try:
            dev_idx = task.device.split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = dev_idx
        except Exception:
            pass

    width, height = task.width, task.height
    src_w, src_h = task.src_w, task.src_h

    injected_dir = os.path.join(task.eval_dir, "injected_frames")
    diff_dir = os.path.join(task.eval_dir, "diff_maps")

    h_tl, w_tl = int(height * 0.18), int(width * 0.30)
    h_tr, w_tr = int(height * 0.15), int(width * 0.98) - int(width * 0.75)
    roi_top_left = (slice(0, h_tl), slice(0, w_tl))
    roi_top_right = (slice(0, h_tr), slice(int(width * 0.75), int(width * 0.98)))
    out.edge_acc_tl = np.zeros((h_tl, w_tl), dtype=np.float32)
    out.edge_acc_tr = np.zeros((h_tr, w_tr), dtype=np.float32)

    window: deque = deque(maxlen=3)
    gray_window: deque = deque(maxlen=3)
    indices_window: deque = deque(maxlen=3)

    prev_frame_src: Optional[np.ndarray] = None
    prev_out_odd: Optional[np.ndarray] = None
    prev_ball_pos: Optional[Tuple[float, float]] = None
    prev_prev_ball_pos: Optional[Tuple[float, float]] = None
    ball_frames_since_last_seen = 0
    base_ball_velocity = 130.0  # Output is strictly 50.0 FPS

    saved_injected = 0
    saved_diff = 0

    dec_out = VideoDecoder(task.output_path, width, height)
    dec_src = VideoDecoder(task.source_path, src_w, src_h)

    out_count = task.out_end_frame - task.out_start_frame
    src_count = task.src_end_frame - task.src_start_frame

    stream_out = dec_out.stream_frames(start_frame=task.out_start_frame, count=out_count, fps=50.0)
    stream_src = dec_src.stream_frames(start_frame=task.src_start_frame, count=src_count, fps=25.0)

    raw_out_idx = task.out_start_frame
    src_idx = task.src_start_frame

    try:
        while True:
            if raw_out_idx >= task.out_end_frame:
                break
            try:
                frame = next(stream_out)
            except StopIteration:
                break

            is_even = (raw_out_idx % 2 == 0)
            is_active = (raw_out_idx >= task.out_active_start)

            # Layer 1 & Source Baseline
            if is_even:
                try:
                    frame_src = next(stream_src)
                except StopIteration:
                    frame_src = None

                if frame_src is not None:
                    if is_active and (src_idx % task.source_baseline_stride == 0):
                        m_src = measure_frame_artifacts(frame_src)
                        out.src_ghosting.append(m_src["ghosting"])
                        out.src_double.append(m_src["double_contour"])
                        out.src_tearing.append(m_src["edge_tearing"])
                        out.src_deform.append(m_src["deformation"])

                        src_cands = detect_ball_candidates(frame_src, min_circularity=0.50)
                        if src_cands:
                            out.src_ball_detected_count += 1

                    if prev_frame_src is not None:
                        diff_cut = compute_frame_difference(prev_frame_src, frame_src)
                        if diff_cut >= task.cut_threshold:
                            cut_idx = src_idx - 1
                            if is_active:
                                out.cuts_detected.append(cut_idx)
                            if prev_out_odd is not None:
                                f_inter = prev_out_odd
                                if f_inter.shape != frame_src.shape:
                                    f_inter = cv2.resize(f_inter, (frame_src.shape[1], frame_src.shape[0]))
                                diff_to_a = compute_frame_difference(f_inter, prev_frame_src)
                                diff_to_b = compute_frame_difference(f_inter, frame_src)
                                ideal_blend = (prev_frame_src.astype(np.float32) * 0.5 + frame_src.astype(np.float32) * 0.5).astype(np.uint8)
                                diff_to_blend = compute_frame_difference(f_inter, ideal_blend)
                                is_hybrid = (diff_to_a > 0.20 and diff_to_b > 0.20) or (diff_to_blend < 0.12 and diff_cut >= task.cut_threshold)
                                if is_hybrid and is_active:
                                    out.hybrid_failures.append({
                                        "source_transition": (cut_idx, src_idx),
                                        "output_intermediate_frame": 2 * cut_idx + 1,
                                        "cut_difference": float(diff_cut),
                                        "diff_to_source_a": float(diff_to_a),
                                        "diff_to_source_b": float(diff_to_b),
                                        "diff_to_blend": float(diff_to_blend)
                                    })

                    if is_active:
                        if frame_src.shape != frame.shape:
                            f_matched = cv2.resize(frame, (frame_src.shape[1], frame_src.shape[0]))
                        else:
                            f_matched = frame
                        out.psnr_list.append(compute_psnr(frame_src, f_matched))
                        out.ssim_list.append(compute_ssim(frame_src, f_matched))
                        diff_even = np.abs(frame_src.astype(np.float32) - f_matched.astype(np.float32))
                        out.mae_list.append(float(np.mean(diff_even)))
                        out.max_diff_list.append(float(np.max(diff_even)))

                    prev_frame_src = frame_src
                src_idx += 1
            else:
                prev_out_odd = frame

            # Layers 2 & 3 Output evaluation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

            if len(window) >= 1 and is_active:
                prev_frame = window[-1]
                prev_gray = gray_window[-1]

                dx, dy, mag = compute_frame_motion_vector(prev_frame, frame)
                out.motion_vectors.append((dx, dy, mag))
                out.shifts.append((dx, dy))

                if raw_out_idx % 10 == 0:
                    p_tl = gray[roi_top_left]
                    p_tr = gray[roi_top_right]
                    if out.edge_acc_tl is not None and p_tl.shape == out.edge_acc_tl.shape:
                        out.edge_acc_tl += (cv2.Canny(p_tl, 80, 180) > 0).astype(np.float32)
                        out.edge_samples_tl += 1
                    if out.edge_acc_tr is not None and p_tr.shape == out.edge_acc_tr.shape:
                        out.edge_acc_tr += (cv2.Canny(p_tr, 80, 180) > 0).astype(np.float32)
                        out.edge_samples_tr += 1

                for (ys, xs), diff_list in [(roi_top_left, out.diffs_tl), (roi_top_right, out.diffs_tr)]:
                    patch1 = prev_gray[ys, xs]
                    patch2 = gray[ys, xs]
                    diff_list.append(float(np.mean(np.abs(patch1.astype(np.float32) - patch2.astype(np.float32)))))

                g_curr_f = gray.astype(np.float32)
                g_prev_f = prev_gray.astype(np.float32)
                out.consecutive_diffs.append(float(np.mean(np.abs(g_curr_f - g_prev_f))))
                lap_curr = cv2.Laplacian(g_curr_f, cv2.CV_32F).var()
                lap_prev = cv2.Laplacian(g_prev_f, cv2.CV_32F).var()
                out.var_flickers.append(abs(lap_curr - lap_prev) / (lap_curr + lap_prev + 1e-5))

            if len(window) >= 2 and is_active:
                prev2_gray = gray_window[-2]
                g_curr_f = gray.astype(np.float32)
                g_prev2_f = prev2_gray.astype(np.float32)
                g_prev_f = gray_window[-1].astype(np.float32)
                out.lag2_diffs.append(float(np.mean(np.abs(g_curr_f - g_prev2_f))))
                pred_lin = 0.5 * (g_curr_f + g_prev2_f)
                out.temporal_residuals.append(float(np.mean(np.abs(g_prev_f - pred_lin))))

            if len(window) >= 2 and (indices_window[-1] % 2 == 1) and (indices_window[-1] >= task.out_active_start):
                fa = window[-2]
                fx = window[-1]
                fb = frame
                odd_idx = indices_window[-1]

                m = measure_frame_artifacts(fx)
                out.out_ghosting.append(m["ghosting"])
                out.out_double.append(m["double_contour"])
                out.out_tearing.append(m["edge_tearing"])
                out.out_deform.append(m["deformation"])

                flow_res = evaluate_optical_flow_consistency(fa, fx, fb)
                w_err = flow_res["mean_warping_error"]
                mb_err = flow_res["motion_boundary_error"]
                mag_val = flow_res["motion_magnitude"]
                out.warping_errors.append(w_err)
                out.motion_boundary_errors.append(mb_err)
                out.motion_magnitudes.append(mag_val)

                out.frame_metrics.append({
                    "frame_index": odd_idx,
                    "warping_error": w_err,
                    "motion_boundary_error": mb_err,
                    "motion_magnitude": mag_val
                })

                # 1. Triplet Ball Evaluation
                cov_x = get_pitch_coverage(fx)
                players_x = detect_player_blobs(fx) if cov_x >= 0.15 else []
                lines_x = extract_pitch_lines(fx) if cov_x >= 0.15 else []
                ball_res = evaluate_ball_triplet(fa, fx, fb, players_x=players_x, pitch_lines_x=lines_x)
                if ball_res["ball_active"]:
                    out.triplets_ball_active += 1
                    if ball_res["ball_matched"]:
                        out.triplets_ball_matched += 1
                        out.ball_trajectories.append(ball_res["pos_x"])
                    if ball_res["ball_dissolved"]:
                        out.triplets_ball_dissolved += 1
                    if ball_res["ball_wobble"]:
                        out.triplets_ball_wobble += 1
                    if ball_res["ball_deformed"]:
                        out.deformed_ball_events += 1
                    if ball_res["ball_duplicate"]:
                        out.duplicate_ball_events += 1

                # 2. Triplet Pitch Geometry Evaluation
                pitch_res = evaluate_pitch_geometry_triplet(fa, fx, fb)
                if pitch_res["pitch_active"]:
                    out.pitch_line_residuals.append(pitch_res["line_residual"])
                    out.pitch_warp_events += pitch_res["warp_events"]
                    out.total_lines_detected += pitch_res["lines_detected_x"]

                if task.generate_visuals:
                    if task.extract_injected and saved_injected < (task.max_injected // 2):
                        fname = f"injected_frame_{odd_idx:06d}.jpg"
                        cv2.imwrite(os.path.join(injected_dir, fname), fx, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        saved_injected += 1
                    if task.generate_diff and saved_diff < (task.max_diff // 2):
                        dname = f"diff_heatmap_{odd_idx:06d}.jpg"
                        generate_difference_heatmap(fa, fx, os.path.join(diff_dir, dname))
                        saved_diff += 1

            if is_active:
                p_cov = get_pitch_coverage(frame)
                if raw_out_idx % 10 == 0:
                    out.pitch_coverage_samples.append(p_cov)

                # Players & Occlusion (only evaluate on active football pitch)
                if p_cov >= 0.15:
                    players = detect_player_blobs(frame)
                    out.total_players_checked += len(players)
                    for p in players:
                        if p["solidity"] < 0.45:
                            out.solidity_anomalies += 1
                    for i in range(len(players)):
                        x1, y1, w1, h1 = players[i]["bbox"]
                        for j in range(i + 1, len(players)):
                            x2, y2, w2, h2 = players[j]["bbox"]
                            ix = max(x1, x2)
                            iy = max(y1, y2)
                            iw = min(x1 + w1, x2 + w2) - ix
                            ih = min(y1 + h1, y2 + h2) - iy
                            if iw > 8 and ih > 15:
                                out.occlusion_events += 1
                                roi = frame[iy:iy + ih, ix:ix + iw]
                                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                lap_var = float(cv2.Laplacian(gray_roi, cv2.CV_32F).var())
                                p1_gray = cv2.cvtColor(frame[y1:y1 + h1, x1:x1 + w1], cv2.COLOR_BGR2GRAY)
                                p2_gray = cv2.cvtColor(frame[y2:y2 + h2, x2:x2 + w2], cv2.COLOR_BGR2GRAY)
                                ref_texture = float((cv2.Laplacian(p1_gray, cv2.CV_32F).var() + cv2.Laplacian(p2_gray, cv2.CV_32F).var()) / 2.0)
                                if ref_texture > 65.0 and lap_var < max(25.0, 0.35 * ref_texture):
                                    out.occlusion_failures += 1

                if raw_out_idx % 25 == 0:
                    g_roi = detect_goal_candidate_roi(frame)
                    if g_roi is not None:
                        out.goal_candidate_rois.append(g_roi)

                if out.goal_candidate_rois:
                    gx, gy, gw, gh = out.goal_candidate_rois[-1]
                    crop = gray[gy:gy + gh, gx:gx + gw]
                    if crop.size > 0:
                        _, thresh = cv2.threshold(crop, 200, 255, cv2.THRESH_BINARY)
                        out.laplacian_variances.append(float(cv2.Laplacian(thresh, cv2.CV_32F).var()))
                out.out_evaluated_count += 1

            window.append(frame)
            gray_window.append(gray)
            indices_window.append(raw_out_idx)
            raw_out_idx += 1

    except Exception as e:
        logger.exception(f"Chunk {task.chunk_id} failed: {e}")
        out.error = str(e)
    finally:
        try:
            stream_src.close()
        except Exception:
            pass
        try:
            stream_out.close()
        except Exception:
            pass

    return out


class ParallelEvaluationEngine:
    """Orchestrates temporal chunking, concurrent worker execution, and aggregation."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def evaluate_parallel(
        self,
        source_path: str,
        output_path: str,
        source_meta: VideoMetadata,
        output_meta: VideoMetadata,
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
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[TechnicalQCResult, TemporalQCResult, ArtifactDifferentialResult, FootballQCResult, List[Dict[str, Any]]]:
        """Executes multi-worker parallel evaluation across temporal video chunks."""
        # 1. Resolve workers and devices
        num_workers = max(1, self.config.workers)
        gpus: List[str] = []
        if self.config.gpus:
            gpu_list = [x.strip() for x in self.config.gpus.split(",") if x.strip()]
            gpus = [f"cuda:{g}" for g in gpu_list]

        if not gpus:
            devices = ["cpu"] * num_workers
        else:
            devices = [gpus[i % len(gpus)] for i in range(num_workers)]

        # 2. Determine frame boundaries aligned to source frames
        max_eval_output = (max_frames * 2) if max_frames else None
        total_output_frames = min(output_meta.nb_frames, max_eval_output) if max_eval_output else output_meta.nb_frames
        total_source_frames = (total_output_frames + 1) // 2

        src_per_worker = max(1, total_source_frames // num_workers)
        tasks: List[ChunkEvaluationTask] = []

        for i in range(num_workers):
            src_active_start = i * src_per_worker
            src_active_end = total_source_frames if i == num_workers - 1 else (i + 1) * src_per_worker

            chunk_active_start = src_active_start * 2
            chunk_active_end = total_output_frames if i == num_workers - 1 else src_active_end * 2

            # Overlap: 2 output frames and 1 source frame for sliding window warmup
            if i > 0:
                out_start = chunk_active_start - 2
                src_start = src_active_start - 1
            else:
                out_start = 0
                src_start = 0

            out_end = chunk_active_end
            src_end = src_active_end

            tasks.append(
                ChunkEvaluationTask(
                    chunk_id=i,
                    source_path=source_path,
                    output_path=output_path,
                    eval_dir=eval_dir,
                    out_start_frame=out_start,
                    out_end_frame=out_end,
                    out_active_start=chunk_active_start,
                    src_start_frame=src_start,
                    src_end_frame=src_end,
                    src_active_start=src_active_start,
                    width=output_meta.width,
                    height=output_meta.height,
                    src_w=source_meta.width,
                    src_h=source_meta.height,
                    device=devices[i],
                    cut_threshold=self.config.technical_thresholds.scene_cut_threshold,
                    source_baseline_stride=self.config.source_baseline_stride,
                    generate_visuals=self.config.generate_visuals,
                    extract_injected=self.config.extract_injected_frames,
                    generate_diff=self.config.generate_diff_maps,
                    max_injected=self.config.max_injected_frames_to_save,
                    max_diff=self.config.max_diff_maps_to_save
                )
            )

        logger.info(f"Dispatching parallel evaluation across {num_workers} workers: {[t.device for t in tasks]}")

        # 3. Execute via multiprocessing spawn pool
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            results = pool.map(evaluate_chunk_worker, tasks)

        # Sort results by chunk_id
        results.sort(key=lambda x: x.chunk_id)

        # 4. Map-Reduce Aggregation
        return self._aggregate_results(
            results,
            source_meta,
            output_meta,
            fps_ok,
            frame_ok,
            fps_warns,
            audio_ok,
            audio_details,
            audio_warns,
            pts_ok,
            pts_details,
            pts_warns,
            total_output_frames
        )

    def _aggregate_results(
        self,
        results: List[ChunkEvaluationOutput],
        source_meta: VideoMetadata,
        output_meta: VideoMetadata,
        fps_ok: bool,
        frame_ok: bool,
        fps_warns: List[str],
        audio_ok: bool,
        audio_details: Dict[str, Any],
        audio_warns: List[str],
        pts_ok: bool,
        pts_details: Dict[str, Any],
        pts_warns: List[str],
        total_eval_frames: int
    ) -> Tuple[TechnicalQCResult, TemporalQCResult, ArtifactDifferentialResult, FootballQCResult, List[Dict[str, Any]]]:
        from eval.layer3_football.artifacts import build_artifact_differential_result

        # Merge accumulators
        psnr_list: List[float] = []
        ssim_list: List[float] = []
        mae_list: List[float] = []
        max_diff_list: List[float] = []
        cuts_detected: List[int] = []
        hybrid_failures: List[Dict[str, Any]] = []

        src_ghosting: List[float] = []
        src_double: List[float] = []
        src_tearing: List[float] = []
        src_deform: List[float] = []

        motion_vectors: List[Tuple[float, float, float]] = []
        shifts: List[Tuple[float, float]] = []
        consecutive_diffs: List[float] = []
        lag2_diffs: List[float] = []
        var_flickers: List[float] = []
        warping_errors: List[float] = []
        motion_boundary_errors: List[float] = []
        motion_magnitudes: List[float] = []
        frame_metrics: List[Dict[str, Any]] = []

        out_ghosting: List[float] = []
        out_double: List[float] = []
        out_tearing: List[float] = []
        out_deform: List[float] = []

        ball_trajectories: List[Tuple[float, float]] = []
        teleportation_count = 0
        duplicate_ball_events = 0
        deformed_ball_events = 0
        total_players_checked = 0
        solidity_anomalies = 0
        occlusion_events = 0
        occlusion_failures = 0
        total_lines_detected = 0
        wobble_events = 0
        line_counts_per_frame: List[int] = []
        laplacian_variances: List[float] = []
        diffs_tl: List[float] = []
        diffs_tr: List[float] = []
        pitch_coverage_samples: List[float] = []
        src_ball_detected_count = 0
        triplets_ball_active = 0
        triplets_ball_matched = 0
        triplets_ball_dissolved = 0
        triplets_ball_wobble = 0
        pitch_line_residuals: List[float] = []
        pitch_warp_events = 0
        temporal_residuals: List[float] = []
        goal_candidate_rois: List[Tuple[int, int, int, int]] = []
        edge_acc_tl: Optional[np.ndarray] = None
        edge_acc_tr: Optional[np.ndarray] = None
        edge_samples_tl = 0
        edge_samples_tr = 0

        for r in results:
            if r.error:
                logger.warning(f"Chunk {r.chunk_id} reported error: {r.error}")
            psnr_list.extend(r.psnr_list)
            ssim_list.extend(r.ssim_list)
            mae_list.extend(r.mae_list)
            max_diff_list.extend(r.max_diff_list)
            cuts_detected.extend(r.cuts_detected)
            hybrid_failures.extend(r.hybrid_failures)

            src_ghosting.extend(r.src_ghosting)
            src_double.extend(r.src_double)
            src_tearing.extend(r.src_tearing)
            src_deform.extend(r.src_deform)

            motion_vectors.extend(r.motion_vectors)
            shifts.extend(r.shifts)
            consecutive_diffs.extend(r.consecutive_diffs)
            lag2_diffs.extend(r.lag2_diffs)
            var_flickers.extend(r.var_flickers)
            warping_errors.extend(r.warping_errors)
            motion_boundary_errors.extend(r.motion_boundary_errors)
            motion_magnitudes.extend(r.motion_magnitudes)
            frame_metrics.extend(r.frame_metrics)

            out_ghosting.extend(r.out_ghosting)
            out_double.extend(r.out_double)
            out_tearing.extend(r.out_tearing)
            out_deform.extend(r.out_deform)

            ball_trajectories.extend(r.ball_trajectories)
            teleportation_count += r.teleportation_count
            duplicate_ball_events += r.duplicate_ball_events
            deformed_ball_events += r.deformed_ball_events
            total_players_checked += r.total_players_checked
            solidity_anomalies += r.solidity_anomalies
            occlusion_events += r.occlusion_events
            occlusion_failures += r.occlusion_failures
            total_lines_detected += r.total_lines_detected
            wobble_events += r.wobble_events
            line_counts_per_frame.extend(r.line_counts_per_frame)
            laplacian_variances.extend(r.laplacian_variances)
            diffs_tl.extend(r.diffs_tl)
            diffs_tr.extend(r.diffs_tr)
            pitch_coverage_samples.extend(r.pitch_coverage_samples)
            src_ball_detected_count += r.src_ball_detected_count
            triplets_ball_active += r.triplets_ball_active
            triplets_ball_matched += r.triplets_ball_matched
            triplets_ball_dissolved += r.triplets_ball_dissolved
            triplets_ball_wobble += r.triplets_ball_wobble
            pitch_line_residuals.extend(r.pitch_line_residuals)
            pitch_warp_events += r.pitch_warp_events
            goal_candidate_rois.extend(r.goal_candidate_rois)
            temporal_residuals.extend(r.temporal_residuals)
            edge_samples_tl += r.edge_samples_tl
            edge_samples_tr += r.edge_samples_tr
            if r.edge_acc_tl is not None:
                if edge_acc_tl is None:
                    edge_acc_tl = r.edge_acc_tl.copy()
                else:
                    edge_acc_tl += r.edge_acc_tl
            if r.edge_acc_tr is not None:
                if edge_acc_tr is None:
                    edge_acc_tr = r.edge_acc_tr.copy()
                else:
                    edge_acc_tr += r.edge_acc_tr

        # -------------------------------------------------------------
        # Layer 1 Technical QC
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

        # Scene cut handling tolerance:
        # In multi-minute broadcast footage (>= 5 cuts detected), up to 2 isolated ambiguous/blended cuts
        # trigger an informational warning rather than a fatal pipeline rejection.
        cut_ok = (len(hybrid_failures) == 0) or (len(hybrid_failures) <= 2 and len(cuts_detected) >= 5)
        cut_warns = []
        if hybrid_failures:
            cut_warns.append(f"Detected {len(hybrid_failures)} blended hybrid frames at hard scene transitions (out of {len(cuts_detected)} cuts detected).")

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
        # Layer 2 Temporal QC
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
        discontinuity_rate = len(discontinuity_indices) / float(max(1, len(motion_vectors)))
        discontinuity_penalty = min(0.6, (discontinuity_rate / 0.02) * 0.6)
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
        mean_res = float(np.mean(temporal_residuals)) if temporal_residuals else mean_d1
        flicker_score = float(mean_res * 0.15 + mean_var_flicker * 3.5)

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
        # Layer 3 Artifact Differential
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
            motion_magnitudes=motion_magnitudes,
            src_flicker=0.0,
            out_flicker=flicker_score
        )

        # -------------------------------------------------------------
        # Layer 3 Football QC
        # -------------------------------------------------------------
        # Ball Integrity
        avg_pitch_pct = float(np.mean(pitch_coverage_samples)) if pitch_coverage_samples else 0.0
        scene_has_pitch = (avg_pitch_pct >= 0.15)

        if triplets_ball_active == 0:
            # Ball-free or non-pitch scene: no ball artifacts expected -> clean (5.0)
            ball_score = 5.0
        else:
            dissolve_rate = triplets_ball_dissolved / float(triplets_ball_active)
            matched_count = max(1, triplets_ball_matched)
            dup_rate = duplicate_ball_events / float(matched_count)
            wobble_rate = triplets_ball_wobble / float(matched_count)
            deform_rate = deformed_ball_events / float(matched_count)

            ball_score = 5.0 - min(2.5, dissolve_rate * 4.0) - min(1.5, (dup_rate / 0.15) * 1.5) - min(1.0, (wobble_rate / 0.15) * 1.0) - min(0.5, (deform_rate / 0.15) * 0.5)
            ball_score = max(1.0, min(5.0, ball_score))

        player_score = 5.0
        if total_players_checked > 0:
            anomaly_rate = solidity_anomalies / float(total_players_checked)
            player_score -= min(3.5, anomaly_rate * 25.0)
        player_score = max(1.0, min(5.0, player_score))

        occlusion_score = 5.0
        if occlusion_events > 0:
            fail_rate = occlusion_failures / float(occlusion_events)
            occlusion_score -= min(3.5, fail_rate * 5.0)
        occlusion_score = max(1.0, min(5.0, occlusion_score))

        # Pitch Geometry
        if not scene_has_pitch or not pitch_line_residuals:
            pitch_score = 5.0
        else:
            mean_line_res = float(np.mean(pitch_line_residuals))
            warp_rate = pitch_warp_events / float(max(1, len(pitch_line_residuals)))
            
            res_penalty = 0.0
            warp_penalty = 0.0
            if mean_line_res > 0.012:
                res_penalty = min(2.5, ((mean_line_res - 0.012) / 0.035) * 2.5)
                warp_penalty = min(2.0, (warp_rate / 0.05) * 2.0)
            
            pitch_score = max(1.0, min(5.0, 5.0 - res_penalty - warp_penalty))

        # Goal net (Presence-gated)
        goal_score = 5.0
        if goal_candidate_rois and len(laplacian_variances) >= 4:
            even_mean = float(np.mean(laplacian_variances[0::2]))
            odd_mean = float(np.mean(laplacian_variances[1::2]))
            diff_ratio = abs(even_mean - odd_mean) / (even_mean + 1e-5)
            if diff_ratio > 0.20:
                goal_score -= min(2.5, diff_ratio * 4.0)
            goal_score = max(1.0, min(5.0, goal_score))

        # Broadcast Graphics (Static Persistence-gated)
        has_static_graphics = False
        valid_jitter_diffs: List[float] = []
        for d_list, edge_acc, n_samples in [
            (diffs_tl, edge_acc_tl, edge_samples_tl),
            (diffs_tr, edge_acc_tr, edge_samples_tr)
        ]:
            if n_samples >= 5 and edge_acc is not None:
                mean_edge = edge_acc / float(n_samples)
                if np.sum(mean_edge >= 0.60) >= 40:
                    has_static_graphics = True
                    if d_list:
                        valid_jitter_diffs.append(float(np.mean(d_list)))

        if not has_static_graphics or not valid_jitter_diffs:
            graphics_score = 5.0
            avg_jitter = 0.0
        else:
            avg_jitter = float(np.mean(valid_jitter_diffs))
            graphics_score = 5.0
            if avg_jitter > 3.0:
                graphics_score -= min(3.0, (avg_jitter - 3.0) * 0.4)
            graphics_score = max(1.0, min(5.0, graphics_score))
        graphics_score = max(1.0, min(5.0, graphics_score))

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
            detected_ball_teleportations=triplets_ball_wobble,
            detected_duplicate_balls=duplicate_ball_events,
            detected_bent_lines_count=pitch_warp_events,
            graphics_jitter_detected=(avg_jitter > 5.0),
            details={
                "triplets_ball_active": triplets_ball_active,
                "triplets_ball_matched": triplets_ball_matched,
                "triplets_ball_dissolved": triplets_ball_dissolved,
                "triplets_ball_wobble": triplets_ball_wobble,
                "mean_pitch_line_residual": round(float(np.mean(pitch_line_residuals)), 5) if pitch_line_residuals else 0.0,
                "pitch_warp_events": pitch_warp_events,
            }
        )

        return technical_qc, temporal_qc, artifact_diff, football_qc, frame_metrics
