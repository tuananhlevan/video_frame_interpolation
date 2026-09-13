"""Production Performance Profiler: throughput, Realtime Factor (RTF), and GPU/CPU stats."""

import time
from typing import Optional
import torch
from eval.config import PerformanceThresholds
from eval.types import PerformanceQCResult


class PerformanceTracker:
    """Context manager / timer for tracking video processing performance."""

    def __init__(self, source_duration_sec: float, total_generated_frames: int):
        self.source_duration_sec = max(0.001, source_duration_sec)
        self.total_generated_frames = total_generated_frames
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.peak_vram_mb: Optional[float] = None

    def start(self) -> None:
        self.start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def stop(self) -> None:
        self.end_time = time.perf_counter()
        if torch.cuda.is_available():
            self.peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    @property
    def elapsed_seconds(self) -> float:
        if self.end_time > self.start_time:
            return self.end_time - self.start_time
        return max(0.0, time.perf_counter() - self.start_time)


def evaluate_performance(
    processing_time_sec: Optional[float],
    source_duration_sec: float,
    output_frames_count: int,
    source_frames_count: int,
    peak_vram_mb: Optional[float] = None,
    thresholds: PerformanceThresholds = PerformanceThresholds()
) -> PerformanceQCResult:
    """Computes realtime factor and throughput metrics if production time was measured."""
    if processing_time_sec is None:
        return PerformanceQCResult(
            total_processing_time_sec=None,
            source_duration_sec=round(source_duration_sec, 2),
            realtime_factor=None,
            target_rtf_achieved=None,
            interpolations_per_sec=None,
            frames_per_sec=None,
            peak_gpu_memory_mb=None,
            performance_score=None,
            is_measured=False
        )

    src_dur = max(0.001, source_duration_sec)
    proc_time = max(0.001, processing_time_sec)

    rtf = proc_time / src_dur
    target_achieved = (rtf <= thresholds.target_rtf_max)

    generated_frames = max(0, output_frames_count - source_frames_count)
    interpolations_per_sec = generated_frames / proc_time
    frames_per_sec = output_frames_count / proc_time

    # Performance score
    if rtf <= 1.0:
        perf_score = 10.0
    elif rtf <= thresholds.target_rtf_max:
        perf_score = 10.0 - ((rtf - 1.0) / 0.3) * 1.5
    else:
        perf_score = max(1.0, 8.5 - ((rtf - 1.3) / 0.7) * 4.0)

    return PerformanceQCResult(
        total_processing_time_sec=round(proc_time, 2),
        source_duration_sec=round(src_dur, 2),
        realtime_factor=round(rtf, 3),
        target_rtf_achieved=target_achieved,
        interpolations_per_sec=round(interpolations_per_sec, 2),
        frames_per_sec=round(frames_per_sec, 2),
        peak_gpu_memory_mb=round(peak_vram_mb, 1) if peak_vram_mb is not None else None,
        performance_score=round(perf_score, 2),
        is_measured=True
    )
