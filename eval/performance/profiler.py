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


def _parse_time_string_or_number(val: object) -> Optional[float]:
    """Parses a duration string or number into float seconds."""
    import re
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        m_hms = re.search(r'(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)', val)
        if m_hms:
            h = int(m_hms.group(1) or 0)
            m = int(m_hms.group(2))
            s = float(m_hms.group(3))
            return h * 3600 + m * 60 + s
        m_sec = re.search(r'([0-9.]+)', val)
        if m_sec:
            try:
                return float(m_sec.group(1))
            except ValueError:
                return None
    return None


def _extract_time_from_text(content: str) -> Optional[float]:
    """Extracts processing time in seconds from arbitrary text/log output."""
    import re
    m_hms = re.search(r'Processing\s*time\s*:\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)', content, re.IGNORECASE)
    if m_hms:
        h = int(m_hms.group(1) or 0)
        m = int(m_hms.group(2))
        s = float(m_hms.group(3))
        return h * 3600 + m * 60 + s

    patterns = [
        r'["\']?(?:Processing\s*time|processing_time_sec|processing_time|total_proc_time)["\']?\s*[:=]\s*["\']?([0-9.]+)',
        r'(?:completed|finished)\s*(?:processing\s*)?in\s*([0-9.]+)\s*s'
    ]
    for pat in patterns:
        m = re.search(pat, content, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def find_upframe_processing_time(
    output_path: str,
    source_path: Optional[str] = None,
    log_file: Optional[str] = None
) -> Optional[float]:
    """Finds and extracts processing time from an upframe log/report matching the processing video.
    
    Searches for report files ({video_stem}_report.json, {video_stem}_report.txt, {video_stem}.log)
    in standard log locations (upframe_log/, logs/, output video directory, current directory).
    Returns None if no matching processing time log is found.
    """
    import json
    import logging
    import os

    logger = logging.getLogger(__name__)
    candidates: list[str] = []

    if log_file and os.path.exists(log_file):
        candidates.append(os.path.abspath(log_file))

    out_basename = os.path.basename(output_path)
    out_stem = os.path.splitext(out_basename)[0]
    out_dir = os.path.dirname(os.path.abspath(output_path))
    src_stem = os.path.splitext(os.path.basename(source_path))[0] if source_path else ""

    search_dirs = [
        "upframe_log",
        "upframe_logs",
        "logs",
        "log",
        out_dir,
        ".",
        "evaluation_log"
    ]

    filenames = [
        f"{out_stem}_report.json",
        f"{out_stem}_report.txt",
        f"{out_stem}.log",
        f"{out_stem}_upframe.log",
        f"upframe_{out_stem}.log",
        f"{out_stem}.json",
        f"{out_stem}.txt",
    ]
    if src_stem:
        filenames.extend([
            f"{src_stem}_report.json",
            f"{src_stem}_report.txt",
            f"{src_stem}.log",
        ])

    expanded_search_dirs: list[str] = []
    for d in search_dirs:
        if os.path.exists(d) and os.path.isdir(d):
            if d not in expanded_search_dirs:
                expanded_search_dirs.append(d)
            try:
                for entry in os.scandir(d):
                    if entry.is_dir() and entry.path not in expanded_search_dirs:
                        expanded_search_dirs.append(entry.path)
            except Exception:
                pass

    filenames = [
        f"{out_stem}_report.json",
        f"{out_stem}_report.txt",
        f"{out_stem}.log",
        f"{out_stem}_upframe.log",
        f"upframe_{out_stem}.log",
        f"{out_stem}.json",
        f"{out_stem}.txt",
    ]
    if src_stem:
        filenames.extend([
            f"{src_stem}_report.json",
            f"{src_stem}_report.txt",
            f"{src_stem}.log",
            f"{src_stem}_upframe.log",
            f"upframe_{src_stem}.log",
            f"{src_stem}.json",
            f"{src_stem}.txt",
        ])

    for d in expanded_search_dirs:
        for fn in filenames:
            cand_path = os.path.abspath(os.path.join(d, fn))
            if os.path.isfile(cand_path) and cand_path not in candidates:
                candidates.append(cand_path)
        for generic_fn in ("report.json", "report.txt"):
            cand_path = os.path.abspath(os.path.join(d, generic_fn))
            if os.path.isfile(cand_path) and cand_path not in candidates:
                candidates.append(cand_path)

    for cand in candidates:
        if not os.path.isfile(cand):
            continue
        try:
            cand_base = os.path.basename(cand)
            if cand.endswith(".json"):
                with open(cand, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    json_out = str(data.get("output_path", data.get("output_file", "")))
                    json_in = str(data.get("input_path", data.get("input_file", "")))
                    if cand_base in ("report.json", "report.txt"):
                        matched = False
                        if out_stem and (out_stem in json_out or out_stem in json_in or out_stem in cand):
                            matched = True
                        if src_stem and (src_stem in json_out or src_stem in json_in or src_stem in cand):
                            matched = True
                        if not matched:
                            continue
                    elif json_out and out_stem not in json_out and out_basename not in json_out:
                        continue
                    for key in ("processing_time_sec", "total_processing_time_sec", "processing_time"):
                        if key in data and data[key] is not None:
                            val = _parse_time_string_or_number(data[key])
                            if val is not None and val > 0:
                                logger.info(f"Discovered upframe processing time ({val:.2f}s) in log: {cand}")
                                return val
            else:
                with open(cand, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if cand_base in ("report.txt", "report.json"):
                    matched = False
                    if out_stem and (out_stem in content or out_stem in cand):
                        matched = True
                    if src_stem and (src_stem in content or src_stem in cand):
                        matched = True
                    if not matched:
                        continue
                val = _extract_time_from_text(content)
                if val is not None and val > 0:
                    logger.info(f"Discovered upframe processing time ({val:.2f}s) in log: {cand}")
                    return val
        except Exception as e:
            logger.debug(f"Error reading candidate log {cand}: {e}")

    return None
