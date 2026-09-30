"""Dynamic resource allocator and chunk size optimizer based on GPU VRAM, System RAM, and Model Footprint."""

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Typical peak VRAM footprint (GB) for 1080p frame interpolation
MODEL_VRAM_PROFILE_GB: Dict[str, float] = {
    "rife": 1.5,
    "blend": 0.5,
    "amt": 12.5,
    "amt-s": 8.0,
    "amt-l": 11.0,
    "amt-g": 12.5,
    "film": 4.0,
    "ifrnet": 2.0,
    "gmfss": 4.5,
    "ema-vfi": 5.2,
    "ema-vfi-s": 3.0,
    "interpany": 3.5,
    "interpany-vgg": 3.5,
    "interpany-pro": 4.0,
    "bwdif": 0.2,
    "auto": 1.5,
}


def get_gpu_info() -> List[Dict[str, Any]]:
    """Inspects available NVIDIA GPUs and their VRAM capacity in GB."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []

        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024 ** 3)
            # Query actual free memory from CUDA runtime
            try:
                free_bytes, _ = torch.cuda.mem_get_info(i)
                free_gb = free_bytes / (1024 ** 3)
            except Exception:
                free_gb = total_gb * 0.90
            gpus.append({
                "index": i,
                "name": props.name,
                "total_vram_gb": round(total_gb, 2),
                "free_vram_gb": round(free_gb, 2)
            })
        return gpus
    except Exception as e:
        logger.warning(f"Failed to query GPU info via PyTorch: {e}")
        return []


def get_system_ram_gb() -> float:
    """Returns available system RAM in GB using standard Linux /proc/meminfo or os fallback."""
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)
    except Exception:
        pass
    return 16.0  # Safe default


def calculate_optimal_allocation(
    model_name: str,
    total_frames: int,
    width: int = 1920,
    height: int = 1080,
    user_workers: Optional[int] = None,
    user_chunk_size: Optional[int] = None,
    user_gpus: Optional[List[int]] = None,
    user_batch_size: Optional[int] = None,
    target_resource_ratio: float = 0.65,
    tta: bool = False,
    safety_margin: float = 0.85
) -> Tuple[List[str], int, Dict[str, Any]]:
    """Calculates optimal worker device mapping, batch size, and chunk size.
    
    Dynamically analyzes surplus hardware capacity (free GPU VRAM, free CPU cores,
    and available system RAM) and budgets 50% - 70% of available resources to maximize
    throughput while preserving system stability.

    Returns:
        (allocated_devices, optimal_chunk_size, diagnostic_info)
    """
    target_ratio = max(0.50, min(0.70, float(target_resource_ratio)))
    model_key = model_name.lower()
    base_vram = MODEL_VRAM_PROFILE_GB.get(model_key, 4.0)
    if tta and "ema-vfi" in model_key:
        base_vram += 0.5

    # Scale model VRAM footprint if resolution differs from 1080p
    res_factor = (width * height) / (1920 * 1080)
    model_vram_gb = max(0.5, base_vram * max(0.5, min(4.0, res_factor)))

    gpu_list = get_gpu_info()
    sys_ram_gb = get_system_ram_gb()
    cpu_count = os.cpu_count() or 4
    frame_ram_bytes = width * height * 3  # Uncompressed RGB24 frame in RAM

    # Filter target GPUs
    if user_gpus is not None:
        if len(user_gpus) == 0:
            target_gpus = []
        else:
            target_gpus = [g for g in gpu_list if g["index"] in user_gpus]
    else:
        target_gpus = gpu_list

    # Compute CPU worker ceiling: use 50% - 70% of logical CPU cores
    max_cpu_workers = max(1, int(cpu_count * target_ratio))
    num_gpus = len(target_gpus)
    max_w_per_gpu_cpu = max(1, max_cpu_workers // max(1, num_gpus))

    allocated_devices: List[str] = []
    optimal_batch_size = 1
    total_est_vram_gb = 0.0

    # 1. Calculate optimal workers & batch size per GPU based on surplus VRAM budget
    if not target_gpus:
        # CPU Mode
        cpu_workers = user_workers if (user_workers and user_workers > 0) else min(4, max_cpu_workers)
        allocated_devices = ["cpu"] * cpu_workers
        optimal_batch_size = user_batch_size or 1
    else:
        if user_workers is not None and user_workers > 0:
            # User specified explicit total worker count: distribute across target GPUs
            for i in range(user_workers):
                gpu = target_gpus[i % len(target_gpus)]
                allocated_devices.append(f"cuda:{gpu['index']}")
            optimal_batch_size = user_batch_size or 1
            total_est_vram_gb = user_workers * model_vram_gb * optimal_batch_size
        else:
            # Co-optimize Workers & Batch Size to utilize 50% - 70% of surplus VRAM and CPU
            for gpu in target_gpus:
                usable_vram = gpu["total_vram_gb"] * safety_margin
                free_vram = gpu.get("free_vram_gb", usable_vram)
                # Surplus VRAM budget targeting 50-70% of free VRAM
                budget_vram = free_vram * target_ratio

                if model_vram_gb >= 5.0:
                    # Heavy models (e.g. AMT-G, EMA-VFI): worker-bound with batch_size=1
                    max_w = max(1, int(usable_vram // model_vram_gb))
                    # Clamp by CPU core budget, allowing up to 6 per GPU on high-end nodes
                    max_w = min(max(6, max_w_per_gpu_cpu), max_w)
                    optimal_batch_size = user_batch_size or 1
                    cost_per_w = model_vram_gb
                else:
                    # Light / Medium models (e.g. RIFE, Blend, IFRNet, EMA-VFI-s):
                    # Co-optimize concurrent workers AND batch size
                    vram_step = model_vram_gb * 0.45
                    if user_batch_size is not None and user_batch_size > 0:
                        optimal_batch_size = user_batch_size
                    else:
                        # Auto-calculate batch size to saturate VRAM budget
                        target_w = min(max_w_per_gpu_cpu, min(12, max(2, int(usable_vram // (model_vram_gb * 3)))))
                        vram_per_w = budget_vram / max(1, target_w)
                        raw_b = int(max(1, 1 + (vram_per_w - model_vram_gb) / vram_step))
                        # Snap to power of 2: 1, 2, 4, 8, 16
                        optimal_batch_size = min(16, 2 ** int(math.log2(max(1, raw_b))))

                    cost_per_w = model_vram_gb + (optimal_batch_size - 1) * vram_step
                    max_w = max(1, min(max_w_per_gpu_cpu, int(budget_vram // cost_per_w)))
                    if gpu["total_vram_gb"] >= 16.0 and model_key == "rife":
                        max_w = max(4, max_w)

                total_est_vram_gb += max_w * cost_per_w
                for _ in range(max_w):
                    allocated_devices.append(f"cuda:{gpu['index']}")

    num_workers = len(allocated_devices)

    # 2. Calculate optimal chunk size
    # Target: ensure all workers are saturated (at least 2-3 chunks per worker for dynamic load balancing)
    if user_chunk_size is not None and user_chunk_size > 0:
        chunk_size = user_chunk_size
    else:
        target_chunks = max(1, num_workers * 2)
        raw_chunk_size = max(50, math.ceil(total_frames / target_chunks))

        # Clamp between bounds:
        # - Min chunk: 200 frames (~8s at 25fps) to keep seek/encode overhead < 2%
        # - Max chunk: 1500 frames (~60s) to keep progressive encoding responsive
        min_chunk = min(200, max(50, total_frames // num_workers))
        max_chunk = 1500

        chunk_size = max(min_chunk, min(max_chunk, raw_chunk_size))

        # 3. System RAM safety constraint
        # RAM used if all workers hold a full chunk in memory simultaneously:
        # num_workers * chunk_size * frame_ram_bytes <= target_ratio of available system RAM
        max_safe_ram_bytes = sys_ram_gb * (1024 ** 3) * target_ratio
        ram_per_chunk_frame = num_workers * frame_ram_bytes
        if ram_per_chunk_frame > 0:
            ram_bounded_chunk = int(max_safe_ram_bytes // ram_per_chunk_frame)
            if ram_bounded_chunk < chunk_size:
                logger.info(
                    f"Throttling chunk_size from {chunk_size} to {ram_bounded_chunk} "
                    f"to respect System RAM budget ({sys_ram_gb:.1f} GB available, target {target_ratio*100:.0f}%)."
                )
                chunk_size = max(50, ram_bounded_chunk)

    estimated_chunks = math.ceil(total_frames / max(1, chunk_size))
    primary_free_vram = target_gpus[0]["free_vram_gb"] if target_gpus else 0.0
    pct_vram_used = round((total_est_vram_gb / max(0.1, primary_free_vram * len(target_gpus or [1]))) * 100, 1) if target_gpus else 0.0

    info = {
        "model_name": model_name,
        "model_vram_gb": round(model_vram_gb, 2),
        "total_frames": total_frames,
        "num_workers": num_workers,
        "devices": allocated_devices,
        "optimal_batch_size": optimal_batch_size,
        "optimal_chunk_size": chunk_size,
        "estimated_chunks": estimated_chunks,
        "system_ram_gb": round(sys_ram_gb, 2),
        "target_resource_ratio": target_ratio,
        "estimated_vram_gb": round(total_est_vram_gb, 2),
        "pct_free_vram_budget": pct_vram_used,
        "gpus_detected": [
            f"{g['name']} ({g['total_vram_gb']}GB VRAM, {g.get('free_vram_gb', g['total_vram_gb'])}GB free)"
            for g in gpu_list
        ] or ["None (CPU)"]
    }

    return allocated_devices, chunk_size, info
