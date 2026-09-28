"""Dynamic resource allocator and chunk size optimizer based on GPU VRAM, System RAM, and Model Footprint."""

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

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


def get_gpu_info() -> List[Dict[str, any]]:
    """Inspects available NVIDIA GPUs and their VRAM capacity in GB."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024 ** 3)
            # Query free memory if possible
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
    tta: bool = False,
    safety_margin: float = 0.85
) -> Tuple[List[str], int, Dict[str, any]]:
    """Calculates optimal worker device mapping and chunk size.
    
    Returns:
        (allocated_devices, optimal_chunk_size, diagnostic_info)
    """
    model_key = model_name.lower()
    base_vram = MODEL_VRAM_PROFILE_GB.get(model_key, 4.0)
    if tta and "ema-vfi" in model_key:
        base_vram += 0.5

    # Scale model VRAM footprint if resolution differs from 1080p
    res_factor = (width * height) / (1920 * 1080)
    model_vram_gb = max(0.5, base_vram * max(0.5, min(4.0, res_factor)))

    gpu_list = get_gpu_info()
    sys_ram_gb = get_system_ram_gb()
    frame_ram_bytes = width * height * 3  # Uncompressed RGB24 frame in RAM

    # Filter target GPUs
    if user_gpus is not None:
        if len(user_gpus) == 0:
            target_gpus = []
        else:
            target_gpus = [g for g in gpu_list if g["index"] in user_gpus]
    else:
        target_gpus = gpu_list

    allocated_devices: List[str] = []

    # 1. Calculate optimal workers per GPU based on VRAM capacity
    if not target_gpus:
        # CPU Mode
        cpu_workers = user_workers if (user_workers and user_workers > 0) else min(4, max(1, os.cpu_count() or 1))
        allocated_devices = ["cpu"] * cpu_workers
    else:
        if user_workers is not None and user_workers > 0:
            # User specified explicit total worker count: distribute across target GPUs
            for i in range(user_workers):
                gpu = target_gpus[i % len(target_gpus)]
                allocated_devices.append(f"cuda:{gpu['index']}")
        else:
            # Auto-calculate maximum safe concurrent workers per GPU
            for gpu in target_gpus:
                usable_vram = gpu["total_vram_gb"] * safety_margin
                max_w = max(1, int(usable_vram // model_vram_gb))
                # For very small models (e.g. RIFE), cap at 4-6 workers per GPU to avoid CPU decode bottlenecks
                max_w = min(6, max_w)
                for _ in range(max_w):
                    allocated_devices.append(f"cuda:{gpu['index']}")

    num_workers = len(allocated_devices)

    # 2. Calculate optimal chunk size
    # Target: ensure all workers are saturated (at least 2-3 chunks per worker for dynamic load balancing)
    if user_chunk_size is not None and user_chunk_size > 0:
        chunk_size = user_chunk_size
    else:
        # Ideal chunks: 2 to 3 chunks per worker
        target_chunks = max(1, num_workers * 2)
        raw_chunk_size = max(50, math.ceil(total_frames / target_chunks))

        # Clamp between bounds:
        # - Min chunk: 200 frames (~8s at 25fps) to keep seek/encode overhead < 2%
        # - Max chunk: 1500 frames (~60s) to keep progressive encoding and checkpoints responsive
        min_chunk = min(200, max(50, total_frames // num_workers))
        max_chunk = 1500

        chunk_size = max(min_chunk, min(max_chunk, raw_chunk_size))

        # 3. System RAM safety constraint
        # RAM used if all workers hold a full chunk in memory simultaneously:
        # num_workers * chunk_size * frame_ram_bytes <= 65% of available system RAM
        max_safe_ram_bytes = sys_ram_gb * (1024 ** 3) * 0.65
        ram_per_chunk_frame = num_workers * frame_ram_bytes
        if ram_per_chunk_frame > 0:
            ram_bounded_chunk = int(max_safe_ram_bytes // ram_per_chunk_frame)
            if ram_bounded_chunk < chunk_size:
                logger.info(
                    f"Throttling chunk_size from {chunk_size} to {ram_bounded_chunk} "
                    f"to respect System RAM budget ({sys_ram_gb:.1f} GB available)."
                )
                chunk_size = max(50, ram_bounded_chunk)

    estimated_chunks = math.ceil(total_frames / max(1, chunk_size))

    info = {
        "model_name": model_name,
        "model_vram_gb": round(model_vram_gb, 2),
        "total_frames": total_frames,
        "num_workers": num_workers,
        "devices": allocated_devices,
        "optimal_chunk_size": chunk_size,
        "estimated_chunks": estimated_chunks,
        "system_ram_gb": round(sys_ram_gb, 2),
        "gpus_detected": [
            f"{g['name']} ({g['total_vram_gb']}GB VRAM)" for g in gpu_list
        ] or ["None (CPU)"]
    }

    return allocated_devices, chunk_size, info
