import pytest
from unittest.mock import patch
from upframe.pipeline import resource_allocator

def test_single_gpu_allocation_5060ti():
    mock_gpu = [
        {'index': 0, 'name': 'NVIDIA GeForce RTX 5060 Ti', 'total_vram_gb': 16.0, 'free_vram_gb': 15.0}
    ]
    with patch.object(resource_allocator, 'get_gpu_info', return_value=mock_gpu):
        # 1. AMT-G (12.5GB) should get exactly 1 worker
        devs_amt, csize_amt, info_amt = resource_allocator.calculate_optimal_allocation("amt-g", total_frames=751)
        assert len(devs_amt) == 1
        assert devs_amt == ["cuda:0"]
        assert csize_amt > 100

        # 2. EMA-VFI (5.2GB) should get 2 workers on 16GB
        devs_ema, csize_ema, info_ema = resource_allocator.calculate_optimal_allocation("ema-vfi", total_frames=751)
        assert len(devs_ema) == 2
        assert devs_ema == ["cuda:0", "cuda:0"]

        # 3. RIFE (1.5GB) should get multiple workers (capped at 6)
        devs_rife, csize_rife, info_rife = resource_allocator.calculate_optimal_allocation("rife", total_frames=751)
        assert len(devs_rife) >= 4


def test_multi_gpu_allocation_3x_l40s():
    mock_3x_l40s = [
        {'index': 0, 'name': 'NVIDIA L40S', 'total_vram_gb': 48.0, 'free_vram_gb': 46.0},
        {'index': 1, 'name': 'NVIDIA L40S', 'total_vram_gb': 48.0, 'free_vram_gb': 46.0},
        {'index': 2, 'name': 'NVIDIA L40S', 'total_vram_gb': 48.0, 'free_vram_gb': 46.0},
    ]
    with patch.object(resource_allocator, 'get_gpu_info', return_value=mock_3x_l40s), \
         patch.object(resource_allocator, 'get_system_ram_gb', return_value=128.0):
        # AMT-G: 48GB / 12.5GB = 3 workers per GPU * 3 GPUs = 9 workers
        devs_amt, csize_amt, info_amt = resource_allocator.calculate_optimal_allocation("amt-g", total_frames=135000)
        assert len(devs_amt) == 9
        assert devs_amt.count("cuda:0") == 3
        assert devs_amt.count("cuda:1") == 3
        assert devs_amt.count("cuda:2") == 3
        assert csize_amt >= 200

        # EMA-VFI: 6 workers per GPU * 3 GPUs = 18 workers
        devs_ema, csize_ema, info_ema = resource_allocator.calculate_optimal_allocation("ema-vfi", total_frames=135000)
        assert len(devs_ema) == 18
        assert devs_ema.count("cuda:0") == 6
        assert devs_ema.count("cuda:1") == 6
        assert devs_ema.count("cuda:2") == 6


def test_user_override():
    mock_gpu = [
        {'index': 0, 'name': 'NVIDIA GeForce RTX 5060 Ti', 'total_vram_gb': 16.0, 'free_vram_gb': 15.0}
    ]
    with patch.object(resource_allocator, 'get_gpu_info', return_value=mock_gpu):
        # User explicitly demands workers=3 and chunk_size=500
        devs, csize, _ = resource_allocator.calculate_optimal_allocation(
            "rife", total_frames=1000, user_workers=3, user_chunk_size=500
        )
        assert len(devs) == 3
        assert csize == 500


def test_surplus_resource_optimization_a100():
    """Verify that A100 with 38GB free VRAM allocates in the 50-70% target range with batched inference."""
    mock_a100 = [
        {'index': 0, 'name': 'NVIDIA A100-SXM4-40GB', 'total_vram_gb': 40.0, 'free_vram_gb': 38.0}
    ]
    with patch.object(resource_allocator, 'get_gpu_info', return_value=mock_a100), \
         patch.object(resource_allocator, 'get_system_ram_gb', return_value=64.0), \
         patch("os.cpu_count", return_value=32):
        devs_rife, csize_rife, info_rife = resource_allocator.calculate_optimal_allocation(
            "rife", total_frames=13151, target_resource_ratio=0.65
        )
        # Should allocate multiple workers and batch_size safely capped
        assert len(devs_rife) >= 6
        assert 4 <= info_rife["optimal_batch_size"] <= 28
        # Target surplus VRAM usage must fall within 50% - 70% of free VRAM
        est_vram = info_rife["estimated_vram_gb"]
        pct_used = (est_vram / 38.0) * 100
        assert 50.0 <= pct_used <= 70.0, f"Expected 50-70% VRAM usage, got {pct_used:.1f}% ({est_vram}GB)"
        # Guaranteed safety headroom: must leave >= 12GB free on 40GB GPU
        assert (38.0 - est_vram) >= 12.0, f"Headroom too low: {38.0 - est_vram}GB"


def test_surplus_resource_optimization_rtx6000_blackwell_96gb():
    """Verify that RTX 6000 / Blackwell with 96GB VRAM allocates optimal workers (6) and safe batch size (16-24) with massive headroom."""
    mock_96gb = [
        {"index": 0, "name": "NVIDIA RTX 6000 Ada / Blackwell", "total_vram_gb": 96.0, "free_vram_gb": 92.0}
    ]
    with patch.object(resource_allocator, "get_gpu_info", return_value=mock_96gb), \
         patch.object(resource_allocator, "get_system_ram_gb", return_value=128.0), \
         patch("os.cpu_count", return_value=64):
        devs_rife, csize_rife, info_rife = resource_allocator.calculate_optimal_allocation(
            "rife", total_frames=13151, target_resource_ratio=0.65
        )
        # On 96GB with 64 CPU cores, workers are safely capped at 6 to prevent NVENC/CPU exhaustion
        assert len(devs_rife) >= 6, f"Workers did not reach 6 on 96GB: {len(devs_rife)}"
        # Batch size should be safely capped between 16 and 24 frames to avoid OOM
        assert 16 <= info_rife["optimal_batch_size"] <= 24, f"Batch size out of safe bounds [16, 24]: {info_rife['optimal_batch_size']}"
        est_vram = info_rife["estimated_vram_gb"]
        # Must retain at least 40GB free safety headroom
        assert (92.0 - est_vram) >= 40.0, f"Headroom too low: {92.0 - est_vram}GB"
