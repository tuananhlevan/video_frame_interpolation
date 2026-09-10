"""Scene cut and shot-boundary detection to prevent hybrid frame synthesis across camera cuts."""

from typing import List, Union
import numpy as np
import torch


def compute_frame_difference(
    frame_a: Union[np.ndarray, torch.Tensor],
    frame_b: Union[np.ndarray, torch.Tensor],
    downsample_size: int = 64
) -> float:
    """Computes a normalized difference metric [0.0, 1.0] between consecutive frames.
    
    Uses combined downsampled luminance L1 delta + color histogram correlation.
    """
    if isinstance(frame_a, torch.Tensor):
        # (C, H, W) in [0, 1]
        a = frame_a.detach().cpu().float().numpy()
        b = frame_b.detach().cpu().float().numpy()
        if a.ndim == 4:
            a = a.squeeze(0)
            b = b.squeeze(0)
        # Convert (C, H, W) to (H, W, C)
        a = np.transpose(a, (1, 2, 0))
        b = np.transpose(b, (1, 2, 0))
        if a.max() <= 1.0:
            a = (a * 255.0).astype(np.uint8)
            b = (b * 255.0).astype(np.uint8)
    else:
        a = frame_a
        b = frame_b
        if a.dtype != np.uint8 and a.max() <= 1.0:
            a = (a * 255.0).astype(np.uint8)
            b = (b * 255.0).astype(np.uint8)

    # Convert to grayscale / downsampled grid
    # Fast decimation
    h, w = a.shape[:2]
    step_y = max(1, h // downsample_size)
    step_x = max(1, w // downsample_size)
    
    sub_a = a[::step_y, ::step_x, :3].astype(np.float32)
    sub_b = b[::step_y, ::step_x, :3].astype(np.float32)

    # Pixel absolute difference normalized
    l1_diff = np.mean(np.abs(sub_a - sub_b)) / 255.0

    # Color histogram comparison (32 bins per channel)
    hist_diff = 0.0
    for c in range(3):
        ha, _ = np.histogram(sub_a[..., c], bins=32, range=(0, 256))
        hb, _ = np.histogram(sub_b[..., c], bins=32, range=(0, 256))
        sum_a = ha.sum()
        sum_b = hb.sum()
        norm_ha = ha.astype(np.float32) / (sum_a if sum_a > 0 else 1.0)
        norm_hb = hb.astype(np.float32) / (sum_b if sum_b > 0 else 1.0)
        intersection = float(np.sum(np.minimum(norm_ha, norm_hb)))
        hist_diff += (1.0 - intersection)
    hist_diff /= 3.0

    # Combined score
    score = 0.6 * l1_diff + 0.4 * hist_diff
    return float(score)


def is_scene_cut(
    frame_a: Union[np.ndarray, torch.Tensor],
    frame_b: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.35
) -> bool:
    """Returns True if the transition between frame_a and frame_b is a hard scene cut."""
    diff = compute_frame_difference(frame_a, frame_b)
    return diff >= threshold


def detect_cuts(
    frames: List[Union[np.ndarray, torch.Tensor]],
    threshold: float = 0.35
) -> List[int]:
    """Scans a sequence of consecutive frames and returns indices i where frames[i]->frames[i+1] is a cut."""
    cuts: List[int] = []
    for i in range(len(frames) - 1):
        if is_scene_cut(frames[i], frames[i + 1], threshold=threshold):
            cuts.append(i)
    return cuts
