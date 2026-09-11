"""PyTorch tensor and NumPy frame transformation utilities."""

from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


def frame_to_tensor(
    frame: np.ndarray,
    device: torch.device,
    half: bool = False
) -> torch.Tensor:
    """Converts a (H, W, 3) uint8 or float [0, 255] NumPy frame to a (1, 3, H, W) PyTorch tensor [0, 1]."""
    if not frame.flags.writeable:
        frame = frame.copy()

    if frame.ndim == 3:
        # (H, W, C) -> (C, H, W)
        t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
    elif frame.ndim == 4:
        # (B, H, W, C) -> (B, C, H, W)
        t = torch.from_numpy(frame).permute(0, 3, 1, 2).float()
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")

    if frame.dtype == np.uint8 or frame.max() > 1.0:
        t = t / 255.0

    t = t.to(device)
    if half:
        t = t.half()
    return t


def tensor_to_frame(tensor: torch.Tensor) -> np.ndarray:
    """Converts a (1, 3, H, W) or (3, H, W) PyTorch tensor [0, 1] to a (H, W, 3) uint8 NumPy frame."""
    t = tensor.detach().cpu().float()
    if t.ndim == 4:
        t = t.squeeze(0)
    frame = t.permute(1, 2, 0).numpy()
    frame = np.nan_to_num(frame, nan=0.0)
    frame = np.clip(frame, 0.0, 1.0) * 255.0
    return np.round(frame).astype(np.uint8)


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int = 32,
    mode: str = "replicate"
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pads tensor (B, C, H, W) height and width to multiples of `multiple`.
    
    Returns padded tensor and (pad_h, pad_w).
    """
    _, _, h, w = tensor.shape
    ph = ((h - 1) // multiple + 1) * multiple
    pw = ((w - 1) // multiple + 1) * multiple
    pad_h = ph - h
    pad_w = pw - w

    if pad_h > 0 or pad_w > 0:
        padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
        return padded, (pad_h, pad_w)
    return tensor, (0, 0)


def unpad(tensor: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
    """Crops tensor back to original (orig_h, orig_w) dimensions."""
    return tensor[:, :, :orig_h, :orig_w]


def warp(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    """Backward optical flow warping via F.grid_sample."""
    flow_device = tenFlow.device
    flow_dtype = tenInput.dtype
    B, C, H, W = tenInput.shape

    horizontal = torch.linspace(-1.0, 1.0, W, device=flow_device, dtype=flow_dtype).view(1, 1, 1, W).expand(B, -1, H, -1)
    vertical = torch.linspace(-1.0, 1.0, H, device=flow_device, dtype=flow_dtype).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([horizontal, vertical], 1)

    norm_flow = torch.cat([
        tenFlow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
        tenFlow[:, 1:2, :, :] / ((H - 1.0) / 2.0)
    ], 1).to(flow_dtype)

    g = (grid + norm_flow).permute(0, 2, 3, 1).to(flow_dtype)
    return F.grid_sample(tenInput, g, mode='bilinear', padding_mode='border', align_corners=True)
