"""Optical Flow and Temporal Warping Consistency evaluation.

Computes dense optical flow between source frames A and B, predicts
the intermediate frame X_pred via motion-compensated warping at t=0.5,
and measures warping error and motion-boundary error against generated X.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


_dis_instance = None
_grid_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}


def _get_dis_flow():
    global _dis_instance
    if _dis_instance is None:
        try:
            _dis_instance = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        except Exception:
            _dis_instance = False
    return _dis_instance if _dis_instance is not False else None


def _get_cached_grid(w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    key = (w, h)
    if key not in _grid_cache:
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        _grid_cache[key] = (grid_x, grid_y)
    return _grid_cache[key]


def compute_dense_flow(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Computes dense optical flow from img_a to img_b using OpenCV DIS (or Farneback fallback).
    
    Returns:
        flow: [H, W, 2] in pixels
    """
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if img_a.ndim == 3 else img_a
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) if img_b.ndim == 3 else img_b

    dis = _get_dis_flow()
    if dis is not None:
        try:
            return dis.calc(gray_a, gray_b, None)
        except Exception:
            pass

    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow


def warp_frame(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warps an image using backward mapping optical flow [H, W, 2]."""
    h, w = img.shape[:2]
    grid_x, grid_y = _get_cached_grid(w, h)
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped


def compute_motion_boundary_mask(flow: np.ndarray, percentile: float = 85.0) -> np.ndarray:
    """Extracts binary mask of motion boundaries (high flow gradients)."""
    # Flow gradient magnitude
    grad_ux = cv2.Sobel(flow[..., 0], cv2.CV_32F, 1, 0)
    grad_uy = cv2.Sobel(flow[..., 0], cv2.CV_32F, 0, 1)
    grad_vx = cv2.Sobel(flow[..., 1], cv2.CV_32F, 1, 0)
    grad_vy = cv2.Sobel(flow[..., 1], cv2.CV_32F, 0, 1)

    mag = np.sqrt(grad_ux ** 2 + grad_uy ** 2) + np.sqrt(grad_vx ** 2 + grad_vy ** 2)
    thresh = np.percentile(mag, percentile)
    return (mag >= thresh).astype(np.uint8)


def evaluate_optical_flow_consistency(
    frame_a: np.ndarray,
    frame_generated: np.ndarray,
    frame_b: np.ndarray
) -> Dict[str, float]:
    """Evaluates how consistently frame_generated interpolates between frame_a and frame_b.
    
    Returns:
        metrics: {
            "mean_warping_error": float,
            "p95_warping_error": float,
            "motion_boundary_error": float,
            "motion_magnitude": float
        }
    """
    # Forward flow A -> B and backward flow B -> A
    flow_ab = compute_dense_flow(frame_a, frame_b)
    flow_ba = compute_dense_flow(frame_b, frame_a)
    motion_mag = float(np.mean(np.sqrt(flow_ab[..., 0] ** 2 + flow_ab[..., 1] ** 2)))

    # Predict intermediate frame:
    # In backward mapping (cv2.remap), mapping coordinate map(x) = x - 0.5 * flow
    # shifts source content forward along the motion trajectory.
    pred_x_from_a = warp_frame(frame_a, -flow_ab * 0.5)
    pred_x_from_b = warp_frame(frame_b, -flow_ba * 0.5)

    # Occlusion & Disocclusion handling:
    # Forward-backward flow consistency check: in non-occluded regions, flow_ab + flow_ba approx 0
    fb_diff = np.sqrt((flow_ab[..., 0] + flow_ba[..., 0]) ** 2 + (flow_ab[..., 1] + flow_ba[..., 1]) ** 2)
    occlusion_mask = (fb_diff > 3.0)

    res_a = np.abs(frame_generated.astype(np.float32) - pred_x_from_a.astype(np.float32))
    res_b = np.abs(frame_generated.astype(np.float32) - pred_x_from_b.astype(np.float32))

    # In occluded regions, choose the best matching visible frame to prevent artificial ghosting penalties
    pred_x = np.where(
        occlusion_mask[..., np.newaxis],
        np.where(res_a < res_b, pred_x_from_a, pred_x_from_b),
        pred_x_from_a.astype(np.float32) * 0.5 + pred_x_from_b.astype(np.float32) * 0.5
    )

    # Pixel residual error
    diff = np.abs(frame_generated.astype(np.float32) - pred_x)
    mean_err = float(np.mean(diff))
    p95_err = float(np.percentile(diff, 95))

    # Motion boundary error
    boundary_mask = compute_motion_boundary_mask(flow_ab)
    if np.sum(boundary_mask) > 0:
        mask_3d = np.repeat(boundary_mask[:, :, np.newaxis], 3, axis=2)
        mb_err = float(np.mean(diff[mask_3d > 0]))
    else:
        mb_err = mean_err

    return {
        "mean_warping_error": mean_err,
        "p95_warping_error": p95_err,
        "motion_boundary_error": mb_err,
        "motion_magnitude": motion_mag
    }
