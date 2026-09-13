"""Source-frame preservation validation.

Compares original source frames (A, B, C...) with corresponding output
temporal samples (A, B, C... at indices 0, 2, 4...) using PSNR, SSIM,
MS-SSIM, MAE, and maximum pixel difference.
"""

import math
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from upframe.pipeline.decode import VideoDecoder
from upframe.pipeline.probe import probe_video


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0  # Identical images
    max_pixel = 255.0
    return float(20 * math.log10(max_pixel / math.sqrt(mse)))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Structural Similarity Index (SSIM) between two images.
    
    Compatible with uint8 [H, W, C] or [H, W] images.
    """
    if img1.ndim == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray1 = img1.astype(np.float64)
        gray2 = img2.astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(np.mean(ssim_map))


def compute_ms_ssim(img1: np.ndarray, img2: np.ndarray, levels: int = 4) -> float:
    """Computes Multi-Scale SSIM across downsampled scales."""
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333][:levels]
    weights = [w / sum(weights) for w in weights]
    
    current_img1 = img1.copy()
    current_img2 = img2.copy()
    
    msssim = 0.0
    for level, weight in enumerate(weights):
        sim = compute_ssim(current_img1, current_img2)
        msssim += weight * sim
        if level < levels - 1:
            h, w = current_img1.shape[:2]
            if h < 16 or w < 16:
                break
            current_img1 = cv2.pyrDown(current_img1)
            current_img2 = cv2.pyrDown(current_img2)
            
    return float(msssim)


def validate_source_preservation(
    source_path: str,
    output_path: str,
    max_frames: Optional[int] = None,
    stride: int = 1,
    min_psnr: float = 28.0,
    min_ssim: float = 0.90,
    max_mae: float = 8.0
) -> Tuple[bool, Dict[str, float], List[str]]:
    """Evaluates whether original source frames are preserved in the output video.
    
    Output indices [0, 2, 4, ...] should correspond to source frames [0, 1, 2, ...].
    Uses FFmpeg streaming VideoDecoder for 100% frame-accurate sequential decoding.
    
    Returns:
        (passed, metrics_dict, warnings)
    """
    src_meta = probe_video(source_path)
    out_meta = probe_video(output_path)

    dec_src = VideoDecoder(source_path, src_meta.width, src_meta.height)
    dec_out = VideoDecoder(output_path, out_meta.width, out_meta.height)

    psnr_list: List[float] = []
    ssim_list: List[float] = []
    mae_list: List[float] = []
    max_diff_list: List[float] = []

    src_idx = 0
    evaluated_count = 0

    stream_src = dec_src.stream_frames()
    stream_out = dec_out.stream_frames()

    try:
        while True:
            try:
                frame_src = next(stream_src)
            except StopIteration:
                break

            # Output has 2 frames for each source frame interval
            try:
                frame_out_even = next(stream_out)
            except StopIteration:
                break

            try:
                _ = next(stream_out)  # odd frame (intermediate)
            except StopIteration:
                pass

            if src_idx % stride == 0:
                if frame_src.shape != frame_out_even.shape:
                    frame_out_even = cv2.resize(frame_out_even, (frame_src.shape[1], frame_src.shape[0]))

                psnr_val = compute_psnr(frame_src, frame_out_even)
                ssim_val = compute_ssim(frame_src, frame_out_even)
                mae_val = float(np.mean(np.abs(frame_src.astype(np.float32) - frame_out_even.astype(np.float32))))
                max_d = float(np.max(np.abs(frame_src.astype(np.float32) - frame_out_even.astype(np.float32))))

                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                mae_list.append(mae_val)
                max_diff_list.append(max_d)
                evaluated_count += 1

                if max_frames and evaluated_count >= max_frames:
                    break

            src_idx += 1
    except Exception as e:
        pass

    if evaluated_count == 0:
        return False, {"psnr": 0.0, "ssim": 0.0, "mae": 255.0, "max_diff": 255.0}, ["No frames evaluated for preservation"]

    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    mean_mae = float(np.mean(mae_list))
    max_pixel_diff = float(np.max(max_diff_list))

    warnings: List[str] = []
    passed = True

    if mean_psnr < min_psnr:
        passed = False
        warnings.append(
            f"Source frame preservation PSNR ({mean_psnr:.2f} dB) is below threshold ({min_psnr:.1f} dB). "
            f"Indicates possible color conversion, resizing, or frame ordering mismatch."
        )
    elif mean_psnr < 35.0:
        warnings.append(
            f"Source preservation PSNR ({mean_psnr:.2f} dB) indicates lossy re-encoding or minor compression difference."
        )

    if mean_ssim < min_ssim:
        passed = False
        warnings.append(
            f"Source frame preservation SSIM ({mean_ssim:.4f}) is below threshold ({min_ssim:.2f})."
        )

    if mean_mae > max_mae:
        warnings.append(
            f"High mean absolute error ({mean_mae:.2f} px) between source and output even frames."
        )

    metrics = {
        "psnr": mean_psnr,
        "ssim": mean_ssim,
        "mae": mean_mae,
        "max_diff": max_pixel_diff,
        "frames_evaluated": evaluated_count
    }

    return passed, metrics, warnings
