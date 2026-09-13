"""Scientific Ground-Truth Benchmark for high-frame-rate football footage.

When genuine high-FPS footage (e.g. 50/60/100 fps) is available, extracts
the missing ground-truth frames and evaluates true PSNR, SSIM, MS-SSIM,
MAE, and LPIPS against the model's generated intermediate frames.
"""

from typing import Dict, List, Optional
import cv2
import numpy as np
from upframe.pipeline.decode import VideoDecoder
from upframe.pipeline.probe import probe_video
from eval.layer1_technical.source_preservation import compute_ms_ssim, compute_psnr, compute_ssim
from eval.types import GroundTruthQCResult


def compute_lpips_if_available(img1: np.ndarray, img2: np.ndarray) -> Optional[float]:
    """Computes LPIPS distance if the lpips package and PyTorch are available."""
    try:
        import lpips
        import torch
        # Convert BGR/RGB [H, W, 3] uint8 to RGB normalized [-1, 1] tensor [1, 3, H, W]
        t1 = torch.from_numpy(img1.copy()).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        t2 = torch.from_numpy(img2.copy()).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        loss_fn = lpips.LPIPS(net="alex", verbose=False)
        with torch.no_grad():
            dist = float(loss_fn(t1, t2).item())
        return dist
    except Exception:
        return None


def evaluate_against_ground_truth(
    gt_video_path: str,
    interpolated_video_path: str,
    max_frames: Optional[int] = None
) -> GroundTruthQCResult:
    """Compares generated intermediate frames (odd frames) with true ground-truth frames.
    
    Returns:
        GroundTruthQCResult
    """
    gt_meta = probe_video(gt_video_path)
    vfi_meta = probe_video(interpolated_video_path)

    dec_gt = VideoDecoder(gt_video_path, gt_meta.width, gt_meta.height)
    dec_vfi = VideoDecoder(interpolated_video_path, vfi_meta.width, vfi_meta.height)

    psnr_scores: List[float] = []
    ssim_scores: List[float] = []
    ms_ssim_scores: List[float] = []
    lpips_scores: List[float] = []
    mae_scores: List[float] = []
    mse_scores: List[float] = []

    stream_gt = dec_gt.stream_frames()
    stream_vfi = dec_vfi.stream_frames()
    frame_idx = 0

    try:
        while True:
            try:
                frame_gt = next(stream_gt)
                frame_vfi = next(stream_vfi)
            except StopIteration:
                break

            # Evaluate odd frames (the generated intermediate frames)
            if frame_idx % 2 == 1:
                if frame_gt.shape != frame_vfi.shape:
                    frame_vfi = cv2.resize(frame_vfi, (frame_gt.shape[1], frame_gt.shape[0]))

                p = compute_psnr(frame_gt, frame_vfi)
                s = compute_ssim(frame_gt, frame_vfi)
                ms = compute_ms_ssim(frame_gt, frame_vfi)
                mae = float(np.mean(np.abs(frame_gt.astype(np.float32) - frame_vfi.astype(np.float32))))
                mse = float(np.mean((frame_gt.astype(np.float32) - frame_vfi.astype(np.float32)) ** 2))

                psnr_scores.append(p)
                ssim_scores.append(s)
                ms_ssim_scores.append(ms)
                mae_scores.append(mae)
                mse_scores.append(mse)

                # Optional LPIPS on a subset of frames
                if len(lpips_scores) < 20:
                    lp = compute_lpips_if_available(frame_gt, frame_vfi)
                    if lp is not None:
                        lpips_scores.append(lp)

                if max_frames and len(psnr_scores) >= max_frames:
                    break

            frame_idx += 1
    except Exception:
        pass

    return GroundTruthQCResult(
        psnr=round(float(np.mean(psnr_scores)), 3) if psnr_scores else 0.0,
        ssim=round(float(np.mean(ssim_scores)), 4) if ssim_scores else 0.0,
        ms_ssim=round(float(np.mean(ms_ssim_scores)), 4) if ms_ssim_scores else None,
        lpips=round(float(np.mean(lpips_scores)), 4) if lpips_scores else None,
        mae=round(float(np.mean(mae_scores)), 3) if mae_scores else 0.0,
        mse=round(float(np.mean(mse_scores)), 3) if mse_scores else 0.0
    )
