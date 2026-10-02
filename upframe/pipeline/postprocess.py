"""Temporal Postprocess Deflicker for Video Frame Interpolation.

Fast GPU-accelerated postprocessing pass that eliminates 1-frame flickering
white/black firefly pixels, boundary overshoots, and unphysical synthesis
artifacts on any 50fps/60fps/120fps video in ~1 minute without re-running heavy models.
"""

import argparse
import logging
import os
import sys
from typing import Optional
import cv2
import numpy as np
import torch
from tqdm import tqdm

from upframe.pipeline.anti_flicker import TemporalAntiFlicker
from upframe.pipeline.encode import VideoEncoder
from upframe.pipeline.probe import probe_video

logger = logging.getLogger(__name__)


def deflicker_video(
    input_path: str,
    output_path: str,
    radius: int = 10,
    margin: int = 10,
    outlier_margin: int = 20,
    enable_cavity_healing: bool = True,
    cavity_kernel_size: int = 7,
    cavity_threshold: int = 18,
    cavity_min_luma: int = 70,
    mode: str = "odd",
    device: str = "cuda:0",
    use_nvenc: Optional[bool] = None,
    crf: int = 18,
    preset: str = "medium",
    batch_size: int = 16
) -> bool:
    """Streams video frames through temporal envelope outlier suppression."""
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return False

    try:
        meta = probe_video(input_path)
        width = meta.width
        height = meta.height
        fps = meta.nominal_fps or meta.avg_fps or 50.0
        total_frames = meta.nb_frames
        has_audio = meta.has_audio
        color_space = meta.color_space
        color_primaries = meta.color_primaries
        color_transfer = meta.color_transfer
        color_range = meta.color_range
    except Exception as e:
        logger.warning(f"probe_video failed ({e}), falling back to cv2.VideoCapture metadata.")
        cap_temp = cv2.VideoCapture(input_path)
        width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap_temp.get(cv2.CAP_PROP_FPS) or 50.0
        total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        has_audio = True
        color_space = None
        color_primaries = None
        color_transfer = None
        color_range = None
        cap_temp.release()

    logger.info(f"Deflickering {input_path}: {width}x{height} @ {fps:.2f}fps ({total_frames} frames)")
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    logger.info(f"Running deflicker filter on device: {dev} (radius={radius}, margin={margin}, outlier_margin={outlier_margin}, mode={mode})")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {input_path}")
        return False

    encoder = VideoEncoder(
        output_filepath=output_path,
        width=width,
        height=height,
        fps=fps,
        source_audio_path=input_path if has_audio else None,
        crf=crf,
        preset=preset,
        use_nvenc=use_nvenc,
        color_space=color_space,
        color_primaries=color_primaries,
        color_transfer=color_transfer,
        color_range=color_range
    )
    encoder.start()

    anti_flicker = TemporalAntiFlicker(
        radius=radius,
        margin=margin,
        outlier_margin=outlier_margin,
        enable_cavity_healing=enable_cavity_healing,
        cavity_kernel_size=cavity_kernel_size,
        cavity_threshold=cavity_threshold,
        cavity_min_luma=cavity_min_luma
    )

    pbar = tqdm(total=total_frames, desc="Deflickering", unit="frame")

    # Read first source frame (frame 0)
    ret, prev_frame_bgr = cap.read()
    if not ret:
        cap.release()
        encoder.finish()
        pbar.close()
        return False

    prev_frame = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2RGB)
    encoder.write_frame(prev_frame)
    pbar.update(1)

    batch_prev = []
    batch_curr = []
    batch_next = []

    ret, curr_frame_bgr = cap.read()
    while ret:
        curr_frame = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2RGB)
        ret_next, next_frame_bgr = cap.read()
        if not ret_next:
            # Trailing frame without a forward neighbor
            encoder.write_frame(curr_frame)
            pbar.update(1)
            break

        next_frame = cv2.cvtColor(next_frame_bgr, cv2.COLOR_BGR2RGB)

        batch_prev.append(prev_frame)
        batch_curr.append(curr_frame)
        batch_next.append(next_frame)

        if len(batch_curr) >= batch_size:
            t0 = torch.from_numpy(np.stack(batch_prev)).permute(0, 3, 1, 2).float().to(dev) / 255.0
            ti = torch.from_numpy(np.stack(batch_curr)).permute(0, 3, 1, 2).float().to(dev) / 255.0
            t1 = torch.from_numpy(np.stack(batch_next)).permute(0, 3, 1, 2).float().to(dev) / 255.0

            with torch.no_grad():
                clamped = anti_flicker.process_tensor(t0, ti, t1)
                cleaned_batch = (clamped.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            for b_idx in range(len(batch_curr)):
                encoder.write_frame(cleaned_batch[b_idx])
                encoder.write_frame(batch_next[b_idx])
                pbar.update(2)

            batch_prev.clear()
            batch_curr.clear()
            batch_next.clear()

        prev_frame = next_frame
        ret, curr_frame_bgr = cap.read()

    # Flush remaining batch
    if batch_curr:
        t0 = torch.from_numpy(np.stack(batch_prev)).permute(0, 3, 1, 2).float().to(dev) / 255.0
        ti = torch.from_numpy(np.stack(batch_curr)).permute(0, 3, 1, 2).float().to(dev) / 255.0
        t1 = torch.from_numpy(np.stack(batch_next)).permute(0, 3, 1, 2).float().to(dev) / 255.0

        with torch.no_grad():
            clamped = anti_flicker.process_tensor(t0, ti, t1)
            cleaned_batch = (clamped.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        for b_idx in range(len(batch_curr)):
            encoder.write_frame(cleaned_batch[b_idx])
            encoder.write_frame(batch_next[b_idx])
            pbar.update(2)

    cap.release()
    encoder.finish()
    pbar.close()

    logger.info(f"Deflicker completed successfully: {output_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="UpFrame Temporal Deflicker Postprocessor")
    parser.add_argument("-i", "--input", required=True, help="Path to input 50fps/60fps video with flickering artifacts")
    parser.add_argument("-o", "--output", required=True, help="Path to cleaned output video")
    parser.add_argument("-r", "--radius", type=int, default=16, help="Spatial search radius in pixels (default: 16)")
    parser.add_argument("-m", "--margin", type=int, default=10, help="Allowed luminance margin (default: 10)")
    parser.add_argument("--outlier-margin", type=int, default=20, help="Outlier rejection margin (default: 20)")
    parser.add_argument("--no-cavity-healing", action="store_true", help="Disable optical flow cavity/tear healing")
    parser.add_argument("--cavity-kernel-size", type=int, default=7, help="Cavity closing kernel size in pixels (default: 7)")
    parser.add_argument("--cavity-threshold", type=int, default=18, help="Cavity detection threshold (default: 18)")
    parser.add_argument("--cavity-min-luma", type=int, default=70, help="Minimum luma for bright structures (default: 70)")
    parser.add_argument("--device", default="cuda:0", help="CUDA device or cpu (default: cuda:0)")
    parser.add_argument("--no-nvenc", action="store_true", help="Disable NVENC hardware encoding")
    parser.add_argument("--crf", type=int, default=18, help="CRF quality (default: 18)")
    parser.add_argument("--preset", default="medium", help="Encoding preset (default: medium)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for GPU acceleration (default: 16)")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    success = deflicker_video(
        input_path=args.input,
        output_path=args.output,
        radius=args.radius,
        margin=args.margin,
        outlier_margin=args.outlier_margin,
        enable_cavity_healing=not args.no_cavity_healing,
        cavity_kernel_size=args.cavity_kernel_size,
        cavity_threshold=args.cavity_threshold,
        cavity_min_luma=args.cavity_min_luma,
        device=args.device,
        use_nvenc=False if args.no_nvenc else None,
        crf=args.crf,
        preset=args.preset,
        batch_size=args.batch_size
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
