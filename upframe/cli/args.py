"""Argument parsing and configuration resolution for the upframe CLI."""

import argparse
import os
from typing import List, Optional, Tuple
from upframe.core.config import PipelineConfig


def parse_gpus(gpu_arg: Optional[str]) -> Optional[List[int]]:
    """Parses a comma-separated list of GPU indices."""
    if not gpu_arg or gpu_arg.lower() in ["cpu", "none"]:
        return []
    try:
        return [int(x.strip()) for x in gpu_arg.split(",") if x.strip()]
    except ValueError:
        return None


def build_parser() -> argparse.ArgumentParser:
    """Constructs the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="upframe",
        description="Production-grade video frame insertion (VFI) pipeline for 25 -> 50 fps upframing."
    )
    # Positional or named input/output
    parser.add_argument("input_pos", nargs="?", help="Input video file path (e.g. match.mp4)")
    parser.add_argument("output_pos", nargs="?", help="Output video file path (e.g. match_50fps.mp4)")
    parser.add_argument("-i", "--input", dest="input_flag", help="Input video file path")
    parser.add_argument("-o", "--output", dest="output_flag", help="Output video file path")

    parser.add_argument("-m", "--model", default="rife",
                        choices=["rife", "blend", "amt", "amt-s", "amt-l", "amt-g", "film", "ifrnet"],
                        help="VFI model to use (default: rife)")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU indices (e.g. '0,1,2') or 'cpu'")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Frames per processing chunk (default: 1000)")
    parser.add_argument("--scene-threshold", type=float, default=0.35,
                        help="Scene cut sensitivity threshold [0.0 - 1.0] (default: 0.35)")
    parser.add_argument("--weights", default=None, help="Path to pretrained model checkpoint")
    parser.add_argument("--crf", type=int, default=18, help="FFmpeg H.264 CRF quality level (default: 18)")
    parser.add_argument("--preset", default="medium", help="FFmpeg encoding preset (default: medium)")
    parser.add_argument("--nvenc", dest="use_nvenc", action="store_true", default=None,
                        help="Force NVIDIA NVENC hardware encoding")
    parser.add_argument("--no-nvenc", dest="use_nvenc", action="store_false",
                        help="Disable NVENC hardware encoding (use libx264)")
    parser.add_argument("--temp-dir", default=None, help="Directory for intermediate chunk files")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted job using existing chunks")
    parser.add_argument("--config", default=None, help="Path to YAML configuration file")
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=True, help="Enable FP16 inference")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable FP16 inference")
    parser.add_argument("--log-dir", default="upframe_log", help="Directory for execution log files (default: upframe_log)")
    parser.add_argument("--log-file", default=None, help="Path to write execution log file (default: upframe_log/<output_name>.log)")

    return parser


def resolve_log_folder(base_log_dir: str, input_path: str, output_path: str, model_name: str) -> str:
    """Computes run-specific log directory: <base_log_dir>/<video_name>_<model>."""
    input_stem = os.path.splitext(os.path.basename(input_path))[0]
    output_stem = os.path.splitext(os.path.basename(output_path))[0]

    if input_stem.lower() in ("input", "in", "video") and output_stem.lower() not in ("output", "out"):
        chosen_stem = output_stem
    else:
        chosen_stem = input_stem

    model_suffix = model_name.lower().replace("-", "_")
    if chosen_stem.lower().endswith(f"_{model_suffix}"):
        folder_name = chosen_stem
    else:
        folder_name = f"{chosen_stem}_{model_suffix}"

    if os.path.basename(os.path.normpath(base_log_dir)) == folder_name:
        return base_log_dir
    return os.path.join(base_log_dir, folder_name)


def parse_cli_args(args_list: Optional[List[str]] = None) -> Tuple[str, str, PipelineConfig]:
    """Parses CLI arguments, merges with YAML config if supplied, and returns (input, output, PipelineConfig)."""
    parser = build_parser()
    args = parser.parse_args(args_list)

    input_path = args.input_flag or args.input_pos
    output_path = args.output_flag or args.output_pos

    if not input_path or not output_path:
        parser.print_help()
        raise ValueError("Both input and output paths must be specified.")

    config = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()

    # CLI overrides
    if args.model != "rife" or not args.config:
        config.model = args.model
    if args.chunk_size != 1000 or not args.config:
        config.chunk_size = args.chunk_size
    if args.scene_threshold != 0.35 or not args.config:
        config.scene_threshold = args.scene_threshold
    if args.crf != 18 or not args.config:
        config.crf = args.crf
    if args.preset != "medium" or not args.config:
        config.preset = args.preset
    if args.use_nvenc is not None:
        config.use_nvenc = args.use_nvenc
    if args.weights:
        config.weights = args.weights
    if args.temp_dir:
        config.temp_dir = args.temp_dir
    if args.resume:
        config.resume = args.resume
    if args.gpus:
        config.gpus = parse_gpus(args.gpus)
    config.fp16 = args.fp16
    if args.log_dir != "upframe_log" or not args.config:
        config.log_dir = args.log_dir

    base_log_dir = config.log_dir or "upframe_log"
    config.log_dir = resolve_log_folder(base_log_dir, input_path, output_path, config.model)
    folder_name = os.path.basename(config.log_dir)

    if args.log_file:
        config.log_file = args.log_file
    elif not config.log_file:
        config.log_file = os.path.join(config.log_dir, f"{folder_name}.log")

    return input_path, output_path, config
