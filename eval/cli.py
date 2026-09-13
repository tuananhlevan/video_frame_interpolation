"""Command Line Interface for the UPFRAME Evaluation System."""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

from upframe.pipeline.probe import probe_video
from upframe.utils.ffmpeg import format_duration
from eval.config import EvaluationConfig
from eval.pipeline import EvaluationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("eval.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.cli",
        description="UPFRAME Video Frame Interpolation Evaluation System"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. run command
    run_p = subparsers.add_parser("run", help="Evaluate an interpolated video against its source")
    run_p.add_argument("--source", "-s", required=True, help="Path to original 25 FPS source video")
    run_p.add_argument("--output", "-o", required=True, help="Path to interpolated 50 FPS video")
    run_p.add_argument("--model", "-m", default="vfi", help="Model name (e.g. rife, amt, film)")
    run_p.add_argument("--eval-dir", "-d", default="evaluation", help="Directory for evaluation artifacts")
    run_p.add_argument("--max-frames", type=int, default=None, help="Max source frames to evaluate (for quick runs)")
    run_p.add_argument("--sample-stride", type=int, default=1, help="Evaluation sample stride (1=all frames)")
    run_p.add_argument("--no-visuals", action="store_true", help="Disable generation of visual clips and images")
    run_p.add_argument("--gt", default=None, help="Optional true ground-truth video for full-reference metrics")
    run_p.add_argument("--human-mos-file", default=None, help="Optional JSON file with human MOS survey responses")
    run_p.add_argument("--processing-time", type=float, default=None, help="Production processing time in seconds (for RTF)")

    # 2. benchmark command
    bm_p = subparsers.add_parser("benchmark", help="Benchmark and compare multiple VFI models side-by-side")
    bm_p.add_argument("--source", "-s", required=True, help="Path to original 25 FPS source video")
    bm_p.add_argument(
        "--models",
        required=True,
        help="Comma-separated model outputs in format 'name1=path1,name2=path2' (e.g. 'rife=out_rife.mp4,amt=out_amt.mp4')"
    )
    bm_p.add_argument("--eval-dir", "-d", default="benchmark_evaluation", help="Directory for benchmark artifacts")
    bm_p.add_argument("--max-frames", type=int, default=None, help="Max source frames to evaluate")

    # 3. ground-truth command
    gt_p = subparsers.add_parser("ground-truth", help="Evaluate against high-FPS true ground truth footage")
    gt_p.add_argument("--gt", required=True, help="Path to high-FPS ground truth video")
    gt_p.add_argument("--output", "-o", required=True, help="Path to model interpolated video")
    gt_p.add_argument("--eval-dir", "-d", default="gt_evaluation", help="Directory for ground truth artifacts")
    gt_p.add_argument("--max-frames", type=int, default=None, help="Max frames to evaluate")

    # 4. inspect command
    ins_p = subparsers.add_parser("inspect", help="Generate visual inspection materials (slow-mo, injected frames, diff maps)")
    ins_p.add_argument("--source", "-s", required=True, help="Path to source video")
    ins_p.add_argument("--output", "-o", required=True, help="Path to interpolated video")
    ins_p.add_argument("--eval-dir", "-d", default="inspection_materials", help="Output directory for visual assets")
    ins_p.add_argument("--slowmo", default="2,4", help="Comma-separated slowmo multipliers (e.g. 2,4,8)")

    # 5. probe command
    probe_p = subparsers.add_parser("probe", help="Probe video stream and timing properties")
    probe_p.add_argument("video_path", help="Path to video file to inspect")

    return parser


def handle_run(args: argparse.Namespace) -> int:
    config = EvaluationConfig(
        eval_dir=args.eval_dir,
        max_frames=args.max_frames,
        sample_stride=args.sample_stride,
        generate_visuals=not args.no_visuals,
        ground_truth_path=args.gt,
        human_mos_file=args.human_mos_file
    )
    evaluator = EvaluationPipeline(config=config)
    report = evaluator.evaluate(
        source_path=args.source,
        output_path=args.output,
        eval_dir=args.eval_dir,
        model_name=args.model,
        processing_time_sec=args.processing_time
    )
    print("\n" + report.render_summary_text() + "\n")
    print(f"Full reports generated in: {os.path.abspath(args.eval_dir)}")
    print(f"  - HTML Dashboard: {os.path.join(args.eval_dir, 'report.html')}")
    print(f"  - JSON Report:    {os.path.join(args.eval_dir, 'report.json')}")
    print(f"  - Metrics CSV:    {os.path.join(args.eval_dir, 'metrics.csv')}")
    return 0 if report.technical_qc.passed else 1


def handle_benchmark(args: argparse.Namespace) -> int:
    models_dict: Dict[str, str] = {}
    for item in args.models.split(","):
        if "=" in item:
            name, path = item.split("=", 1)
            models_dict[name.strip()] = path.strip()

    if not models_dict:
        print("Error: No models specified. Format must be name1=path1,name2=path2")
        return 1

    print(f"\nComparing {len(models_dict)} models on source: {args.source}")
    reports = []
    for model_name, out_path in models_dict.items():
        print(f"\n>>> Evaluating candidate: {model_name.upper()} ({out_path})")
        model_eval_dir = os.path.join(args.eval_dir, model_name)
        config = EvaluationConfig(
            eval_dir=model_eval_dir,
            max_frames=args.max_frames,
            generate_visuals=True
        )
        evaluator = EvaluationPipeline(config=config)
        rep = evaluator.evaluate(
            source_path=args.source,
            output_path=out_path,
            eval_dir=model_eval_dir,
            model_name=model_name
        )
        reports.append(rep)

    # Render side-by-side comparison table
    print("\n" + "=" * 80)
    print("                    UPFRAME MODEL BENCHMARK COMPARISON")
    print("=" * 80)
    header = f"{'Metric':<25} " + " ".join([f"{r.model_name.upper():<16}" for r in reports])
    print(header)
    print("-" * 80)

    rows = [
        ("Football Quality Score", [f"{r.scorecard.quality_score:.1f} / 10.0" for r in reports]),
        ("Quality Status", [r.scorecard.quality_status for r in reports]),
        ("Technical Status", [r.scorecard.technical_status for r in reports]),
        ("Performance Score", [f"{r.scorecard.performance_score:.1f} / 10.0" for r in reports]),
        ("Realtime Factor", [f"{r.scorecard.realtime_factor:.2f}x" for r in reports]),
        ("Human MOS", [f"{r.scorecard.human_mos:.2f} / 5.0" for r in reports]),
        ("Ball Integrity", [f"{r.football_qc.ball_integrity_score:.1f} / 5.0" for r in reports]),
        ("Player Integrity", [f"{r.football_qc.player_integrity_score:.1f} / 5.0" for r in reports]),
        ("Occlusion Handling", [f"{r.football_qc.occlusion_handling_score:.1f} / 5.0" for r in reports]),
        ("Source Preservation PSNR", [f"{r.technical_qc.source_preservation_psnr:.2f} dB" for r in reports]),
        ("Added Ghosting", [f"+{r.artifact_diff.ghosting.added_level * 100:.2f}%" for r in reports]),
        ("Recommendation", [r.scorecard.recommendation.split()[0] for r in reports])
    ]

    for label, vals in rows:
        val_str = " ".join([f"{v:<16}" for v in vals])
        print(f"{label:<25} {val_str}")

    print("=" * 80)
    return 0


def handle_probe(args: argparse.Namespace) -> int:
    meta = probe_video(args.video_path)
    print("\n" + "=" * 50)
    print("                VIDEO STREAM PROBE")
    print("=" * 50)
    print(f"File:        {meta.filepath}")
    print(f"Resolution:  {meta.width}x{meta.height}")
    print(f"Nominal FPS: {meta.nominal_fps:.3f}")
    print(f"Average FPS: {meta.avg_fps:.3f}")
    print(f"Frame Count: {meta.nb_frames:,}")
    print(f"Duration:    {meta.duration:.2f}s ({format_duration(meta.duration)})")
    print(f"Codec:       {meta.codec_name} ({meta.pix_fmt})")
    print(f"Audio:       {'Present' if meta.has_audio else 'None'}")
    if meta.has_audio:
        for a in meta.audio_streams:
            print(f"  Stream #{a.index}: {a.codec_name}, {a.sample_rate} Hz, {a.channels} channels")
    print("=" * 50 + "\n")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        return handle_run(args)
    elif args.command == "benchmark":
        return handle_benchmark(args)
    elif args.command == "probe":
        return handle_probe(args)
    elif args.command == "ground-truth":
        args.model = "vfi"
        args.no_visuals = False
        args.sample_stride = 1
        return handle_run(args)
    elif args.command == "inspect":
        args.model = "vfi"
        args.no_visuals = False
        args.sample_stride = 1
        args.gt = None
        args.max_frames = 100
        return handle_run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
