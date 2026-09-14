"""Command Line Interface for the upframe video frame insertion system."""

import logging
import os
import sys
from upframe.cli.args import parse_cli_args
from upframe.pipeline.probe import probe_video
from upframe.pipeline.scheduler import PipelineScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("upframe")


def main() -> int:
    try:
        input_path, output_path, config = parse_cli_args()
    except ValueError as e:
        print(f"\nError: {e}")
        return 1

    if config.log_file:
        log_dir = os.path.dirname(os.path.abspath(config.log_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(config.log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Execution log file: {config.log_file}")

    if not os.path.exists(input_path):
        logger.error(f"Input file does not exist: {input_path}")
        return 1

    logger.info(f"Probing input video: {input_path}")
    metadata = probe_video(input_path)

    logger.info(
        f"Input: {metadata.width}x{metadata.height} @ ~{metadata.nominal_fps:.2f} fps, "
        f"Duration: {metadata.duration:.2f}s, Frames: {metadata.nb_frames:,}, "
        f"Audio: {'Yes' if metadata.has_audio else 'No'}"
    )

    scheduler = PipelineScheduler(
        metadata=metadata,
        output_filepath=output_path,
        model_name=config.model,
        gpus=config.gpus,
        chunk_size=config.chunk_size,
        scene_threshold=config.scene_threshold,
        temp_dir=config.temp_dir,
        checkpoint_path=config.weights,
        crf=config.crf,
        preset=config.preset,
        use_nvenc=config.use_nvenc,
        resume=config.resume,
        fp16=config.fp16,
        max_retries=config.max_retries,
        workers=config.workers,
        log_dir=config.log_dir
    )

    try:
        report = scheduler.run()
        report_text = report.render_text()
        print("\n" + report_text + "\n")
        logger.info(f"Final Report:\n{report_text}")
        logger.info(f"Upframing completed successfully! Output: {output_path}")
        return 0
    except Exception as e:
        logger.exception(f"Upframing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
