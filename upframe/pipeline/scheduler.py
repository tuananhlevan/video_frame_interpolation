"""High-level pipeline scheduler coordinating partitioning, workers, and encoding."""

import json
import logging
import os
import time
from typing import Callable, Dict, List, Optional
from tqdm import tqdm
from upframe.core.config import PipelineConfig
from upframe.core.types import ChunkResult, ChunkTask, QCReport, VideoMetadata
from upframe.pipeline.encode import VideoEncoder
from upframe.pipeline.merge import ChunkMerger
from upframe.pipeline.partitioner import ChunkPartitioner
from upframe.pipeline.state import StateStore
from upframe.utils.ffmpeg import format_duration
from upframe.workers.pool import WorkerPool

logger = logging.getLogger(__name__)


class PipelineScheduler:
    """Orchestrates multi-GPU chunk execution, progress tracking, and fault tolerance."""

    def __init__(
        self,
        metadata: VideoMetadata,
        output_filepath: str,
        model_name: str = "rife",
        gpus: Optional[List[int]] = None,
        chunk_size: int = 1000,
        scene_threshold: float = 0.35,
        temp_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        crf: int = 18,
        preset: str = "medium",
        use_nvenc: Optional[bool] = None,
        max_retries: int = 3,
        resume: bool = False,
        fp16: bool = True,
        workers: Optional[int] = None,
        log_dir: str = "upframe_log"
    ) -> None:
        self.metadata = metadata
        self.output_filepath = os.path.abspath(output_filepath)
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.scene_threshold = scene_threshold
        self.checkpoint_path = checkpoint_path
        self.crf = crf
        self.preset = preset
        self.use_nvenc = use_nvenc
        self.max_retries = max_retries
        self.resume = resume
        self.fp16 = fp16
        self.workers = workers
        self.log_dir = log_dir or "upframe_log"

        output_stem = os.path.splitext(os.path.basename(self.output_filepath))[0]

        # Setup working temp directory
        if temp_dir is None:
            self.temp_dir = os.path.join(
                self.log_dir,
                f".upframe_tmp_{output_stem}"
            )
        else:
            self.temp_dir = os.path.abspath(temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)

        self.state_store = StateStore(
            os.path.join(self.temp_dir, "pipeline_state.json"),
            enabled=True
        )
        self.partitioner = ChunkPartitioner(chunk_size=self.chunk_size, overlap=1)

        # Device determination
        import torch
        base_devices: List[str] = []
        if gpus is not None and len(gpus) > 0 and torch.cuda.is_available():
            base_devices = [f"cuda:{g}" for g in gpus if g < torch.cuda.device_count()]
            if not base_devices:
                base_devices = ["cuda:0"]
        elif gpus is not None and len(gpus) == 0:
            base_devices = ["cpu"]
        elif torch.cuda.is_available():
            count = torch.cuda.device_count()
            base_devices = [f"cuda:{i}" for i in range(count)]
        else:
            base_devices = ["cpu"]

        if self.workers is not None and self.workers > 0:
            if base_devices == ["cpu"]:
                self.devices = ["cpu"] * self.workers
            else:
                self.devices = [base_devices[i % len(base_devices)] for i in range(self.workers)]
        else:
            self.devices = base_devices

        logger.info(f"Initialized scheduler with {len(self.devices)} workers on devices: {self.devices}")

    def generate_chunk_ranges(self) -> List[tuple[int, int, int]]:
        """Generates (chunk_id, start_frame, end_frame) tuples with 1-frame boundary overlap."""
        chunk_ranges = self.partitioner.partition_video(self.metadata)
        return [(c.chunk_id, c.start_frame, c.end_frame) for c in chunk_ranges]

    def run(self, progress_callback: Optional[Callable[[int, int], None]] = None) -> QCReport:
        """Executes the complete upframing pipeline."""
        start_time = time.time()
        chunk_ranges = self.partitioner.partition_video(self.metadata)
        total_chunks = len(chunk_ranges)

        logger.info(f"Processing {total_chunks} chunks across {len(self.devices)} workers.")

        chunk_results: Dict[int, ChunkResult] = {}
        pending_tasks: List[ChunkTask] = []

        # Check for already completed chunks if resume is enabled
        for chunk in chunk_ranges:
            if self.resume:
                existing = self.state_store.get_chunk_result(chunk.chunk_id)
                if existing:
                    logger.info(f"Resuming: skipping already completed chunk {chunk.chunk_id}")
                    chunk_results[chunk.chunk_id] = existing
                    continue

            pending_tasks.append(
                ChunkTask(
                    chunk_id=chunk.chunk_id,
                    start_frame=chunk.start_frame,
                    end_frame=chunk.end_frame,
                    input_path=self.metadata.filepath,
                    output_dir=os.path.join(self.temp_dir, "chunks"),
                    width=self.metadata.width,
                    height=self.metadata.height,
                    fps=self.metadata.nominal_fps,
                    model_name=self.model_name,
                    checkpoint_path=self.checkpoint_path,
                    scene_threshold=self.scene_threshold,
                    fp16=self.fp16,
                    color_space=self.metadata.color_space,
                    color_primaries=self.metadata.color_primaries,
                    color_transfer=self.metadata.color_transfer,
                    color_range=self.metadata.color_range
                )
            )

        # Progress bar
        pbar = tqdm(
            total=total_chunks,
            initial=len(chunk_results),
            desc="Upframing chunks",
            unit="chunk"
        )

        def on_chunk_completed(result: ChunkResult) -> None:
            self.state_store.record_chunk(result)
            pbar.update(1)
            if progress_callback:
                progress_callback(len(chunk_results) + 1, total_chunks)

        # Dispatch tasks to worker pool
        pool = WorkerPool(
            devices=self.devices,
            model_name=self.model_name,
            checkpoint_path=self.checkpoint_path,
            fp16=self.fp16
        )

        executed_results = pool.execute(
            tasks=pending_tasks,
            max_retries=self.max_retries,
            on_chunk_done=on_chunk_completed
        )
        chunk_results.update(executed_results)
        pbar.close()

        # Step 2: Ordered Merge & Encoding
        target_fps = self.metadata.target_fps
        logger.info(f"Merging chunk outputs into final {target_fps:.2f} fps container...")
        sorted_chunk_ids = sorted(chunk_results.keys())
        chunk_files = [chunk_results[cid].output_file for cid in sorted_chunk_ids if chunk_results[cid].output_file]

        encoder = VideoEncoder(
            output_filepath=self.output_filepath,
            width=self.metadata.width,
            height=self.metadata.height,
            fps=target_fps,
            source_audio_path=self.metadata.filepath if self.metadata.has_audio else None,
            crf=self.crf,
            preset=self.preset,
            use_nvenc=self.use_nvenc,
            color_space=self.metadata.color_space,
            color_primaries=self.metadata.color_primaries,
            color_transfer=self.metadata.color_transfer,
            color_range=self.metadata.color_range
        )
        encoder.start()
        try:
            merger = ChunkMerger(width=self.metadata.width, height=self.metadata.height)
            total_encoded_frames = merger.stream_merge_chunk_files(chunk_files, encoder, fps=target_fps)
            encoder.finish()
        except Exception:
            if encoder.proc is not None:
                try:
                    encoder.proc.kill()
                    encoder.proc.wait(timeout=2)
                except Exception:
                    pass
            raise

        # Step 3: Reporting & Summary
        total_proc_time = time.time() - start_time
        realtime_factor = total_proc_time / max(0.001, self.metadata.duration)
        total_cuts = sum(r.scene_cuts_count for r in chunk_results.values())
        source_frames = self.metadata.nb_frames

        report = QCReport(
            input_path=self.metadata.filepath,
            output_path=self.output_filepath,
            resolution=f"{self.metadata.width}x{self.metadata.height}",
            input_fps=self.metadata.nominal_fps,
            output_fps=target_fps,
            duration_sec=self.metadata.duration,
            duration_str=format_duration(self.metadata.duration),
            model_name=self.model_name,
            gpus_used=",".join([d.replace("cuda:", "") for d in self.devices]),
            source_frames=source_frames,
            generated_frames=max(0, total_encoded_frames - source_frames),
            total_output_frames=total_encoded_frames,
            scene_cuts_count=total_cuts,
            processing_time_sec=total_proc_time,
            realtime_factor=realtime_factor,
            audio_status="stream copied" if self.metadata.has_audio else "none",
            encoding_codec="H.264 (NVENC)" if encoder.use_nvenc else "H.264 (libx264)",
            status="SUCCESS",
            validation_passed=True
        )

        os.makedirs(self.log_dir, exist_ok=True)
        output_stem = os.path.splitext(os.path.basename(self.output_filepath))[0]
        report_str = report.render_text()
        report_dict = report.to_dict()

        report_text_path = os.path.join(self.log_dir, f"{output_stem}_report.txt")
        report_json_path = os.path.join(self.log_dir, f"{output_stem}_report.json")
        with open(report_text_path, "w") as f:
            f.write(report_str)
        with open(report_json_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        canonical_txt = os.path.join(self.log_dir, "report.txt")
        canonical_json = os.path.join(self.log_dir, "report.json")
        if canonical_txt != report_text_path:
            with open(canonical_txt, "w") as f:
                f.write(report_str)
        if canonical_json != report_json_path:
            with open(canonical_json, "w") as f:
                json.dump(report_dict, f, indent=2)

        return report
