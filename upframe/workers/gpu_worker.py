"""Independent GPU/CPU chunk worker for parallel video frame interpolation."""

import logging
import os
import time
from typing import List, Optional
import numpy as np
from upframe.core.types import ChunkResult, ChunkTask
from upframe.models.base import BaseVFIModel, ModelRegistry
from upframe.pipeline.decode import VideoDecoder
from upframe.pipeline.encode import VideoEncoder
from upframe.pipeline.scene_detect import is_scene_cut

logger = logging.getLogger(__name__)


class GPUWorker:
    """Worker instance assigned to a specific GPU/CPU device."""

    def __init__(
        self,
        device: str,
        model_name: str = "rife",
        checkpoint_path: Optional[str] = None,
        fp16: bool = True
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.fp16 = fp16
        self.model: Optional[BaseVFIModel] = None

    def initialize(self) -> None:
        """Initializes model on worker's device."""
        model_cls = ModelRegistry.get(self.model_name)
        self.model = model_cls()
        self.model.load(device=self.device, checkpoint_path=self.checkpoint_path, fp16=self.fp16)
        logger.info(f"Worker initialized on {self.device} with model {self.model_name}")

    def process_chunk(self, task: ChunkTask) -> ChunkResult:
        """Processes a chunk of video frames: decodes, cuts, interpolates, and encodes chunk output."""
        start_time = time.time()
        chunk_filename = f"chunk_{task.chunk_id:05d}.mp4"
        chunk_output_path = os.path.join(task.output_dir, chunk_filename)
        os.makedirs(task.output_dir, exist_ok=True)

        if self.model is None:
            self.initialize()

        count = task.end_frame - task.start_frame + 1
        decoder = VideoDecoder(
            task.input_path,
            width=task.width,
            height=task.height
        )

        try:
            # Decode source frames for this chunk
            source_frames: List[np.ndarray] = decoder.decode_chunk(
                start_frame=task.start_frame,
                count=count,
                fps=task.fps
            )

            if len(source_frames) < 2:
                if len(source_frames) == 1:
                    encoder = VideoEncoder(
                        chunk_output_path,
                        width=task.width,
                        height=task.height,
                        fps=50.0,
                        crf=14,
                        preset="ultrafast",
                        use_nvenc=False
                    )
                    encoder.start()
                    encoder.write_frame(source_frames[0])
                    encoder.finish()
                return ChunkResult(
                    chunk_id=task.chunk_id,
                    status="COMPLETED",
                    output_file=chunk_output_path,
                    source_frames_count=len(source_frames),
                    output_frames_count=len(source_frames),
                    scene_cuts_count=0,
                    elapsed_seconds=time.time() - start_time
                )

            encoder = VideoEncoder(
                chunk_output_path,
                width=task.width,
                height=task.height,
                fps=50.0,
                crf=14,
                preset="ultrafast",
                use_nvenc=False
            )
            encoder.start()

            scene_cuts = 0
            total_output_frames = 0

            # Generate 50 fps interleaved frames:
            # F[0], F_inter[0.5], F[1], F_inter[1.5], ..., F[N-1]
            for i in range(len(source_frames) - 1):
                f_curr = source_frames[i]
                f_next = source_frames[i + 1]

                # Write current source frame
                encoder.write_frame(f_curr)
                total_output_frames += 1

                # Check scene cut
                if is_scene_cut(f_curr, f_next, threshold=task.scene_threshold):
                    scene_cuts += 1
                    # Avoid hybrid blur: hold/duplicate current frame
                    encoder.write_frame(f_curr)
                else:
                    # Run VFI model
                    inter = self.model.interpolate(f_curr, f_next)
                    encoder.write_frame(inter)
                
                total_output_frames += 1

            # Write the last source frame of the chunk
            encoder.write_frame(source_frames[-1])
            total_output_frames += 1

            encoder.finish()

            elapsed = time.time() - start_time
            return ChunkResult(
                chunk_id=task.chunk_id,
                status="COMPLETED",
                output_file=chunk_output_path,
                source_frames_count=len(source_frames),
                output_frames_count=total_output_frames,
                scene_cuts_count=scene_cuts,
                elapsed_seconds=elapsed
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.exception(f"Error processing chunk {task.chunk_id}: {e}")
            return ChunkResult(
                chunk_id=task.chunk_id,
                status="FAILED",
                output_file=None,
                source_frames_count=0,
                output_frames_count=0,
                scene_cuts_count=0,
                elapsed_seconds=elapsed,
                error_message=str(e)
            )


def worker_process_entrypoint(
    task: ChunkTask,
    device: str
) -> ChunkResult:
    """Standalone worker function called by multiprocessing/pools."""
    worker = GPUWorker(
        device=device,
        model_name=task.model_name,
        checkpoint_path=task.checkpoint_path,
        fp16=task.fp16
    )
    return worker.process_chunk(task)
