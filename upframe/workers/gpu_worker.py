"""Independent GPU/CPU chunk worker for parallel video frame interpolation."""

import logging
import os
import time
from typing import List, Optional
import numpy as np
from upframe.core.types import ChunkResult, ChunkTask
from upframe.models.base import BaseVFIModel, ModelRegistry
from upframe.pipeline.anti_flicker import TemporalAntiFlicker
from upframe.pipeline.ball_refiner import BallRefiner
from upframe.pipeline.cadence import is_duplicate_frame
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
        fp16: bool = True,
        tta: bool = False,
        ball_refine: bool = False,
        cadence_filter: bool = False,
        anti_flicker: bool = False,
        scale: float = 1.0,
        batch_size: int = 1
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.fp16 = fp16
        self.tta = tta
        self.ball_refine = ball_refine
        self.cadence_filter = cadence_filter
        self.anti_flicker = anti_flicker
        self.scale = scale
        self.batch_size = max(1, batch_size)
        self.model: Optional[BaseVFIModel] = None
        self.ball_refiner: Optional[BallRefiner] = BallRefiner() if ball_refine else None
        self.temporal_anti_flicker: Optional[TemporalAntiFlicker] = TemporalAntiFlicker() if anti_flicker else None

    def initialize(self) -> None:
        """Initializes model on worker's device."""
        model_cls = ModelRegistry.get(self.model_name)
        self.model = model_cls()
        self.model.load(device=self.device, checkpoint_path=self.checkpoint_path, fp16=self.fp16, tta=self.tta)
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
                        use_nvenc=False,
                        color_space=task.color_space,
                        color_primaries=task.color_primaries,
                        color_transfer=task.color_transfer,
                        color_range=task.color_range
                    )
                    encoder.start()
                    encoder.write_frame(source_frames[0])
                    encoder.finish()
                return ChunkResult(
                    chunk_id=task.chunk_id,
                    status="COMPLETED",
                    output_file=chunk_output_path if len(source_frames) == 1 else None,
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
                use_nvenc=False,
                color_space=task.color_space,
                color_primaries=task.color_primaries,
                color_transfer=task.color_transfer,
                color_range=task.color_range
            )
            encoder.start()

            scene_cuts = 0
            total_output_frames = 0

            use_cadence = getattr(task, "cadence_filter", False) or self.cadence_filter
            use_ball_refine = getattr(task, "ball_refine", False) or self.ball_refine
            use_anti_flicker = getattr(task, "anti_flicker", False) or self.anti_flicker
            scale_val = getattr(task, "scale", 1.0) or self.scale

            if use_ball_refine and self.ball_refiner is None:
                self.ball_refiner = BallRefiner()
            if use_anti_flicker and self.temporal_anti_flicker is None:
                self.temporal_anti_flicker = TemporalAntiFlicker()

            chunk_batch_size = getattr(task, "batch_size", 1) or self.batch_size

            # Generate 50 fps interleaved frames:
            # F[0], F_inter[0.5], F[1], F_inter[1.5], ..., F[N-1]
            if chunk_batch_size > 1:
                from upframe.utils.tensor import frame_to_tensor, tensor_to_frame
                import torch

                i = 0
                n_pairs = len(source_frames) - 1
                while i < n_pairs:
                    end_i = min(i + chunk_batch_size, n_pairs)
                    vfi_indices = []
                    pairs_to_interpolate_a = []
                    pairs_to_interpolate_b = []
                    results = [None] * (end_i - i)

                    for j in range(i, end_i):
                        f_curr = source_frames[j]
                        f_next = source_frames[j + 1]
                        local_idx = j - i

                        if is_scene_cut(f_curr, f_next, threshold=task.scene_threshold):
                            scene_cuts += 1
                            results[local_idx] = f_curr
                        elif use_cadence and is_duplicate_frame(f_curr, f_next):
                            results[local_idx] = f_curr
                        else:
                            vfi_indices.append(local_idx)
                            pairs_to_interpolate_a.append(f_curr)
                            pairs_to_interpolate_b.append(f_next)

                    if vfi_indices:
                        dev = getattr(self.model, "device", torch.device("cpu"))
                        half_mode = getattr(self.model, "half_precision", self.fp16)
                        batch_a = torch.stack([
                            frame_to_tensor(f, dev, half=half_mode).squeeze(0)
                            for f in pairs_to_interpolate_a
                        ])
                        batch_b = torch.stack([
                            frame_to_tensor(f, dev, half=half_mode).squeeze(0)
                            for f in pairs_to_interpolate_b
                        ])

                        pred_batch = self.model.interpolate_batch(
                            batch_a, batch_b, tta=self.tta, scale=scale_val
                        )

                        for k, local_idx in enumerate(vfi_indices):
                            inter_np = tensor_to_frame(pred_batch[k : k + 1])
                            f_curr = pairs_to_interpolate_a[k]
                            f_next = pairs_to_interpolate_b[k]

                            if use_ball_refine and self.ball_refiner is not None:
                                inter_np = self.ball_refiner.refine(f_curr, inter_np, f_next, timestep=0.5)
                            if use_anti_flicker and self.temporal_anti_flicker is not None:
                                inter_np = self.temporal_anti_flicker.process(f_curr, inter_np, f_next)
                            results[local_idx] = inter_np

                    for j in range(i, end_i):
                        encoder.write_frame(source_frames[j])
                        total_output_frames += 1
                        encoder.write_frame(results[j - i])
                        total_output_frames += 1

                    i = end_i
            else:
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
                    elif use_cadence and is_duplicate_frame(f_curr, f_next):
                        # Duplicate pair in slow-motion replay: preserve smooth cadence without AI hallucination
                        encoder.write_frame(f_curr)
                    else:
                        # Run VFI model
                        inter = self.model.interpolate(f_curr, f_next, tta=self.tta, scale=scale_val)
                        if use_ball_refine and self.ball_refiner is not None:
                            inter = self.ball_refiner.refine(f_curr, inter, f_next, timestep=0.5)
                        if use_anti_flicker and self.temporal_anti_flicker is not None:
                            inter = self.temporal_anti_flicker.process(f_curr, inter, f_next)
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
        fp16=task.fp16,
        tta=getattr(task, "tta", False),
        ball_refine=getattr(task, "ball_refine", False),
        cadence_filter=getattr(task, "cadence_filter", False),
        anti_flicker=getattr(task, "anti_flicker", False),
        scale=getattr(task, "scale", 1.0),
    )
    return worker.process_chunk(task)

