"""Worker pool orchestrator managing multi-device concurrent execution and retries."""

from concurrent.futures import Future, ThreadPoolExecutor
import logging
import time
from typing import Callable, Dict, List, Optional, Tuple
from upframe.core.types import ChunkResult, ChunkTask
from upframe.workers.gpu_worker import GPUWorker

logger = logging.getLogger(__name__)


class WorkerPool:
    """Manages worker instances across devices and executes chunk tasks concurrently."""

    def __init__(
        self,
        devices: List[str],
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
        self.devices = devices or ["cpu"]
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.fp16 = fp16
        self.tta = tta
        self.ball_refine = ball_refine
        self.cadence_filter = cadence_filter
        self.anti_flicker = anti_flicker
        self.scale = scale
        self.batch_size = batch_size
        self.workers: Dict[int, GPUWorker] = {}
        self._initialize_workers()

    def _initialize_workers(self) -> None:
        for idx, dev in enumerate(self.devices):
            worker = GPUWorker(
                device=dev,
                model_name=self.model_name,
                checkpoint_path=self.checkpoint_path,
                fp16=self.fp16,
                tta=self.tta,
                ball_refine=self.ball_refine,
                cadence_filter=self.cadence_filter,
                anti_flicker=self.anti_flicker,
                scale=self.scale,
                batch_size=self.batch_size
            )
            worker.initialize()
            self.workers[idx] = worker

    def execute(
        self,
        tasks: List[ChunkTask],
        max_retries: int = 3,
        on_chunk_done: Optional[Callable[[ChunkResult], None]] = None
    ) -> Dict[int, ChunkResult]:
        """Executes tasks across the worker pool, retrying failures up to max_retries."""
        results: Dict[int, ChunkResult] = {}
        if not tasks:
            return results

        num_workers = len(self.devices)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            task_idx = 0
            futures: Dict[Future[ChunkResult], Tuple[ChunkTask, int, int]] = {}
            worker_pool = list(range(num_workers))

            # Fill initial queue
            while worker_pool and task_idx < len(tasks):
                w_id = worker_pool.pop(0)
                task = tasks[task_idx]
                task.device = self.devices[w_id]
                fut = executor.submit(self.workers[w_id].process_chunk, task)
                futures[fut] = (task, w_id, 0)
                task_idx += 1

            while futures:
                done_futs = [f for f in futures if f.done()]
                if not done_futs:
                    time.sleep(0.05)
                    continue

                for fut in done_futs:
                    task, w_id, retry_count = futures.pop(fut)
                    dev = self.devices[w_id]
                    try:
                        res: ChunkResult = fut.result()
                    except Exception as e:
                        res = ChunkResult(
                            chunk_id=task.chunk_id,
                            status="FAILED",
                            output_file=None,
                            source_frames_count=0,
                            output_frames_count=0,
                            scene_cuts_count=0,
                            elapsed_seconds=0.0,
                            error_message=str(e)
                        )

                    if res.status == "COMPLETED":
                        results[task.chunk_id] = res
                        if on_chunk_done:
                            on_chunk_done(res)
                    else:
                        if retry_count < max_retries:
                            logger.warning(
                                f"Chunk {task.chunk_id} failed on worker {w_id} ({dev}): {res.error_message}. "
                                f"Retrying ({retry_count + 1}/{max_retries})..."
                            )
                            new_fut = executor.submit(self.workers[w_id].process_chunk, task)
                            futures[new_fut] = (task, w_id, retry_count + 1)
                            continue
                        else:
                            raise RuntimeError(
                                f"Chunk {task.chunk_id} failed after {max_retries} retries: {res.error_message}"
                            )

                    # Reassign freed worker to next pending task
                    if task_idx < len(tasks):
                        next_task = tasks[task_idx]
                        next_task.device = dev
                        new_fut = executor.submit(self.workers[w_id].process_chunk, next_task)
                        futures[new_fut] = (next_task, w_id, 0)
                        task_idx += 1
                    else:
                        worker_pool.append(w_id)

        return results
