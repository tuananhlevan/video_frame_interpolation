"""FILM (Frame Interpolation for Large Motion) model adapter."""

import logging
import os
import sys
from typing import Any, Optional, Union
import numpy as np
import torch
from upframe.models.base import BaseVFIModel, ModelRegistry
from upframe.models.weights import resolve_checkpoint

logger = logging.getLogger(__name__)


@ModelRegistry.register("film")
class FILMModel(BaseVFIModel):
    """FILM model adapter wrapping the cloned film repository."""

    def __init__(self) -> None:
        super().__init__(name="film")
        self.interpolator = None

    def load(
        self,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        film_dir = os.path.abspath("backbones/film") if os.path.exists("backbones/film") else os.path.abspath("film")
        if film_dir not in sys.path:
            sys.path.insert(0, film_dir)

        resolved_path = resolve_checkpoint("film", checkpoint_path)

        try:
            from eval import interpolator
            if resolved_path and os.path.exists(resolved_path):
                self.interpolator = interpolator.Interpolator(resolved_path, None)
                self.is_loaded = True
                logger.info(f"Loaded FILM model from {resolved_path}")
            else:
                logger.warning(f"FILM saved_model not found. Candidates searched in ~/.cache/upframe/film and film/")
                self.is_loaded = False
        except ImportError as e:
            logger.warning(f"TensorFlow / FILM eval module not available: {e}. FILM requires TensorFlow runtime.")
            self.is_loaded = False

    def interpolate(
        self,
        frame_a: Union[torch.Tensor, np.ndarray],
        frame_b: Union[torch.Tensor, np.ndarray],
        **kwargs: Any
    ) -> Union[torch.Tensor, np.ndarray]:
        if not self.is_loaded or self.interpolator is None:
            raise RuntimeError("FILM model is not loaded. Ensure TensorFlow and FILM weights are available.")

        is_torch = isinstance(frame_a, torch.Tensor)
        if is_torch:
            fa = frame_a.detach().cpu().permute(1, 2, 0).numpy()
            fb = frame_b.detach().cpu().permute(1, 2, 0).numpy()
        else:
            fa = frame_a
            fb = frame_b

        if fa.dtype == np.uint8:
            fa = fa.astype(np.float32) / 255.0
            fb = fb.astype(np.float32) / 255.0

        batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
        out = self.interpolator.interpolate(fa[np.newaxis, ...], fb[np.newaxis, ...], batch_dt)[0]

        if is_torch:
            return torch.from_numpy(out).permute(2, 0, 1).to(self.device)
        
        return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
