"""Baseline Linear Blend / Dummy VFI Model for testing, benchmarking, and CI."""

from typing import Any, Optional, Union
import numpy as np
import torch
from upframe.models.base import BaseVFIModel, ModelRegistry


@ModelRegistry.register("blend")
@ModelRegistry.register("dummy")
class BlendModel(BaseVFIModel):
    """Simple linear blend model: Frame(t=0.5) = 0.5 * A + 0.5 * B.
    
    Useful for testing pipeline synchronization, I/O bottlenecks, and CPU/CI runs.
    """

    def __init__(self) -> None:
        super().__init__(name="blend")

    def load(self, device: str = "cpu", checkpoint_path: Optional[str] = None, **kwargs: Any) -> None:
        self.device = torch.device(device)
        self.is_loaded = True

    def interpolate(
        self,
        frame_a: Union[torch.Tensor, np.ndarray],
        frame_b: Union[torch.Tensor, np.ndarray],
        **kwargs: Any
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(frame_a, np.ndarray) and isinstance(frame_b, np.ndarray):
            if frame_a.dtype == np.uint8:
                return ((frame_a.astype(np.float32) + frame_b.astype(np.float32)) * 0.5).astype(np.uint8)
            return (frame_a + frame_b) * 0.5

        if isinstance(frame_a, torch.Tensor) and isinstance(frame_b, torch.Tensor):
            return (frame_a + frame_b) * 0.5

        raise TypeError("frame_a and frame_b must both be np.ndarray or torch.Tensor")

    def interpolate_batch(
        self,
        batch_a: torch.Tensor,
        batch_b: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        return (batch_a + batch_b) * 0.5
