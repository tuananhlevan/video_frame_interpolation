"""BWDIF (Bob Weaver Deinterlacing Filter) adapter for motion-adaptive field separation."""

from typing import Any, Optional, Union
import numpy as np
import torch
from upframe.models.base import BaseVFIModel, ModelRegistry


@ModelRegistry.register("bwdif")
@ModelRegistry.register("deinterlace")
class BWDIFModel(BaseVFIModel):
    """Adapter for BWDIF motion-adaptive deinterlacing.
    
    In the main pipeline scheduler, BWDIF runs directly in high-performance FFmpeg SIMD/AVX2.
    This adapter provides the BaseVFIModel Python interface for compatibility and unit testing.
    """

    def __init__(self, name: str = "bwdif") -> None:
        super().__init__(name=name)

    def load(self, device: str = "cpu", checkpoint_path: Optional[str] = None, **kwargs: Any) -> None:
        self.device = torch.device(device)
        self.is_loaded = True

    def interpolate(
        self,
        frame_a: Union[torch.Tensor, np.ndarray],
        frame_b: Union[torch.Tensor, np.ndarray],
        **kwargs: Any
    ) -> Union[torch.Tensor, np.ndarray]:
        """Synthesizes intermediate frame using field-aware blend/bob."""
        if isinstance(frame_a, torch.Tensor) and isinstance(frame_b, torch.Tensor):
            return 0.5 * (frame_a + frame_b)
        elif isinstance(frame_a, np.ndarray) and isinstance(frame_b, np.ndarray):
            return ((frame_a.astype(np.float32) + frame_b.astype(np.float32)) * 0.5).astype(frame_a.dtype)
        else:
            raise TypeError("Inputs must both be torch.Tensor or np.ndarray")
