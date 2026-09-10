"""Base interface and registry for Video Frame Insertion (VFI) models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union
import numpy as np
import torch


class BaseVFIModel(ABC):
    """Abstract Base Class for all VFI models.
    
    Any model integrated into the upframe pipeline (RIFE, AMT, FILM, IFRNet, Blend, etc.)
    must inherit from this class and implement its abstract methods.
    """

    def __init__(self, name: str = "vfi_base") -> None:
        self.name = name
        self.device = torch.device("cpu")
        self.is_loaded = False

    @abstractmethod
    def load(self, device: str = "cuda", checkpoint_path: Optional[str] = None, **kwargs: Any) -> None:
        """Load model weights and move the model to the specified target device."""
        pass

    @abstractmethod
    def interpolate(
        self,
        frame_a: Union[torch.Tensor, np.ndarray],
        frame_b: Union[torch.Tensor, np.ndarray],
        **kwargs: Any
    ) -> Union[torch.Tensor, np.ndarray]:
        """Synthesize intermediate frame between frame_a and frame_b (t = 0.5).
        
        Supports both single PyTorch tensors (C, H, W or 1, C, H, W, range [0, 1])
        or NumPy arrays (H, W, C, uint8 [0, 255] or float [0, 1]).
        """
        pass

    def interpolate_batch(
        self,
        batch_a: torch.Tensor,
        batch_b: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        """Synthesize intermediate frames for batches of frames.
        
        batch_a, batch_b: (B, C, H, W) normalized to [0, 1].
        Returns: (B, C, H, W) normalized to [0, 1].
        """
        results = []
        for i in range(batch_a.shape[0]):
            inter = self.interpolate(batch_a[i : i + 1], batch_b[i : i + 1], **kwargs)
            results.append(inter)
        return torch.cat(results, dim=0)

    def unload(self) -> None:
        """Release model from memory and clear GPU cache."""
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ModelRegistry:
    """Registry for available VFI models."""
    _models: Dict[str, Type[BaseVFIModel]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(subclass: Type[BaseVFIModel]):
            cls._models[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseVFIModel]:
        key = name.lower()
        if key not in cls._models:
            raise KeyError(
                f"Model '{name}' is not registered. Available models: {list(cls._models.keys())}"
            )
        return cls._models[key]

    @classmethod
    def list_models(cls) -> list[str]:
        return sorted(list(cls._models.keys()))
