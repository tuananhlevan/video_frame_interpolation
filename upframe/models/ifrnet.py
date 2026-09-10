"""IFRNet (Intermediate Feature Refine Network) model adapter."""

import logging
import os
import sys
from typing import Any, Optional, Union
import numpy as np
import torch
from upframe.models.base import BaseVFIModel, ModelRegistry
from upframe.models.weights import resolve_checkpoint
from upframe.utils.tensor import (
    frame_to_tensor,
    tensor_to_frame,
    pad_to_multiple,
    unpad
)

logger = logging.getLogger(__name__)


@ModelRegistry.register("ifrnet")
class IFRNetModel(BaseVFIModel):
    """IFRNet model adapter wrapping the cloned ifrnet repository."""

    def __init__(self) -> None:
        super().__init__(name="ifrnet")
        self.model = None

    def load(
        self,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        model_type: str = "IFRNet",
        **kwargs: Any
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        ifrnet_dir = os.path.abspath("backbones/ifrnet") if os.path.exists("backbones/ifrnet") else os.path.abspath("ifrnet")
        if ifrnet_dir not in sys.path:
            sys.path.insert(0, ifrnet_dir)

        try:
            if model_type == "IFRNet_L":
                from models.IFRNet_L import Model
            elif model_type == "IFRNet_S":
                from models.IFRNet_S import Model
            else:
                from models.IFRNet import Model

            self.model = Model().to(self.device).eval()
        except ImportError as e:
            logger.error(f"Failed to import IFRNet from {ifrnet_dir}: {e}")
            raise

        resolved_path = resolve_checkpoint("ifrnet", checkpoint_path)
        if resolved_path:
            logger.info(f"Loading IFRNet weights from {resolved_path}")
            state_dict = torch.load(resolved_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning("No IFRNet checkpoint found. Using initialized weights.")

        self.is_loaded = True

    def interpolate(
        self,
        frame_a: Union[torch.Tensor, np.ndarray],
        frame_b: Union[torch.Tensor, np.ndarray],
        **kwargs: Any
    ) -> Union[torch.Tensor, np.ndarray]:
        if not self.is_loaded or self.model is None:
            raise RuntimeError("IFRNet model is not loaded. Call load() first.")

        is_numpy = isinstance(frame_a, np.ndarray)
        if is_numpy:
            ta = frame_to_tensor(frame_a, self.device)
            tb = frame_to_tensor(frame_b, self.device)
        else:
            ta = frame_a.to(self.device)
            tb = frame_b.to(self.device)

        orig_h, orig_w = ta.shape[-2:]
        ta, _ = pad_to_multiple(ta, multiple=32)
        tb, _ = pad_to_multiple(tb, multiple=32)

        embt = torch.tensor(0.5, device=self.device).view(1, 1, 1, 1).float()
        with torch.no_grad():
            pred = self.model.inference(ta, tb, embt)

        pred = unpad(pred, orig_h, orig_w)
        if is_numpy:
            return tensor_to_frame(pred)
        return pred
