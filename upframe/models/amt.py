"""AMT (All-Pairs Multi-Scale Motion Estimation) model adapter."""

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


@ModelRegistry.register("amt")
class AMTModel(BaseVFIModel):
    """AMT model adapter wrapping the cloned amt repository."""

    def __init__(self) -> None:
        super().__init__(name="amt")
        self.model = None
        self.niters = 6

    def load(
        self,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        config_path: str = "amt/cfgs/AMT-S.yaml",
        niters: int = 6,
        **kwargs: Any
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.niters = niters
        amt_dir = os.path.abspath("backbones/amt") if os.path.exists("backbones/amt") else os.path.abspath("amt")
        if amt_dir not in sys.path:
            sys.path.insert(0, amt_dir)

        try:
            from omegaconf import OmegaConf
            from utils.build_utils import build_from_cfg

            if config_path and os.path.exists(config_path):
                resolved_cfg = config_path
            elif os.path.exists(os.path.join(amt_dir, "cfgs/AMT-S.yaml")):
                resolved_cfg = os.path.join(amt_dir, "cfgs/AMT-S.yaml")
            else:
                resolved_cfg = "cfgs/AMT-S.yaml"
            cfg = OmegaConf.load(resolved_cfg)
            self.model = build_from_cfg(cfg.network).to(self.device).eval()
        except Exception as e:
            logger.error(f"Failed to build AMT network: {e}")
            raise

        resolved_path = resolve_checkpoint("amt", checkpoint_path)
        if resolved_path:
            logger.info(f"Loading AMT weights from {resolved_path}")
            try:
                ckpt = torch.load(resolved_path, map_location=self.device, weights_only=False)
            except TypeError:
                ckpt = torch.load(resolved_path, map_location=self.device)
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning("No AMT checkpoint found. Using initialized network.")

        self.is_loaded = True

    def interpolate(
        self,
        frame_a: Union[torch.Tensor, np.ndarray],
        frame_b: Union[torch.Tensor, np.ndarray],
        **kwargs: Any
    ) -> Union[torch.Tensor, np.ndarray]:
        if not self.is_loaded or self.model is None:
            raise RuntimeError("AMT model is not loaded. Call load() first.")

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

        with torch.no_grad():
            embt = torch.tensor(0.5, device=self.device).view(1, 1, 1, 1).float()
            outputs = self.model(ta, tb, embt=embt, iters=self.niters)
            pred = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

        pred = unpad(pred, orig_h, orig_w)
        if is_numpy:
            return tensor_to_frame(pred)
        return pred
