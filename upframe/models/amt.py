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

    def __init__(self, name: str = "amt") -> None:
        super().__init__(name=name)
        self.model = None
        self.niters = 6

    def load(
        self,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
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

        # 1. Resolve checkpoint path first (using self.name, e.g. amt-g, amt-s, amt-l)
        resolved_path = resolve_checkpoint(self.name, checkpoint_path)
        state_dict = None
        if resolved_path:
            logger.info(f"Loading AMT weights from {resolved_path}")
            try:
                try:
                    ckpt = torch.load(resolved_path, map_location=self.device, weights_only=False)
                except TypeError:
                    ckpt = torch.load(resolved_path, map_location=self.device)
                state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            except Exception as e:
                logger.warning(f"Failed to read AMT checkpoint from {resolved_path}: {e}")

        # 2. Resolve configuration (auto-detect AMT-G, AMT-L, or AMT-S)
        resolved_cfg = None
        if config_path and os.path.exists(config_path):
            resolved_cfg = config_path
        elif config_path and os.path.exists(os.path.join(amt_dir, config_path)):
            resolved_cfg = os.path.join(amt_dir, config_path)
        else:
            variant = "AMT-S"
            if self.name in ["amt-g", "amt-l", "amt-s"]:
                variant = self.name.upper()
            elif resolved_path:
                basename = os.path.basename(resolved_path).lower()
                if "amt-g" in basename or "amt_g" in basename:
                    variant = "AMT-G"
                elif "amt-l" in basename or "amt_l" in basename:
                    variant = "AMT-L"
                elif state_dict and "update4.convf1.weight" in state_dict:
                    out_ch = state_dict["update4.convf1.weight"].shape[0]
                    if out_ch == 128:
                        variant = "AMT-G"
                    elif out_ch == 96:
                        variant = "AMT-L"
                    else:
                        variant = "AMT-S"

            candidate_cfg = os.path.join(amt_dir, "cfgs", f"{variant}.yaml")
            if os.path.exists(candidate_cfg):
                resolved_cfg = candidate_cfg
            elif os.path.exists(os.path.join(amt_dir, "cfgs/AMT-S.yaml")):
                resolved_cfg = os.path.join(amt_dir, "cfgs/AMT-S.yaml")
            else:
                resolved_cfg = "cfgs/AMT-S.yaml"

        try:
            from omegaconf import OmegaConf
            from utils.build_utils import build_from_cfg

            logger.info(f"Building AMT network using config: {resolved_cfg}")
            cfg = OmegaConf.load(resolved_cfg)
            self.model = build_from_cfg(cfg.network).to(self.device).eval()
        except Exception as e:
            logger.error(f"Failed to build AMT network: {e}")
            raise

        if state_dict is not None:
            self.model.load_state_dict(state_dict, strict=False)
            logger.info("Successfully loaded AMT weights.")
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
            outputs = self.model(ta, tb, embt=embt, iters=self.niters, eval=True)
            if isinstance(outputs, dict):
                pred = outputs.get("imgt_pred", outputs)
            elif isinstance(outputs, (list, tuple)):
                pred = outputs[0]
            else:
                pred = outputs

        pred = unpad(pred, orig_h, orig_w)
        if is_numpy:
            return tensor_to_frame(pred)
        return pred


@ModelRegistry.register("amt-s")
class AMTSModel(AMTModel):
    """AMT-S (Small) model adapter."""

    def __init__(self) -> None:
        super().__init__(name="amt-s")


@ModelRegistry.register("amt-l")
class AMTLModel(AMTModel):
    """AMT-L (Large) model adapter."""

    def __init__(self) -> None:
        super().__init__(name="amt-l")


@ModelRegistry.register("amt-g")
class AMTGModel(AMTModel):
    """AMT-G (Giant) model adapter."""

    def __init__(self) -> None:
        super().__init__(name="amt-g")
