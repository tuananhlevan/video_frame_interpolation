"""InterpAny-Clearer (ECCV 2024 Oral & TPAMI) model adapter for velocity-disambiguated VFI."""

import logging
import os
import sys
from typing import Any, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F

from upframe.models.base import BaseVFIModel, ModelRegistry
from upframe.models.weights import resolve_checkpoint
from upframe.utils.tensor import (
    frame_to_tensor,
    tensor_to_frame,
    pad_to_multiple,
    unpad
)

logger = logging.getLogger(__name__)

_INTERPANY_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "backbones", "interpany", "models", "DI-RIFE")
)


def _ensure_interpany_repo() -> None:
    """Clones the InterpAny-Clearer repository if not already present."""
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backbones", "interpany"))
    if not os.path.exists(_INTERPANY_DIR):
        logger.info(f"InterpAny backbone code not found at {repo_dir}. Cloning repository...")
        import subprocess
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/zzh-tech/InterpAny-Clearer.git", repo_dir],
                check=True
            )
        except Exception as exc:
            logger.error(f"Failed to auto-clone InterpAny-Clearer: {exc}")


def _prepare_interpany_namespace() -> None:
    """Isolates the InterpAny backbone namespace from other backbones."""
    _ensure_interpany_repo()
    for mod in list(sys.modules.keys()):
        if mod == "model" or mod.startswith("model."):
            del sys.modules[mod]

    sys.path = [
        p for p in sys.path
        if not any(
            p.endswith(os.path.join("backbones", b))
            for b in ("gmfss", "ema_vfi", "rife", "film", "ifrnet", "amt")
        )
    ]
    if _INTERPANY_DIR in sys.path:
        sys.path.remove(_INTERPANY_DIR)
    sys.path.insert(0, _INTERPANY_DIR)


@ModelRegistry.register("interpany")
class InterpAnyModel(BaseVFIModel):
    """InterpAny-Clearer model adapter (DR-RIFE with Distance Indexing and Iterative Reference)."""

    def __init__(self, name: str = "interpany") -> None:
        super().__init__(name=name)
        self.model = None
        self.iters = 2
        self._scale = 1.0

    def load(
        self,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        iters: int = 2,
        fp16: bool = True,
        scale: float = 1.0,
        **kwargs: Any
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(f"CUDA requested ('{device}') but not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.iters = iters
        self._scale = scale

        _prepare_interpany_namespace()

        resolved_path = resolve_checkpoint(self.name, checkpoint_path)
        if not resolved_path:
            raise FileNotFoundError(
                f"InterpAny checkpoint not found for '{self.name}'. "
                f"Ensure backbones/interpany_checkpoints is populated."
            )

        # If a file like flownet_sdi.pkl is passed, get its parent directory
        if os.path.isfile(resolved_path):
            ckpt_dir = os.path.dirname(resolved_path)
        else:
            if os.path.exists(os.path.join(resolved_path, "train_sdi_log")):
                ckpt_dir = os.path.join(resolved_path, "train_sdi_log")
            else:
                ckpt_dir = resolved_path

        logger.info(f"Loading InterpAny ({self.name}) from: {ckpt_dir}")

        try:
            import model.RIFE_sdi_recur as _rife_sdi_mod
            _rife_sdi_mod.device = self.device

            from model.RIFE_sdi_recur import Model
            interp_model = Model()
            interp_model.load_model(ckpt_dir)
            interp_model.eval()
            interp_model.flownet.to(self.device)

            self.model = interp_model
        except Exception as exc:
            logger.error(f"Failed to load InterpAny model from {ckpt_dir}: {exc}")
            raise

        self.half_precision = fp16 and (self.device.type == "cuda")
        self.is_loaded = True
        logger.info(f"InterpAny ({self.name}) loaded successfully on {self.device}.")

    def interpolate(
        self,
        frame_a: Union[torch.Tensor, np.ndarray],
        frame_b: Union[torch.Tensor, np.ndarray],
        **kwargs: Any
    ) -> Union[torch.Tensor, np.ndarray]:
        if not self.is_loaded or self.model is None:
            raise RuntimeError(f"InterpAny ({self.name}) is not loaded. Call load() first.")

        is_numpy = isinstance(frame_a, np.ndarray)
        if is_numpy:
            ta = frame_to_tensor(frame_a, self.device, half=False)
            tb = frame_to_tensor(frame_b, self.device, half=False)
        else:
            ta = frame_a.to(self.device).float()
            tb = frame_b.to(self.device).float()

        if ta.dim() == 3:
            ta = ta.unsqueeze(0)
            tb = tb.unsqueeze(0)

        orig_h, orig_w = ta.shape[-2:]
        ta, _ = pad_to_multiple(ta, multiple=32)
        tb, _ = pad_to_multiple(tb, multiple=32)

        timestep = kwargs.get("timestep", 0.5)
        iters = kwargs.get("iters", self.iters)
        scale = kwargs.get("scale", self._scale)

        sdi_map = torch.zeros_like(ta[:, :1, :, :]) + timestep

        with torch.no_grad():
            if self.half_precision:
                with torch.amp.autocast("cuda"):
                    pred = self.model.inference(
                        ta, tb,
                        scale=scale,
                        iters=iters,
                        sdi_map=sdi_map
                    )
            else:
                pred = self.model.inference(
                    ta, tb,
                    scale=scale,
                    iters=iters,
                    sdi_map=sdi_map
                )

        pred = unpad(pred, orig_h, orig_w)
        pred = torch.clamp(pred, 0.0, 1.0)

        if is_numpy:
            return tensor_to_frame(pred)
        return pred

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        super().unload()


@ModelRegistry.register("interpany-vgg")
class InterpAnyVGGModel(InterpAnyModel):
    """InterpAny with DR-RIFE-vgg (trained with perceptual LPIPS loss for artifact-free interpolation)."""

    def __init__(self) -> None:
        super().__init__(name="interpany-vgg")


@ModelRegistry.register("interpany-pro")
class InterpAnyProModel(InterpAnyModel):
    """InterpAny with DR-RIFE-pro (extended training for large-displacement motion)."""

    def __init__(self) -> None:
        super().__init__(name="interpany-pro")
