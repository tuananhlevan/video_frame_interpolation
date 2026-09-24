"""EMA-VFI (Extracting Motion and Appearance via Inter-Frame Attention) adapter.

EMA-VFI uses a Swin-Transformer-based backbone that decouples motion estimation
and appearance extraction via inter-frame cross-attention, achieving top SOTA
accuracy on Vimeo90K and SNU-FILM benchmarks.

Paper: https://arxiv.org/abs/2303.00440
Backbone repo: https://github.com/MCG-NJU/EMA-VFI
Expected structure:
    backbones/ema_vfi/
        Trainer.py          (Model class, inference methods)
        config.py           (init_model_config helper, MODEL_CONFIG)
        model/              (feature_extractor, flow_estimation, warplayer, …)
        ckpt/
            ours.pkl        (large model — default)
            ours_small.pkl  (small model — 'ema-vfi-s')
"""

import logging
import os
import sys
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from upframe.models.base import BaseVFIModel, ModelRegistry
from upframe.utils.tensor import frame_to_tensor, tensor_to_frame

logger = logging.getLogger(__name__)

_BACKBONE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "backbones", "ema_vfi")
)

_CKPT_CANDIDATES = {
    # (model_variant, ckpt_filename)
    "ema-vfi":   ["ours.pkl", "ema-vfi-l.pkl", "ema_vfi.pkl"],
    "ema-vfi-s": ["ours_small.pkl", "ema-vfi-s.pkl", "ema_vfi_small.pkl"],
}
_CACHE_CKPT_BASE = os.path.expanduser("~/.cache/upframe/ema_vfi/ckpt")


from upframe.models.weights import resolve_checkpoint


def _find_ckpt(variant: str, explicit: Optional[str] = None) -> Optional[str]:
    """Resolve checkpoint path for a given EMA-VFI variant, auto-downloading if missing."""
    return resolve_checkpoint(variant, explicit)


def _make_model(variant: str) -> Any:
    """Instantiate the EMA-VFI `Model` for the given variant.

    We import `config` and `Trainer` from the backbone directory so we can
    reconfigure MODEL_CONFIG before instantiation (large vs. small).
    """
    # Evict other backbones from sys.path and prioritize EMA-VFI
    sys.path = [p for p in sys.path if not any(p.endswith(os.path.join("backbones", b)) for b in ("gmfss", "rife", "amt", "film", "ifrnet"))]
    if _BACKBONE_DIR in sys.path:
        sys.path.remove(_BACKBONE_DIR)
    sys.path.insert(0, _BACKBONE_DIR)

    # Force reimport so patched MODEL_CONFIG and model modules take effect
    for mod in list(sys.modules.keys()):
        if mod in ("config", "Trainer", "model") or mod.startswith("model."):
            del sys.modules[mod]

    import config as ema_cfg  # type: ignore[import]
    from Trainer import Model  # type: ignore[import]

    if variant == "ema-vfi-s":
        ema_cfg.MODEL_CONFIG["LOGNAME"] = "ours_small"
        ema_cfg.MODEL_CONFIG["MODEL_ARCH"] = ema_cfg.init_model_config(F=16, depth=[2, 2, 2, 2, 2])
    else:
        ema_cfg.MODEL_CONFIG["LOGNAME"] = "ours"
        ema_cfg.MODEL_CONFIG["MODEL_ARCH"] = ema_cfg.init_model_config(F=32, depth=[2, 2, 2, 4, 4])

    # local_rank = -1 → no DDP, no optimizer init waste
    m = Model(local_rank=-1)
    return m


class _EMAVFIPadder:
    """Pad tensor to a multiple of `divisor`, then unpad after inference."""

    def __init__(self, shape: torch.Size, divisor: int = 32):
        h, w = shape[-2], shape[-1]
        pad_h = (divisor - h % divisor) % divisor
        pad_w = (divisor - w % divisor) % divisor
        self._pad = (0, pad_w, 0, pad_h)
        self._orig_h = h
        self._orig_w = w

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, self._pad)

    def unpad(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : self._orig_h, : self._orig_w]


def _register_ema_vfi(variant: str):
    """Factory that creates and registers an EMAVFIModel for `variant`."""

    @ModelRegistry.register(variant)
    class EMAVFIModel(BaseVFIModel):
        __doc__ = f"EMA-VFI ({variant}) adapter — Swin-Transformer inter-frame attention VFI."

        def __init__(self) -> None:
            super().__init__(name=variant)
            self._ema_model: Optional[Any] = None
            self.half_precision: bool = False
            self._use_tta: bool = False
            self._variant = variant

        def load(
            self,
            device: str = "cuda",
            checkpoint_path: Optional[str] = None,
            fp16: bool = False,
            tta: bool = False,
            **kwargs: Any,
        ) -> None:
            """Load EMA-VFI weights.

            Args:
                device: Target device.
                checkpoint_path: Optional explicit path to the .pkl checkpoint.
                fp16: Enable half-precision inference.
                tta: Enable test-time augmentation (flip ensemble) for higher
                     quality at ~2× inference cost.
            """
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA not available — falling back to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)

            self._use_tta = tta

            ckpt_path = _find_ckpt(self._variant, checkpoint_path)
            if ckpt_path is None:
                ema_ckpt_dir = os.path.join(_BACKBONE_DIR, "ckpt")
                expected = _CKPT_CANDIDATES[self._variant][0]
                raise FileNotFoundError(
                    f"EMA-VFI ({self._variant}) checkpoint not found.\n"
                    f"Please place '{expected}' in: {ema_ckpt_dir}\n"
                    f"Download from the EMA-VFI GitHub releases page."
                )

            logger.info(f"Loading EMA-VFI ({self._variant}) from: {ckpt_path}")

            model_obj = _make_model(self._variant)

            # Patch load_model to accept explicit path instead of scanning `ckpt/`
            import torch as _torch

            def _convert(param):
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k and "attn_mask" not in k and "HW" not in k
                }

            raw = _torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model_obj.net.load_state_dict(_convert(raw))

            model_obj.net.to(self.device)
            model_obj.net.eval()

            self._ema_model = model_obj
            self.half_precision = fp16 and (self.device.type == "cuda")
            self.is_loaded = True
            logger.info(f"EMA-VFI ({self._variant}) loaded successfully.")

        # ------------------------------------------------------------------

        def _to_tensor(self, frame: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
            if isinstance(frame, np.ndarray):
                t = frame_to_tensor(frame, self.device, half=False)
            else:
                t = frame.to(self.device).float()
            return t

        def interpolate(
            self,
            frame_a: Union[torch.Tensor, np.ndarray],
            frame_b: Union[torch.Tensor, np.ndarray],
            **kwargs: Any,
        ) -> Union[torch.Tensor, np.ndarray]:
            """Synthesize mid-frame between frame_a and frame_b (t = 0.5 default)."""
            if not self.is_loaded or self._ema_model is None:
                raise RuntimeError(
                    f"EMA-VFI ({self._variant}) not loaded. Call model.load() first."
                )

            is_numpy = isinstance(frame_a, np.ndarray)
            img0 = self._to_tensor(frame_a)
            img1 = self._to_tensor(frame_b)

            if img0.dim() == 3:
                img0 = img0.unsqueeze(0)
                img1 = img1.unsqueeze(0)

            padder = _EMAVFIPadder(img0.shape, divisor=32)
            img0_p = padder.pad(img0)
            img1_p = padder.pad(img1)

            timestep = kwargs.get("timestep", 0.5)
            tta = kwargs.get("tta", self._use_tta)

            with torch.no_grad():
                if self.half_precision:
                    with torch.amp.autocast("cuda"):
                        pred = self._ema_model.inference(img0_p, img1_p, TTA=tta, timestep=timestep)
                else:
                    pred = self._ema_model.inference(img0_p, img1_p, TTA=tta, timestep=timestep)

            pred = padder.unpad(pred)
            pred = torch.clamp(pred, 0.0, 1.0)

            if is_numpy:
                return tensor_to_frame(pred)
            return pred

        def unload(self) -> None:
            if self._ema_model is not None:
                del self._ema_model.net
                self._ema_model = None
            super().unload()

    # Return class so registration side-effects happen
    return EMAVFIModel


# Register both variants at import time
_EMAVFILarge = _register_ema_vfi("ema-vfi")
_EMAVFISmall = _register_ema_vfi("ema-vfi-s")
