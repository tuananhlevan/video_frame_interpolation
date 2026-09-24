"""GMFSS Fortuna (Global Matching Frame Super Sampling) adapter for upframe.

GMFSS uses GMFlow-based global feature correlation for large-displacement motion,
which excels at fast-moving objects and camera pans in sports footage.

Backbone repo: https://github.com/98mxr/GMFSS_Fortuna
Expected structure:
    backbones/gmfss/
        model/GMFSS_infer_b.py   (base model)
        model/GMFSS_infer_u.py   (union model, optional)
        train_log/               (default weight directory)
            flownet.pkl
            metric.pkl
            feat.pkl
            fusionnet.pkl
"""

import logging
import os
import sys
from typing import Any, Optional, Union

import numpy as np
import torch

from upframe.models.base import BaseVFIModel, ModelRegistry
from upframe.utils.tensor import frame_to_tensor, tensor_to_frame, pad_to_multiple, unpad

logger = logging.getLogger(__name__)

# Canonical backbone directory
_BACKBONE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "backbones", "gmfss")
)

# Default weight folder inside the backbone (matches GMFSS original layout)
_DEFAULT_WEIGHT_DIR = os.path.join(_BACKBONE_DIR, "train_log")

# Fallback cache location
_CACHE_WEIGHT_DIR = os.path.expanduser("~/.cache/upframe/gmfss/train_log")


from upframe.models.weights import resolve_checkpoint

def _find_weight_dir(explicit: Optional[str] = None) -> Optional[str]:
    """Return weight directory that contains all 4 GMFSS pkl files, auto-downloading if missing."""
    return resolve_checkpoint("gmfss", explicit)


@ModelRegistry.register("gmfss")
class GMFSSModel(BaseVFIModel):
    """GMFSS Fortuna (base variant) adapter.

    Uses global GMFlow optical flow + softsplat warping for artifact-free
    interpolation of fast-motion sports content.
    """

    def __init__(self) -> None:
        super().__init__(name="gmfss")
        self._model: Optional[Any] = None
        self.half_precision: bool = False
        self._scale: float = 1.0

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        fp16: bool = False,
        scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Load GMFSS Fortuna weights.

        Args:
            device: Target device string ('cuda', 'cuda:0', 'cpu', …).
            checkpoint_path: Optional path to the weight *directory* containing
                             flownet.pkl, metric.pkl, feat.pkl, fusionnet.pkl.
                             If not provided, the default backbones/gmfss/train_log
                             directory is used.
            fp16: Enable half-precision inference.
            scale: Optical-flow scale factor (use 0.5 for 4 K to save VRAM).
        """
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available — falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self._scale = scale

        # Ensure clean namespace for GMFSS 'model' package
        for mod in list(sys.modules.keys()):
            if mod == "model" or mod.startswith("model."):
                del sys.modules[mod]

        # Evict other backbones from sys.path and prioritize GMFSS
        sys.path = [p for p in sys.path if not any(p.endswith(os.path.join("backbones", b)) for b in ("ema_vfi", "rife", "amt", "film", "ifrnet"))]
        if _BACKBONE_DIR in sys.path:
            sys.path.remove(_BACKBONE_DIR)
        sys.path.insert(0, _BACKBONE_DIR)

        # Ensure model/__init__.py exists so GMFSS model directory is recognized as a package
        init_file = os.path.join(_BACKBONE_DIR, "model", "__init__.py")
        if not os.path.exists(init_file):
            try:
                open(init_file, "a").close()
            except OSError:
                pass

        weight_dir = _find_weight_dir(checkpoint_path)
        if weight_dir is None:
            raise FileNotFoundError(
                "GMFSS weights not found. Please place flownet.pkl, metric.pkl, "
                f"feat.pkl, fusionnet.pkl in: {_DEFAULT_WEIGHT_DIR}"
            )

        logger.info(f"Loading GMFSS Fortuna weights from: {weight_dir}")

        try:
            from model.GMFSS_infer_b import Model  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                f"Cannot import GMFSS backbone from {_BACKBONE_DIR}. "
                "Ensure the repo is cloned to backbones/gmfss."
            ) from exc

        gmfss = Model()

        # Override the module-level `device` variable in GMFSS source before
        # calling load_model, which uses it for map_location.
        import model.GMFSS_infer_b as _gmfss_mod  # type: ignore[import]
        _gmfss_mod.device = self.device

        gmfss.load_model(weight_dir, rank=-1)
        gmfss.eval()

        # Move each sub-network to target device explicitly
        for sub in (gmfss.flownet, gmfss.metricnet, gmfss.feat_ext, gmfss.fusionnet):
            sub.to(self.device)

        self._model = gmfss
        self.half_precision = fp16 and (self.device.type == "cuda")
        self.is_loaded = True
        logger.info("GMFSS Fortuna loaded successfully.")

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, frame: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(frame, np.ndarray):
            t = frame_to_tensor(frame, self.device, half=False)
        else:
            t = frame.to(self.device).float()
        return t

    # ------------------------------------------------------------------
    # interpolate
    # ------------------------------------------------------------------

    def interpolate(
        self,
        frame_a: Union[torch.Tensor, np.ndarray],
        frame_b: Union[torch.Tensor, np.ndarray],
        **kwargs: Any,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Synthesize the mid-frame between frame_a and frame_b (t = 0.5).

        Accepts either numpy HWC uint8 frames or (1, C, H, W) float tensors.
        Returns the same type as the input.
        """
        if not self.is_loaded or self._model is None:
            raise RuntimeError("GMFSS model is not loaded. Call model.load() first.")

        is_numpy = isinstance(frame_a, np.ndarray)
        img0 = self._to_tensor(frame_a)
        img1 = self._to_tensor(frame_b)

        # Ensure 4-D batch dimension
        if img0.dim() == 3:
            img0 = img0.unsqueeze(0)
            img1 = img1.unsqueeze(0)

        orig_h, orig_w = img0.shape[-2:]

        # GMFSS requires multiples of 64
        pad_multiple = max(64, int(64 / self._scale))
        img0_p, _ = pad_to_multiple(img0, multiple=pad_multiple)
        img1_p, _ = pad_to_multiple(img1, multiple=pad_multiple)

        timestep = kwargs.get("timestep", 0.5)
        scale = kwargs.get("scale", self._scale)

        with torch.no_grad():
            if self.half_precision:
                with torch.amp.autocast("cuda"):
                    reuse = self._model.reuse(img0_p, img1_p, scale)
                    pred = self._model.inference(img0_p, img1_p, reuse, timestep)
            else:
                reuse = self._model.reuse(img0_p, img1_p, scale)
                pred = self._model.inference(img0_p, img1_p, reuse, timestep)

        pred = unpad(pred, orig_h, orig_w)

        if is_numpy:
            return tensor_to_frame(pred)
        return pred

    def unload(self) -> None:
        """Release GMFSS sub-networks from GPU memory."""
        if self._model is not None:
            for sub in (
                self._model.flownet,
                self._model.metricnet,
                self._model.feat_ext,
                self._model.fusionnet,
            ):
                del sub
            self._model = None
        super().unload()
