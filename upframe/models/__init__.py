"""VFI Model registry, adapters, and weight resolution."""

from upframe.models.base import BaseVFIModel, ModelRegistry
from upframe.models.weights import resolve_checkpoint
from upframe.models.blend import BlendModel
from upframe.models.rife import RIFEModel
from upframe.models.amt import AMTModel
from upframe.models.ifrnet import IFRNetModel
from upframe.models.film import FILMModel
from upframe.models.gmfss import GMFSSModel
from upframe.models.ema_vfi import _EMAVFILarge as EMAVFIModel, _EMAVFISmall as EMAVFISmallModel
from upframe.models.interpany import InterpAnyModel
from upframe.models.bwdif import BWDIFModel

__all__ = [
    "BaseVFIModel",
    "ModelRegistry",
    "resolve_checkpoint",
    "BlendModel",
    "RIFEModel",
    "AMTModel",
    "IFRNetModel",
    "FILMModel",
    "GMFSSModel",
    "EMAVFIModel",
    "EMAVFISmallModel",
    "InterpAnyModel",
    "BWDIFModel",
]
