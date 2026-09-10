"""Centralized pretrained model weights discovery and management."""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.expanduser("~/.cache/upframe")

MODEL_WEIGHT_CANDIDATES = {
    "rife": [
        "backbones/rife/train_log/flownet.pkl",
        "backbones/rife/flownet.pkl",
        "rife/train_log/flownet.pkl",
        "rife/flownet.pkl",
        "train_log/flownet.pkl",
        "weights/rife.pth",
        os.path.join(CACHE_DIR, "rife", "flownet.pkl"),
    ],
    "amt": [
        "backbones/amt/pretrained/amt-s.pth",
        "amt/pretrained/amt-s.pth",
        "pretrained/amt-s.pth",
        "weights/amt-s.pth",
        os.path.join(CACHE_DIR, "amt", "amt-s.pth"),
    ],
    "ifrnet": [
        "backbones/ifrnet/checkpoints/IFRNet/IFRNet_Vimeo90K.pth",
        "ifrnet/checkpoints/IFRNet/IFRNet_Vimeo90K.pth",
        "checkpoints/IFRNet/IFRNet_Vimeo90K.pth",
        "checkpoints/IFRNet_Vimeo90K.pth",
        os.path.join(CACHE_DIR, "ifrnet", "IFRNet_Vimeo90K.pth"),
    ],
    "film": [
        "backbones/film/pretrained_models/film_net/Style/saved_model",
        "film/pretrained_models/film_net/Style/saved_model",
        "pretrained_models/film_net/Style/saved_model",
        os.path.join(CACHE_DIR, "film", "saved_model"),
    ]
}


def resolve_checkpoint(model_name: str, explicit_path: Optional[str] = None) -> Optional[str]:
    """Resolves checkpoint path for a given model.
    
    If explicit_path is provided and exists, returns it.
    Otherwise searches candidate locations in cloned repositories and user cache.
    """
    if explicit_path and os.path.exists(explicit_path):
        return os.path.abspath(explicit_path)

    key = model_name.lower()
    candidates = MODEL_WEIGHT_CANDIDATES.get(key, [])
    for candidate in candidates:
        if os.path.exists(candidate):
            logger.info(f"Discovered {model_name} weights at: {candidate}")
            return os.path.abspath(candidate)

    return None
