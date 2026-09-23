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
        "backbones/amt/ckpt/amt-g.pth",
        "backbones/amt/ckpt/amt-s.pth",
        "backbones/amt/ckpt/amt-l.pth",
        "backbones/amt/pretrained/amt-g.pth",
        "backbones/amt/pretrained/amt-s.pth",
        "amt/ckpt/amt-g.pth",
        "amt/ckpt/amt-s.pth",
        "amt/pretrained/amt-g.pth",
        "amt/pretrained/amt-s.pth",
        "ckpt/amt-g.pth",
        "ckpt/amt-s.pth",
        "pretrained/amt-g.pth",
        "pretrained/amt-s.pth",
        "weights/amt-g.pth",
        "weights/amt-s.pth",
        os.path.join(CACHE_DIR, "amt", "amt-g.pth"),
        os.path.join(CACHE_DIR, "amt", "amt-s.pth"),
    ],
    "amt-s": [
        "backbones/amt/ckpt/amt-s.pth",
        "backbones/amt/pretrained/amt-s.pth",
        "amt/ckpt/amt-s.pth",
        "amt/pretrained/amt-s.pth",
        "ckpt/amt-s.pth",
        "pretrained/amt-s.pth",
        "weights/amt-s.pth",
        os.path.join(CACHE_DIR, "amt", "amt-s.pth"),
    ],
    "amt-l": [
        "backbones/amt/ckpt/amt-l.pth",
        "backbones/amt/pretrained/amt-l.pth",
        "amt/ckpt/amt-l.pth",
        "amt/pretrained/amt-l.pth",
        "ckpt/amt-l.pth",
        "pretrained/amt-l.pth",
        "weights/amt-l.pth",
        os.path.join(CACHE_DIR, "amt", "amt-l.pth"),
    ],
    "amt-g": [
        "backbones/amt/ckpt/amt-g.pth",
        "backbones/amt/pretrained/amt-g.pth",
        "amt/ckpt/amt-g.pth",
        "amt/pretrained/amt-g.pth",
        "ckpt/amt-g.pth",
        "pretrained/amt-g.pth",
        "weights/amt-g.pth",
        os.path.join(CACHE_DIR, "amt", "amt-g.pth"),
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
    ],
    # GMFSS Fortuna — checkpoint is a *directory* containing 4 pkl files
    "gmfss": [
        "backbones/gmfss/train_log",
        "gmfss/train_log",
        "train_log",
        os.path.join(CACHE_DIR, "gmfss", "train_log"),
    ],
    # EMA-VFI (large) — single pkl checkpoint
    "ema-vfi": [
        "backbones/ema_vfi/ckpt/ours.pkl",
        "backbones/ema_vfi/ckpt/ema-vfi-l.pkl",
        "ema_vfi/ckpt/ours.pkl",
        "ckpt/ours.pkl",
        os.path.join(CACHE_DIR, "ema_vfi", "ours.pkl"),
    ],
    # EMA-VFI (small)
    "ema-vfi-s": [
        "backbones/ema_vfi/ckpt/ours_small.pkl",
        "backbones/ema_vfi/ckpt/ema-vfi-s.pkl",
        "ema_vfi/ckpt/ours_small.pkl",
        "ckpt/ours_small.pkl",
        os.path.join(CACHE_DIR, "ema_vfi", "ours_small.pkl"),
    ],
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
