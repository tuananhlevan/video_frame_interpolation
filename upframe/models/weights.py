"""Centralized pretrained model weights discovery, management, and auto-download."""

import logging
import os
import shutil
import sys
import zipfile
from typing import Any, Dict, List, Optional
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.expanduser("~/.cache/upframe")

MODEL_WEIGHT_CANDIDATES: Dict[str, List[str]] = {
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
        "backbones/ifrnet/checkpoints/IFRNet/IFRNet_GoPro.pth",
        "ifrnet/checkpoints/IFRNet/IFRNet_Vimeo90K.pth",
        "checkpoints/IFRNet/IFRNet_Vimeo90K.pth",
        os.path.join(CACHE_DIR, "ifrnet", "IFRNet_Vimeo90K.pth"),
    ],
    "ifrnet-gopro": [
        "backbones/ifrnet/checkpoints/IFRNet/IFRNet_GoPro.pth",
        "ifrnet/checkpoints/IFRNet/IFRNet_GoPro.pth",
        "checkpoints/IFRNet/IFRNet_GoPro.pth",
        os.path.join(CACHE_DIR, "ifrnet", "IFRNet_GoPro.pth"),
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
    # EMA-VFI (large)
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
    "interpany": [
        "backbones/interpany_checkpoints/DR-RIFE-vgg/train_sdi_log/flownet_sdi.pkl",
        "backbones/interpany_checkpoints/DR-RIFE-vgg/train_sdi_log",
        "backbones/interpany_checkpoints/DR-RIFE-vgg",
        os.path.join(CACHE_DIR, "interpany", "DR-RIFE-vgg", "train_sdi_log", "flownet_sdi.pkl"),
        os.path.join(CACHE_DIR, "interpany", "DR-RIFE-vgg"),
    ],
    "interpany-vgg": [
        "backbones/interpany_checkpoints/DR-RIFE-vgg/train_sdi_log/flownet_sdi.pkl",
        "backbones/interpany_checkpoints/DR-RIFE-vgg/train_sdi_log",
        "backbones/interpany_checkpoints/DR-RIFE-vgg",
        os.path.join(CACHE_DIR, "interpany", "DR-RIFE-vgg", "train_sdi_log", "flownet_sdi.pkl"),
        os.path.join(CACHE_DIR, "interpany", "DR-RIFE-vgg"),
    ],
    "interpany-pro": [
        "backbones/interpany_checkpoints/DR-RIFE-pro/train_sdi_log/flownet_sdi.pkl",
        "backbones/interpany_checkpoints/DR-RIFE-pro/train_sdi_log",
        "backbones/interpany_checkpoints/DR-RIFE-pro",
        os.path.join(CACHE_DIR, "interpany", "DR-RIFE-pro", "train_sdi_log", "flownet_sdi.pkl"),
        os.path.join(CACHE_DIR, "interpany", "DR-RIFE-pro"),
    ],
}

MODEL_DOWNLOAD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "rife": {
        "type": "direct",
        "url": "https://huggingface.co/daydreamlive/RIFE/resolve/main/flownet.pkl",
        "dest": os.path.join(CACHE_DIR, "rife", "flownet.pkl"),
    },
    "amt": {
        "type": "direct",
        "url": "https://huggingface.co/lalala125/AMT/resolve/main/amt-g.pth",
        "dest": os.path.join(CACHE_DIR, "amt", "amt-g.pth"),
    },
    "amt-g": {
        "type": "direct",
        "url": "https://huggingface.co/lalala125/AMT/resolve/main/amt-g.pth",
        "dest": os.path.join(CACHE_DIR, "amt", "amt-g.pth"),
    },
    "amt-s": {
        "type": "direct",
        "url": "https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth",
        "dest": os.path.join(CACHE_DIR, "amt", "amt-s.pth"),
    },
    "amt-l": {
        "type": "direct",
        "url": "https://huggingface.co/lalala125/AMT/resolve/main/amt-l.pth",
        "dest": os.path.join(CACHE_DIR, "amt", "amt-l.pth"),
    },
    "ifrnet": {
        "type": "direct",
        "url": "https://huggingface.co/pavlichenko/ifrnet_vimeo/resolve/main/IFRNet_Vimeo90K.pth",
        "dest": os.path.join(CACHE_DIR, "ifrnet", "IFRNet_Vimeo90K.pth"),
    },
    "ifrnet-gopro": {
        "type": "direct",
        "url": "https://huggingface.co/pavlichenko/ifrnet_vimeo/resolve/main/IFRNet_Vimeo90K.pth",
        "dest": os.path.join(CACHE_DIR, "ifrnet", "IFRNet_Vimeo90K.pth"),
    },
    "ema-vfi": {
        "type": "direct",
        "url": "https://huggingface.co/xmanifold/emavfi/resolve/main/ours.pkl",
        "dest": os.path.join(CACHE_DIR, "ema_vfi", "ours.pkl"),
    },
    "ema-vfi-s": {
        "type": "direct",
        "url": "https://huggingface.co/xmanifold/emavfi/resolve/main/ours_small.pkl",
        "dest": os.path.join(CACHE_DIR, "ema_vfi", "ours_small.pkl"),
    },
    "gmfss": {
        "type": "multi_direct",
        "dest_dir": os.path.join(CACHE_DIR, "gmfss", "train_log"),
        "files": {
            "flownet.pkl": "https://huggingface.co/NexusAex/GMFSS_Fortuna/resolve/main/GMFSS/train_log/flownet.pkl",
            "metric.pkl": "https://huggingface.co/NexusAex/GMFSS_Fortuna/resolve/main/GMFSS/train_log/metric.pkl",
            "feat.pkl": "https://huggingface.co/NexusAex/GMFSS_Fortuna/resolve/main/GMFSS/train_log/feat.pkl",
            "fusionnet.pkl": "https://huggingface.co/NexusAex/GMFSS_Fortuna/resolve/main/GMFSS/train_log/fusionnet.pkl",
        },
    },
    "interpany": {
        "type": "gdrive",
        "gdrive_id": "14GJSqsX4H5EcQjd-tLb5CM_jzD-577bl",
        "dest_archive": os.path.join(CACHE_DIR, "interpany", "checkpoints.zip"),
        "extract_to": os.path.join(CACHE_DIR, "interpany"),
        "resolved_target": os.path.join(CACHE_DIR, "interpany", "DR-RIFE-vgg", "train_sdi_log", "flownet_sdi.pkl"),
    },
    "interpany-vgg": {
        "type": "gdrive",
        "gdrive_id": "14GJSqsX4H5EcQjd-tLb5CM_jzD-577bl",
        "dest_archive": os.path.join(CACHE_DIR, "interpany", "checkpoints.zip"),
        "extract_to": os.path.join(CACHE_DIR, "interpany"),
        "resolved_target": os.path.join(CACHE_DIR, "interpany", "DR-RIFE-vgg", "train_sdi_log", "flownet_sdi.pkl"),
    },
    "interpany-pro": {
        "type": "gdrive",
        "gdrive_id": "14GJSqsX4H5EcQjd-tLb5CM_jzD-577bl",
        "dest_archive": os.path.join(CACHE_DIR, "interpany", "checkpoints.zip"),
        "extract_to": os.path.join(CACHE_DIR, "interpany"),
        "resolved_target": os.path.join(CACHE_DIR, "interpany", "DR-RIFE-pro", "train_sdi_log", "flownet_sdi.pkl"),
    },
    "film": {
        "type": "gdrive_folder",
        "gdrive_id": "1q8110-qp225asX3DQvZnfLfJPkCHmDpy",
        "dest_dir": os.path.join(CACHE_DIR, "film"),
        "resolved_target": os.path.join(CACHE_DIR, "film", "saved_model"),
    },
}


def download_file_http(url: str, dest_path: str, desc: Optional[str] = None) -> str:
    """Download a file via HTTP streaming with tqdm progress bar and atomic rename."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    temp_path = dest_path + ".tmp"
    headers = {"User-Agent": "UpFrame/1.0"}

    logger.info(f"Downloading {desc or os.path.basename(dest_path)} from: {url}")
    response = requests.get(url, stream=True, timeout=60, headers=headers)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    label = desc or os.path.basename(dest_path)

    with open(temp_path, "wb") as f, tqdm(
        desc=f"Downloading {label}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=True,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    os.replace(temp_path, dest_path)
    logger.info(f"Saved weights to: {dest_path}")
    return dest_path


def download_weight(model_name: str) -> Optional[str]:
    """Automatically downloads weights for the requested model into CACHE_DIR.
    
    Supports single-file HTTP downloads, multi-file HTTP downloads, and Google Drive.
    """
    key = model_name.lower()
    spec = MODEL_DOWNLOAD_REGISTRY.get(key)
    if not spec:
        logger.warning(f"No auto-download source configured for model '{model_name}'.")
        return None

    dl_type = spec.get("type")
    logger.info(f"Auto-downloading weights for '{model_name}'...")

    try:
        if dl_type == "direct":
            url = spec["url"]
            dest = spec["dest"]
            return download_file_http(url, dest, desc=f"{model_name} ({os.path.basename(dest)})")

        elif dl_type == "multi_direct":
            dest_dir = spec["dest_dir"]
            os.makedirs(dest_dir, exist_ok=True)
            files = spec["files"]
            for fname, url in files.items():
                target = os.path.join(dest_dir, fname)
                if not os.path.exists(target) or os.path.getsize(target) == 0:
                    download_file_http(url, target, desc=f"{model_name}/{fname}")
            return dest_dir

        elif dl_type == "gdrive":
            import gdown

            gdrive_id = spec["gdrive_id"]
            dest_archive = spec["dest_archive"]
            extract_to = spec["extract_to"]
            os.makedirs(os.path.dirname(dest_archive), exist_ok=True)

            if not os.path.exists(dest_archive):
                logger.info(f"Downloading Google Drive file id={gdrive_id}...")
                gdown.download(id=gdrive_id, output=dest_archive, quiet=False)

            if os.path.exists(dest_archive) and zipfile.is_zipfile(dest_archive):
                logger.info(f"Extracting {dest_archive} to {extract_to}...")
                with zipfile.ZipFile(dest_archive, "r") as z:
                    z.extractall(extract_to)

            resolved = spec.get("resolved_target")
            if resolved and os.path.exists(resolved):
                return resolved
            return extract_to

        elif dl_type == "gdrive_folder":
            import gdown

            gdrive_id = spec["gdrive_id"]
            dest_dir = spec["dest_dir"]
            os.makedirs(dest_dir, exist_ok=True)
            logger.info(f"Downloading Google Drive folder id={gdrive_id}...")
            gdown.download_folder(id=gdrive_id, output=dest_dir, quiet=False)

            resolved = spec.get("resolved_target")
            if resolved and os.path.exists(resolved):
                return resolved
            return dest_dir

    except Exception as exc:
        logger.error(f"Failed to auto-download weights for '{model_name}': {exc}")
        return None

    return None


def is_valid_candidate(model_name: str, path: str) -> bool:
    """Verifies that a candidate path contains valid, non-empty weights."""
    if not os.path.exists(path):
        return False

    key = model_name.lower()

    if key == "gmfss":
        if os.path.isdir(path):
            required = {"flownet.pkl", "metric.pkl", "feat.pkl", "fusionnet.pkl"}
            return required.issubset(set(os.listdir(path)))
        return False

    if "interpany" in key:
        if os.path.isfile(path):
            return os.path.basename(path) == "flownet_sdi.pkl" and os.path.getsize(path) > 0
        if os.path.isdir(path):
            p1 = os.path.join(path, "flownet_sdi.pkl")
            p2 = os.path.join(path, "train_sdi_log", "flownet_sdi.pkl")
            return (os.path.isfile(p1) and os.path.getsize(p1) > 0) or (os.path.isfile(p2) and os.path.getsize(p2) > 0)
        return False

    if key == "film":
        return os.path.isdir(path) and len(os.listdir(path)) > 0

    return os.path.isfile(path) and os.path.getsize(path) > 0


def resolve_checkpoint(
    model_name: str,
    explicit_path: Optional[str] = None,
    auto_download: bool = True
) -> Optional[str]:
    """Resolves checkpoint path for a given model.
    
    1. If explicit_path is provided and valid, returns it.
    2. Searches candidate locations in cloned repositories and user cache.
    3. If not found and auto_download is True, automatically downloads weights
       into CACHE_DIR and returns the resolved path.
    """
    key = model_name.lower()

    # 1. Explicit path check
    if explicit_path and is_valid_candidate(model_name, explicit_path):
        return os.path.abspath(explicit_path)

    # 2. Local candidates check
    candidates = MODEL_WEIGHT_CANDIDATES.get(key, [])
    for candidate in candidates:
        if is_valid_candidate(model_name, candidate):
            logger.info(f"Discovered {model_name} weights at: {candidate}")
            return os.path.abspath(candidate)

    # 3. Auto-download fallback
    if auto_download:
        downloaded = download_weight(model_name)
        if downloaded and is_valid_candidate(model_name, downloaded):
            logger.info(f"Successfully auto-downloaded and verified weights for {model_name}: {downloaded}")
            return os.path.abspath(downloaded)

    return None
