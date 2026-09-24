"""Configuration structures and presets for the upframe system."""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class PipelineConfig:
    """Complete configuration for an upframing job."""
    model: str = "rife"
    gpus: Optional[List[int]] = None
    chunk_size: int = 1000
    scene_threshold: float = 0.35
    crf: int = 18
    preset: str = "medium"
    use_nvenc: Optional[bool] = None
    bitrate: Optional[str] = None
    temp_dir: Optional[str] = None
    weights: Optional[str] = None
    resume: bool = False
    fp16: bool = True
    tta: bool = False
    max_retries: int = 3
    target_fps: float = 50.0
    workers: Optional[int] = None
    log_dir: str = "upframe_log"
    log_file: Optional[str] = None
    deinterlace: str = "auto"

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Loads configuration from a YAML file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Builds PipelineConfig from dictionary, handling types cleanly."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        
        # Parse gpus if given as comma-separated string
        if "gpus" in filtered and isinstance(filtered["gpus"], str):
            gpus_str = filtered["gpus"].strip()
            if gpus_str.lower() in ["cpu", "none", ""]:
                filtered["gpus"] = []
            else:
                filtered["gpus"] = [int(x.strip()) for x in gpus_str.split(",") if x.strip()]

        return cls(**filtered)


# Broadcast Football default preset
FOOTBALL_PRESET = PipelineConfig(
    model="rife",
    gpus=[0, 1, 2],
    chunk_size=1000,
    scene_threshold=0.35,
    crf=18,
    preset="medium",
    use_nvenc=True,
    bitrate="14M",
    fp16=True,
    target_fps=50.0
)
