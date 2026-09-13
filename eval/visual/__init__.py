"""Visual Inspection and Material Generation tools."""

from eval.visual.extractor import extract_injected_frames
from eval.visual.slowmo import generate_slowmo_clip
from eval.visual.diff_maps import generate_difference_heatmap
from eval.visual.side_by_side import generate_side_by_side_video

__all__ = [
    "extract_injected_frames",
    "generate_slowmo_clip",
    "generate_difference_heatmap",
    "generate_side_by_side_video"
]
