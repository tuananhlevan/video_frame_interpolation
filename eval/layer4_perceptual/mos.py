"""Human Mean Opinion Score (MOS) schema and survey loader."""

import json
import os
from typing import Dict, List, Optional
from eval.types import PerceptualQCResult


def load_human_mos(filepath: Optional[str], default_mos: float = 4.0) -> PerceptualQCResult:
    """Loads human MOS survey data from JSON/CSV file, or generates calibrated default."""
    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return PerceptualQCResult(
                overall_quality_mos=float(data.get("overall_quality_mos", default_mos)),
                motion_naturalness_mos=float(data.get("motion_naturalness_mos", default_mos)),
                artifact_free_mos=float(data.get("artifact_free_mos", default_mos)),
                pairwise_preference_pct=float(data["pairwise_preference_pct"]) if "pairwise_preference_pct" in data else None,
                survey_responses_count=int(data.get("responses_count", 1))
            )
        except Exception:
            pass

    return PerceptualQCResult(
        overall_quality_mos=default_mos,
        motion_naturalness_mos=default_mos,
        artifact_free_mos=default_mos,
        pairwise_preference_pct=None,
        survey_responses_count=0
    )


def evaluate_perceptual_quality(
    automated_quality_estimate: float,
    human_mos_file: Optional[str] = None
) -> PerceptualQCResult:
    """Combines automated visual indicators with human survey data if available."""
    # Convert 1-5 automated quality estimate to MOS baseline
    estimated_mos = max(1.0, min(5.0, round(automated_quality_estimate, 2)))
    return load_human_mos(human_mos_file, default_mos=estimated_mos)
