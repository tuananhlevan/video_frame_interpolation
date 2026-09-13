"""Exports EvaluationReport to machine-readable JSON."""

import json
import os
from eval.types import EvaluationReport


def export_report_json(report: EvaluationReport, output_path: str) -> str:
    """Saves the complete evaluation report as formatted JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    return output_path
