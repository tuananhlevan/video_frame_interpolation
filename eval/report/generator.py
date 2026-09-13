"""Coordinates evaluation artifact and report generation."""

import os
from typing import Any, Dict, List, Optional
from eval.report.csv_exporter import export_report_csv
from eval.report.html_dashboard import render_html_dashboard
from eval.report.json_exporter import export_report_json
from eval.suspicious.detector import export_suspicious_moments_csv
from eval.types import EvaluationReport


def generate_evaluation_reports(
    report: EvaluationReport,
    eval_dir: str,
    frame_metrics: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, str]:
    """Generates all evaluation files into eval_dir:
    - report.json
    - report.html
    - metrics.csv
    - suspicious/suspicious_timestamps.csv
    
    Returns:
        Dict of filepaths
    """
    os.makedirs(eval_dir, exist_ok=True)
    report.eval_dir = os.path.abspath(eval_dir)

    json_path = os.path.join(eval_dir, "report.json")
    html_path = os.path.join(eval_dir, "report.html")
    csv_path = os.path.join(eval_dir, "metrics.csv")

    export_report_json(report, json_path)
    render_html_dashboard(report, html_path)
    export_report_csv(report, csv_path, frame_metrics=frame_metrics)

    # Suspicious moments CSV
    suspicious_dir = os.path.join(eval_dir, "suspicious")
    suspicious_csv_path = os.path.join(suspicious_dir, "suspicious_timestamps.csv")
    export_suspicious_moments_csv(report.suspicious_moments, suspicious_csv_path)

    return {
        "json": json_path,
        "html": html_path,
        "csv": csv_path,
        "suspicious_csv": suspicious_csv_path
    }
