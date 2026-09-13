"""HTML Dashboard generator for comprehensive evaluation reporting."""

import html
import os
from eval.types import EvaluationReport


def render_html_dashboard(report: EvaluationReport, output_path: str) -> str:
    """Renders a standalone interactive HTML dashboard."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    tech = report.technical_qc
    temp = report.temporal_qc
    fb = report.football_qc
    art = report.artifact_diff
    card = report.scorecard
    perf = report.performance_qc

    # Status pill color helper
    def status_pill(status: str) -> str:
        s = status.upper()
        if "PASS" in s or "EXCELLENT" in s or "CANDIDATE" in s:
            return f'<span class="badge badge-pass">{html.escape(status)}</span>'
        elif "WARN" in s or "ACCEPTABLE" in s or "GOOD" in s:
            return f'<span class="badge badge-warn">{html.escape(status)}</span>'
        else:
            return f'<span class="badge badge-fail">{html.escape(status)}</span>'

    # Progress bar helper for 1-5 scores
    def score_bar(score: float) -> str:
        pct = (score / 5.0) * 100.0
        color = "#2ecc71" if score >= 4.0 else ("#f1c40f" if score >= 3.0 else "#e74c3c")
        return f'''
        <div class="progress-container">
            <div class="progress-bar" style="width: {pct:.1f}%; background-color: {color};"></div>
            <span class="progress-text">{score:.1f} / 5.0</span>
        </div>
        '''

    # Suspicious moments table rows
    suspicious_rows = ""
    for sm in report.suspicious_moments[:20]:
        suspicious_rows += f"""
        <tr>
            <td class="font-mono">{html.escape(sm.timestamp_str)}</td>
            <td>{sm.frame_index:,}</td>
            <td><span class="badge badge-warn">{html.escape(sm.anomaly_type)}</span></td>
            <td>{sm.severity} / 5</td>
            <td>{sm.anomaly_score:.1f}</td>
            <td>{html.escape(sm.description)}</td>
        </tr>
        """
    if not suspicious_rows:
        suspicious_rows = "<tr><td colspan='6' class='text-muted text-center'>No suspicious moments flagged.</td></tr>"

    # HTML template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upframe Evaluation Report - {html.escape(report.model_name.upper())}</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: #1e293b;
            --border: #334155;
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --accent: #38bdf8;
            --pass: #10b981;
            --warn: #f59e0b;
            --fail: #ef4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.5;
            padding: 24px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }}
        h1 {{ font-size: 28px; font-weight: 700; color: var(--accent); }}
        .header-meta {{ font-size: 14px; color: var(--text-dim); }}
        
        .grid-scorecard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }}
        .card-title {{ font-size: 13px; font-weight: 600; text-transform: uppercase; color: var(--text-dim); margin-bottom: 8px; }}
        .card-value {{ font-size: 32px; font-weight: 700; color: var(--text); }}
        .card-sub {{ font-size: 13px; color: var(--text-dim); margin-top: 4px; }}

        .section {{ margin-bottom: 32px; }}
        .section-title {{ font-size: 18px; font-weight: 600; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }}
        
        .table-card {{ background-color: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; text-align: left; }}
        th {{ background-color: rgba(51, 65, 85, 0.5); padding: 12px 16px; color: var(--text-dim); font-weight: 600; }}
        td {{ padding: 12px 16px; border-bottom: 1px solid var(--border); }}
        tr:last-child td {{ border-bottom: none; }}

        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; text-transform: uppercase; }}
        .badge-pass {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
        .badge-warn {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
        .badge-fail {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}

        .progress-container {{ display: flex; align-items: center; gap: 12px; }}
        .progress-bar {{ height: 8px; border-radius: 4px; }}
        .progress-text {{ font-size: 13px; font-weight: 600; min-width: 60px; }}
        .font-mono {{ font-family: monospace; }}
        .text-center {{ text-align: center; }}
        .text-muted {{ color: var(--text-dim); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>UPFRAME EVALUATION REPORT</h1>
                <div class="header-meta">
                    Model: <strong>{html.escape(report.model_name.upper())}</strong> &bull; 
                    Input: {html.escape(os.path.basename(report.source_file))} &bull; 
                    Output: {html.escape(os.path.basename(report.output_file))}
                </div>
            </div>
            <div>
                {status_pill(card.recommendation)}
            </div>
        </header>

        <!-- High-level scorecard -->
        <div class="grid-scorecard">
            <div class="card">
                <div class="card-title">Football Quality Score</div>
                <div class="card-value">{card.quality_score:.1f} <span style="font-size: 16px; color: var(--text-dim);">/ 10</span></div>
                <div class="card-sub">{status_pill(card.quality_status)}</div>
            </div>
            <div class="card">
                <div class="card-title">Performance Score</div>
                <div class="card-value">{card.performance_score:.1f} <span style="font-size: 16px; color: var(--text-dim);">/ 10</span></div>
                <div class="card-sub">{status_pill(card.performance_status)}</div>
            </div>
            <div class="card">
                <div class="card-title">Realtime Factor (RTF)</div>
                <div class="card-value">{card.realtime_factor:.2f}x</div>
                <div class="card-sub">Target: 1.0x &ndash; 1.3x</div>
            </div>
            <div class="card">
                <div class="card-title">Perceptual Human MOS</div>
                <div class="card-value">{card.human_mos:.2f} <span style="font-size: 16px; color: var(--text-dim);">/ 5</span></div>
                <div class="card-sub">Responses: {report.perceptual_qc.survey_responses_count}</div>
            </div>
        </div>

        <!-- Layer 1 Technical Integrity -->
        <div class="section">
            <div class="section-title">
                <span>Layer 1: Technical &amp; Pipeline Integrity</span>
                {status_pill(tech.status)}
            </div>
            <div class="table-card">
                <table>
                    <thead>
                        <tr>
                            <th>Check</th>
                            <th>Requirement</th>
                            <th>Observed Value</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Source-Frame Preservation</td>
                            <td>Original frames (0, 2, 4...) preserved intact</td>
                            <td>PSNR: {tech.source_preservation_psnr:.2f} dB, SSIM: {tech.source_preservation_ssim:.4f}, MAE: {tech.source_preservation_mae:.2f} px</td>
                            <td>{status_pill('PASS' if tech.source_preservation_passed else 'FAIL')}</td>
                        </tr>
                        <tr>
                            <td>Frame Rate (FPS)</td>
                            <td>Nominal ~25 fps &rarr; ~50 fps</td>
                            <td>Source: {tech.source_fps:.2f} &rarr; Output: {tech.output_fps:.2f} fps</td>
                            <td>{status_pill('PASS' if tech.fps_check_passed else 'FAIL')}</td>
                        </tr>
                        <tr>
                            <td>Frame Count</td>
                            <td>2x temporal expansion (2*N - 1)</td>
                            <td>Expected ~{tech.expected_nb_frames:,}, Actual: {tech.output_nb_frames:,} (diff: {tech.frame_count_diff})</td>
                            <td>{status_pill('PASS' if tech.frame_count_passed else 'FAIL')}</td>
                        </tr>
                        <tr>
                            <td>Audio Stream &amp; Sync</td>
                            <td>Audio preserved without drift</td>
                            <td>Present: {'Yes' if tech.audio_present else 'No'}, Duration delta: {tech.audio_duration_diff_sec:.3f}s</td>
                            <td>{status_pill('PASS' if tech.audio_sync_passed else 'FAIL')}</td>
                        </tr>
                        <tr>
                            <td>Timestamp (PTS) Continuity</td>
                            <td>Monotonic, no gaps or duplicate PTS</td>
                            <td>Monotonic: {'Yes' if tech.pts_monotonic else 'No'}, Duplicates: {tech.pts_duplicates_detected}, Gaps: {tech.pts_gaps_detected}</td>
                            <td>{status_pill('PASS' if tech.pts_check_passed else 'FAIL')}</td>
                        </tr>
                        <tr>
                            <td>Scene Cut Guard</td>
                            <td>No synthetic hybrid frames across camera cuts</td>
                            <td>Cuts detected: {tech.scene_cuts_detected_source}</td>
                            <td>{status_pill('PASS' if tech.scene_cuts_properly_handled else 'FAIL')}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Layer 2 & 3: Football QC Breakdown & Temporal Metrics -->
        <div class="section">
            <div class="section-title">Layer 3: Football-Specific Quality Evaluation (1-5 Scale)</div>
            <div class="table-card">
                <table>
                    <thead>
                        <tr>
                            <th>Category</th>
                            <th>Weight</th>
                            <th>Score &amp; Visual Bar</th>
                            <th>Key Diagnostics</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Player Integrity</strong></td>
                            <td>15%</td>
                            <td>{score_bar(fb.player_integrity_score)}</td>
                            <td>Silhouette solidity, limbs, natural body shape</td>
                        </tr>
                        <tr>
                            <td><strong>Ball Integrity</strong></td>
                            <td>15%</td>
                            <td>{score_bar(fb.ball_integrity_score)}</td>
                            <td>Teleports: {fb.detected_ball_teleportations}, Duplicate balls: {fb.detected_duplicate_balls}</td>
                        </tr>
                        <tr>
                            <td><strong>Temporal Stability</strong></td>
                            <td>15%</td>
                            <td>{score_bar(1.0 + 4.0 * temp.motion_smoothness_score)}</td>
                            <td>Smoothness: {temp.motion_smoothness_score:.3f}, Flicker score: {temp.temporal_flicker_score:.3f}</td>
                        </tr>
                        <tr>
                            <td><strong>Occlusion Handling</strong></td>
                            <td>10%</td>
                            <td>{score_bar(fb.occlusion_handling_score)}</td>
                            <td>Player-player crossings, ghost limbs, merged silhouettes</td>
                        </tr>
                        <tr>
                            <td><strong>Camera Motion</strong></td>
                            <td>10%</td>
                            <td>{score_bar(fb.camera_motion_score)}</td>
                            <td>Smoothness of broadcast pans, tilts, and tracking</td>
                        </tr>
                        <tr>
                            <td><strong>Pitch Geometry</strong></td>
                            <td>5%</td>
                            <td>{score_bar(fb.pitch_geometry_score)}</td>
                            <td>Touchlines &amp; penalty boxes curvature wobble</td>
                        </tr>
                        <tr>
                            <td><strong>Broadcast Graphics</strong></td>
                            <td>10%</td>
                            <td>{score_bar(fb.broadcast_graphics_score)}</td>
                            <td>Scorebug &amp; watermark stability, jitter: {'Detected' if fb.graphics_jitter_detected else 'None'}</td>
                        </tr>
                        <tr>
                            <td><strong>Goal &amp; Net</strong></td>
                            <td>5%</td>
                            <td>{score_bar(fb.goal_net_score)}</td>
                            <td>Posts, crossbar, and thin net mesh distortion</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Artifact Differential (Added vs Source) -->
        <div class="section">
            <div class="section-title">Source vs. Output Artifact Differential (&Delta; Added Artifacts)</div>
            <div class="table-card">
                <table>
                    <thead>
                        <tr>
                            <th>Artifact Class</th>
                            <th>Source Baseline</th>
                            <th>Output Level</th>
                            <th>&Delta; Added by VFI</th>
                            <th>Severity Level</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Ghosting / Phantom Trails</td>
                            <td>{art.ghosting.source_level * 100:.2f}%</td>
                            <td>{art.ghosting.output_level * 100:.2f}%</td>
                            <td><strong>+{art.ghosting.added_level * 100:.2f}%</strong></td>
                            <td>Level {art.ghosting.severity_level} / 5</td>
                        </tr>
                        <tr>
                            <td>Double Contour / Edge Ringing</td>
                            <td>{art.double_contour.source_level * 100:.2f}%</td>
                            <td>{art.double_contour.output_level * 100:.2f}%</td>
                            <td><strong>+{art.double_contour.added_level * 100:.2f}%</strong></td>
                            <td>Level {art.double_contour.severity_level} / 5</td>
                        </tr>
                        <tr>
                            <td>Edge Tearing / Serrated Boundaries</td>
                            <td>{art.edge_tearing.source_level * 100:.2f}%</td>
                            <td>{art.edge_tearing.output_level * 100:.2f}%</td>
                            <td><strong>+{art.edge_tearing.added_level * 100:.2f}%</strong></td>
                            <td>Level {art.edge_tearing.severity_level} / 5</td>
                        </tr>
                        <tr>
                            <td>Object Deformation</td>
                            <td>{art.deformation.source_level * 100:.2f}%</td>
                            <td>{art.deformation.output_level * 100:.2f}%</td>
                            <td><strong>+{art.deformation.added_level * 100:.2f}%</strong></td>
                            <td>Level {art.deformation.severity_level} / 5</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Suspicious Moments -->
        <div class="section">
            <div class="section-title">Automated Bad-Moment &amp; Suspicious-Timestamp Detection</div>
            <div class="table-card">
                <table>
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Frame Index</th>
                            <th>Anomaly Type</th>
                            <th>Severity</th>
                            <th>Score</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        {suspicious_rows}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
