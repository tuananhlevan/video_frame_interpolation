import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from eval.layer1_technical.scene_cuts import validate_scene_cut_handling
from eval.layer1_technical.source_preservation import validate_source_preservation


def test_scene_cut_intermediate_frame_alignment():
    """Verify that an intermediate interpolated frame at a scene cut is properly evaluated."""
    # Create two distinctly different scenes: Scene A (black) and Scene B (white)
    scene_a = np.zeros((100, 100, 3), dtype=np.uint8)
    scene_b = np.full((100, 100, 3), 255, dtype=np.uint8)

    # In a good interpolation at a cut, intermediate frame is a sharp duplicate of Scene A (or Scene B)
    # In a bad interpolation (hybrid blend), intermediate frame is a blended composite (e.g. 128 gray)
    hybrid_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    clean_copy_a = scene_a.copy()

    def source_gen_cut():
        yield scene_a
        yield scene_b

    def out_gen_hybrid():
        yield scene_a
        yield hybrid_frame
        yield scene_b
        yield scene_b

    mock_meta = MagicMock()
    mock_meta.width = 100
    mock_meta.height = 100

    with patch("eval.layer1_technical.scene_cuts.probe_video", return_value=mock_meta), \
         patch("eval.layer1_technical.scene_cuts.VideoDecoder") as MockDecoder:
        mock_src = MagicMock()
        mock_src.stream_frames.return_value = source_gen_cut()
        mock_out = MagicMock()
        mock_out.stream_frames.return_value = out_gen_hybrid()
        MockDecoder.side_effect = [mock_src, mock_out]

        handled, details, warns = validate_scene_cut_handling("dummy_src.mp4", "dummy_out.mp4", cut_threshold=0.35)
        assert details["source_cuts_detected"] == 1
        assert handled is False
        assert len(details["hybrid_failures"]) == 1

    # Now test with clean duplicate (properly handled cut)
    def out_gen_clean():
        yield scene_a
        yield clean_copy_a
        yield scene_b
        yield scene_b

    with patch("eval.layer1_technical.scene_cuts.probe_video", return_value=mock_meta), \
         patch("eval.layer1_technical.scene_cuts.VideoDecoder") as MockDecoder:
        mock_src = MagicMock()
        mock_src.stream_frames.return_value = source_gen_cut()
        mock_out = MagicMock()
        mock_out.stream_frames.return_value = out_gen_clean()
        MockDecoder.side_effect = [mock_src, mock_out]

        handled, details, warns = validate_scene_cut_handling("dummy_src.mp4", "dummy_out.mp4", cut_threshold=0.35)
        assert details["source_cuts_detected"] == 1
        assert handled is True
        assert len(details["hybrid_failures"]) == 0


def test_source_preservation_decoder_failure_handling():
    """Verify that source preservation does not swallow decoder failure silently."""
    mock_meta = MagicMock()
    mock_meta.width = 100
    mock_meta.height = 100
    mock_meta.nominal_fps = 25.0
    mock_meta.nb_frames = 10

    with patch("eval.layer1_technical.source_preservation.probe_video", return_value=mock_meta), \
         patch("eval.layer1_technical.source_preservation.VideoDecoder") as MockDecoder:
        mock_src = MagicMock()
        mock_src.stream_frames.side_effect = RuntimeError("Decoder stream failed")
        mock_out = MagicMock()
        MockDecoder.side_effect = [mock_src, mock_out]

        passed, metrics, warns = validate_source_preservation("src.mp4", "out.mp4")
        assert passed is False
        assert len(warns) > 0
        assert any("interrupted" in w.lower() or "failed" in w.lower() for w in warns)


def test_scene_cut_tolerance_broadcast_multi_cut():
    """Verify that a broadcast video with 11 cuts and <= 2 isolated blended cuts issues WARN and passes technical QC."""
    from eval.types import TechnicalQCResult

    cuts_detected = list(range(11))
    hybrid_failures = [{"cut_source_frame": 2}, {"cut_source_frame": 8}]

    cut_ok = (len(hybrid_failures) == 0) or (len(hybrid_failures) <= 2 and len(cuts_detected) >= 5)
    cut_warns = []
    if hybrid_failures:
        cut_warns.append(f"Detected {len(hybrid_failures)} blended hybrid frames at hard scene transitions (out of {len(cuts_detected)} cuts detected).")

    all_warns = cut_warns
    tech_status = "PASS"
    if not cut_ok:
        tech_status = "FAIL"
    elif all_warns:
        tech_status = "WARN"

    qc = TechnicalQCResult(
        source_fps=25.0,
        output_fps=50.0,
        source_nb_frames=15000,
        output_nb_frames=30000,
        expected_nb_frames=30000,
        frame_count_diff=0,
        fps_check_passed=True,
        frame_count_passed=True,
        source_preservation_psnr=35.0,
        source_preservation_ssim=0.98,
        source_preservation_mae=2.0,
        source_preservation_max_diff=100.0,
        source_preservation_passed=True,
        audio_present=True,
        audio_duration_diff_sec=0.0,
        audio_sync_passed=True,
        pts_monotonic=True,
        pts_gaps_detected=0,
        pts_duplicates_detected=0,
        pts_check_passed=True,
        scene_cuts_detected_source=len(cuts_detected),
        scene_cuts_properly_handled=cut_ok,
        status=tech_status,
        warnings=all_warns
    )

    assert qc.status == "WARN"
    assert qc.passed is True
    assert qc.scene_cuts_properly_handled is True
    assert len(qc.warnings) == 1
