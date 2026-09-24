"""Tests for video probing and broadcast interlace detection."""

import os
import pytest
from upframe.pipeline.probe import probe_video, detect_interlacing


def test_detect_interlacing_broadcast_clip():
    sample_clip = "30s_slice.mp4"
    if not os.path.exists(sample_clip):
        pytest.skip(f"{sample_clip} not found in test environment.")

    is_interlaced, field_order, details = detect_interlacing(sample_clip, max_frames=30)
    assert is_interlaced is True
    assert field_order == "tff"
    assert details.get("TFF", 0) > 0


def test_probe_video_interlaced_metadata():
    sample_clip = "30s_slice.mp4"
    if not os.path.exists(sample_clip):
        pytest.skip(f"{sample_clip} not found in test environment.")

    metadata = probe_video(sample_clip)
    assert metadata.is_interlaced is True
    assert metadata.field_order == "tff"
    assert metadata.width == 1920
    assert metadata.height == 1080
    assert metadata.target_fps == 50.0
