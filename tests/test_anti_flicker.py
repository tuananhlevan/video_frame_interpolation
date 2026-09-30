import numpy as np
import pytest
from upframe.pipeline.anti_flicker import TemporalAntiFlicker
from upframe.cli.args import parse_cli_args


def test_anti_flicker_suppresses_white_pop():
    filter_mod = TemporalAntiFlicker(radius=3, margin=15, protect_ball=False)
    
    # Dark scene (e.g. black letterbox or dark crowd)
    f0 = np.zeros((100, 100, 3), dtype=np.uint8)
    f1 = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # AI synthesis produces a single-frame white pop / firefly spike
    f_inter = np.zeros((100, 100, 3), dtype=np.uint8)
    f_inter[50, 50] = [220, 220, 220]
    f_inter[51, 50] = [190, 180, 200]
    
    cleaned = filter_mod.process(f0, f_inter, f1)
    
    # Spikes should be clamped to margin (15)
    assert cleaned[50, 50, 0] <= 15
    assert cleaned[50, 50, 1] <= 15
    assert cleaned[50, 50, 2] <= 15
    assert cleaned[51, 50, 0] <= 15


def test_anti_flicker_preserves_legitimate_motion():
    filter_mod = TemporalAntiFlicker(radius=3, margin=15, protect_ball=False)
    
    # Smooth gradient motion between f0 and f1
    f0 = np.full((100, 100, 3), 100, dtype=np.uint8)
    f1 = np.full((100, 100, 3), 110, dtype=np.uint8)
    f_inter = np.full((100, 100, 3), 105, dtype=np.uint8)
    
    cleaned = filter_mod.process(f0, f_inter, f1)
    
    # Should be completely unchanged
    np.testing.assert_array_equal(cleaned, f_inter)


def test_anti_flicker_cli_arg():
    _, _, config = parse_cli_args([
        "input.mp4", "output.mp4",
        "--anti-flicker"
    ])
    assert config.anti_flicker is True
