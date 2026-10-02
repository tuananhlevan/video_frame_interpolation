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


def test_anti_flicker_process_tensor_matches_numpy():
    import torch
    from upframe.utils.tensor import frames_to_tensor_batch, tensor_to_frames_batch

    filter_mod = TemporalAntiFlicker(radius=3, margin=15, protect_ball=False)
    np.random.seed(42)
    f0 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    f1 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    f_inter = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    # CPU numpy baseline
    cpu_cleaned = filter_mod.process(f0, f_inter, f1)

    # Tensor vectorized process
    dev = torch.device("cpu")
    t0 = frames_to_tensor_batch([f0], dev)
    t1 = frames_to_tensor_batch([f1], dev)
    t_inter = frames_to_tensor_batch([f_inter], dev)

    t_cleaned = filter_mod.process_tensor(t0, t_inter, t1)
    tensor_cleaned = tensor_to_frames_batch(t_cleaned)[0]

    # Verify equivalence within 1 LSB (due to float32 1/255.0 vs uint8 quantization)
    np.testing.assert_allclose(cpu_cleaned, tensor_cleaned, atol=1)


def test_anti_flicker_default_margin():
    filter_mod = TemporalAntiFlicker()
    assert filter_mod.margin == 10
    assert filter_mod.radius == 10
    assert filter_mod.outlier_margin == 20


def test_anti_flicker_suppresses_severe_outliers():
    """Verify that artificial ghost blobs or severe firefly sparks are restored to clean background blend."""
    filter_mod = TemporalAntiFlicker(radius=10, margin=10, outlier_margin=20, protect_ball=False)
    # Dark green pitch background [30, 120, 80]
    f0 = np.full((100, 100, 3), [30, 120, 80], dtype=np.uint8)
    f1 = np.full((100, 100, 3), [30, 120, 80], dtype=np.uint8)
    
    # Severe synthesis ghost blob / white flash in intermediate frame at (50, 50)
    f_inter = np.full((100, 100, 3), [30, 120, 80], dtype=np.uint8)
    f_inter[45:55, 45:55] = [200, 240, 200]  # Bright flash blob
    
    cleaned = filter_mod.process(f0, f_inter, f1)
    
    # Severe outlier blob must be completely replaced by ground truth background [30, 120, 80]
    assert np.all(cleaned[45:55, 45:55] == [30, 120, 80])


def test_anti_flicker_preserves_fast_moving_socks():
    filter_mod = TemporalAntiFlicker(radius=16, margin=15, protect_ball=False)
    # Green pitch background
    f0 = np.full((100, 100, 3), [40, 120, 40], dtype=np.uint8)
    f1 = np.full((100, 100, 3), [40, 120, 40], dtype=np.uint8)
    # White sock at x=30 in f0
    f0[30:70, 28:34] = [240, 240, 240]
    # White sock moves 20 pixels to x=50 in f1
    f1[30:70, 48:54] = [240, 240, 240]
    # Intermediate frame at t=0.5: white sock at x=40 (moved 10px from f0, 10px from f1)
    f_inter = np.full((100, 100, 3), [40, 120, 40], dtype=np.uint8)
    f_inter[30:70, 38:44] = [240, 240, 240]

    cleaned = filter_mod.process(f0, f_inter, f1)
    # Moving white sock must remain pure white (>200), not shredded into green grass (<60)
    assert np.all(cleaned[30:70, 38:44] >= 200)


def test_anti_flicker_rejects_mismatched_ball_candidates():
    # If candidates in F0 and F1 differ greatly in radius, protect_ball should not carve an un-clamped hole
    filter_mod = TemporalAntiFlicker(protect_ball=True)
    f0 = np.zeros((200, 200, 3), dtype=np.uint8)
    f1 = np.zeros((200, 200, 3), dtype=np.uint8)
    f_inter = np.zeros((200, 200, 3), dtype=np.uint8)
    # Firefly in the middle
    f_inter[100, 100] = [255, 255, 255]

    cleaned = filter_mod.process(f0, f_inter, f1)
    # Since no consistent ball exists, firefly should be clamped to margin (12)
    assert cleaned[100, 100, 0] <= 12
    assert cleaned[100, 100, 1] <= 12
    assert cleaned[100, 100, 2] <= 12


def test_anti_flicker_normalizes_luminance():
    filter_mod = TemporalAntiFlicker(radius=3, margin=15, protect_ball=False, normalize_luminance=True)
    f0 = np.full((50, 50, 3), 100, dtype=np.uint8)
    f1 = np.full((50, 50, 3), 100, dtype=np.uint8)
    # Synthesized frame with depressed luminance
    f_inter = np.full((50, 50, 3), 90, dtype=np.uint8)

    cleaned = filter_mod.process(f0, f_inter, f1)
    # Target luminance is 100. Scale = 100 / 90 = 1.10 (max clamp 1.10).
    # 90 * 1.10 = 99
    assert cleaned.mean() > 95.0


def test_anti_flicker_process_tensor_preserves_fast_flying_ball():
    """Verify that process_tensor on GPU/CPU preserves high-speed balls (80px motion) without erasing them."""
    import torch
    import cv2
    from upframe.utils.tensor import frames_to_tensor_batch, tensor_to_frames_batch

    filter_protected = TemporalAntiFlicker(radius=10, margin=10, protect_ball=True)
    filter_unprotected = TemporalAntiFlicker(radius=10, margin=10, protect_ball=False)

    f0 = np.full((200, 200, 3), [30, 120, 80], dtype=np.uint8)
    f1 = np.full((200, 200, 3), [30, 120, 80], dtype=np.uint8)
    f_inter = np.full((200, 200, 3), [30, 120, 80], dtype=np.uint8)

    # Ball moves 80px: from x=50 in F0 to x=130 in F1, interpolated at x=90 in F_inter
    cv2.circle(f0, (50, 100), 6, (255, 255, 255), -1)
    cv2.circle(f1, (130, 100), 6, (255, 255, 255), -1)
    cv2.circle(f_inter, (90, 100), 6, (255, 255, 255), -1)

    dev = torch.device("cpu")
    t0 = frames_to_tensor_batch([f0], dev)
    t1 = frames_to_tensor_batch([f1], dev)
    ti = frames_to_tensor_batch([f_inter], dev)

    # With protect_ball=True: Ball must be preserved (> 220)
    t_clean_prot = filter_protected.process_tensor(t0, ti, t1)
    np_clean_prot = tensor_to_frames_batch(t_clean_prot)[0]
    assert np_clean_prot[100, 90, 0] > 220
    assert np_clean_prot[100, 90, 1] > 220
    assert np_clean_prot[100, 90, 2] > 220

    # With protect_ball=False: Radius 10 cannot reach 40px away, so ball is clamped to green grass (< 100)
    t_clean_unprot = filter_unprotected.process_tensor(t0, ti, t1)
    np_clean_unprot = tensor_to_frames_batch(t_clean_unprot)[0]
    assert np_clean_unprot[100, 90, 0] < 100


def test_anti_flicker_preserves_large_fast_moving_objects():
    """Verify that large semantic objects moving at high speed (>150px) are preserved without black holes."""
    import torch
    from upframe.utils.tensor import frames_to_tensor_batch, tensor_to_frames_batch

    filter_mod = TemporalAntiFlicker(radius=10, margin=10, outlier_margin=20, protect_ball=False, protect_large_motion=True)

    # Dark background scene [35, 35, 35]
    f0 = np.full((200, 300, 3), [35, 35, 35], dtype=np.uint8)
    f1 = np.full((200, 300, 3), [35, 35, 35], dtype=np.uint8)
    f_inter = np.full((200, 300, 3), [35, 35, 35], dtype=np.uint8)

    # Golden jersey number 40x40 px moves 150px: from x=30 in F0 to x=180 in F1
    # Interpolated midpoint at x=105 in F_inter (75px away from F0 and F1, far exceeding radius 10)
    f0[50:90, 30:70] = [120, 180, 220]
    f1[50:90, 180:220] = [120, 180, 220]
    f_inter[50:90, 105:145] = [120, 180, 220]

    # CPU numpy test:
    cleaned_cpu = filter_mod.process(f0, f_inter, f1)
    # The 40x40 object must be preserved (>180 in green/red), not hollowed out to 35
    assert np.all(cleaned_cpu[55:85, 110:140, 2] >= 180)

    # GPU tensor test:
    dev = torch.device("cpu")
    t0 = frames_to_tensor_batch([f0], dev)
    t1 = frames_to_tensor_batch([f1], dev)
    ti = frames_to_tensor_batch([f_inter], dev)

    t_cleaned = filter_mod.process_tensor(t0, ti, t1)
    cleaned_tensor = tensor_to_frames_batch(t_cleaned)[0]
    assert np.all(cleaned_tensor[55:85, 110:140, 2] >= 180)


def test_anti_flicker_simultaneous_firefly_and_large_motion():
    """Verify that small fireflies are eliminated while large fast motion in the same frame is preserved."""
    import torch
    from upframe.utils.tensor import frames_to_tensor_batch, tensor_to_frames_batch

    filter_mod = TemporalAntiFlicker(radius=10, margin=10, outlier_margin=20, protect_ball=False, protect_large_motion=True)

    f0 = np.full((200, 300, 3), [35, 35, 35], dtype=np.uint8)
    f1 = np.full((200, 300, 3), [35, 35, 35], dtype=np.uint8)
    f_inter = np.full((200, 300, 3), [35, 35, 35], dtype=np.uint8)

    # 1. Large moving object at (50:90, 105:145)
    f0[50:90, 30:70] = [120, 180, 220]
    f1[50:90, 180:220] = [120, 180, 220]
    f_inter[50:90, 105:145] = [120, 180, 220]

    # 2. Isolated single-frame firefly noise (2x2 px) at (150:152, 50:52)
    f_inter[150:152, 50:52] = [255, 255, 255]

    dev = torch.device("cpu")
    t0 = frames_to_tensor_batch([f0], dev)
    t1 = frames_to_tensor_batch([f1], dev)
    ti = frames_to_tensor_batch([f_inter], dev)

    t_cleaned = filter_mod.process_tensor(t0, ti, t1)
    cleaned = tensor_to_frames_batch(t_cleaned)[0]

    # Large object must be preserved
    assert np.all(cleaned[55:85, 110:140, 2] >= 180)

    # Firefly must be suppressed to background envelope (<= 45)
    assert np.all(cleaned[150:152, 50:52] <= 45)



