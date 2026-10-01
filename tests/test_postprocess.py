import os
import tempfile
import cv2
import numpy as np
import pytest
from upframe.pipeline.postprocess import deflicker_video


def test_deflicker_video_e2e():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_vid = os.path.join(tmpdir, "input.mp4")
        output_vid = os.path.join(tmpdir, "output.mp4")

        # Create a small 5-frame 64x64 video at 50fps
        # Frame 0: Dark background [20, 20, 20]
        # Frame 1: Outlier artifact [250, 250, 250] at (32, 32)
        # Frame 2: Dark background [22, 22, 22]
        # Frame 3: Normal motion [24, 24, 24]
        # Frame 4: Dark background [26, 26, 26]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(input_vid, fourcc, 50.0, (64, 64))
        
        f0 = np.full((64, 64, 3), 20, dtype=np.uint8)
        f1 = np.full((64, 64, 3), 21, dtype=np.uint8)
        f1[30:35, 30:35] = 250  # bright glitch
        f2 = np.full((64, 64, 3), 22, dtype=np.uint8)
        f3 = np.full((64, 64, 3), 24, dtype=np.uint8)
        f4 = np.full((64, 64, 3), 26, dtype=np.uint8)

        for f in [f0, f1, f2, f3, f4]:
            writer.write(f)
        writer.release()

        # Run deflicker
        success = deflicker_video(
            input_path=input_vid,
            output_path=output_vid,
            radius=4,
            margin=10,
            outlier_margin=20,
            device="cpu",
            use_nvenc=False,
            batch_size=2
        )
        assert success is True
        assert os.path.exists(output_vid)

        # Verify output frames
        cap = cv2.VideoCapture(output_vid)
        assert cap.isOpened()
        frames = []
        ret, frame = cap.read()
        while ret:
            frames.append(frame)
            ret, frame = cap.read()
        cap.release()

        assert len(frames) == 5
        # Frame 1 at (32, 32) should no longer be 250! It should be close to 21 (due to video compression/outlier restore)
        f1_out = frames[1]
        assert f1_out[32, 32, 0] < 45  # Glitch suppressed!
