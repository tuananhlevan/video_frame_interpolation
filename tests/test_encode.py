from upframe.pipeline.encode import VideoEncoder


def test_video_encoder_nvenc_fallback():
    # When use_nvenc is True but FFmpeg lacks h264_nvenc, it should cleanly fall back to False (libx264)
    encoder = VideoEncoder(
        output_filepath="test_output.mp4",
        width=1920,
        height=1080,
        fps=50.0,
        use_nvenc=True
    )
    assert encoder.use_nvenc is False
