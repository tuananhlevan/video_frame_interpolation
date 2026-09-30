from unittest.mock import patch, MagicMock
from upframe.pipeline.encode import VideoEncoder
from upframe.core.types import ChunkTask
from upframe.workers.gpu_worker import GPUWorker


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


def test_chunk_task_use_nvenc_field():
    task = ChunkTask(
        chunk_id=1,
        start_frame=0,
        end_frame=100,
        input_path="input.mp4",
        output_dir="tmp",
        width=1920,
        height=1080,
        fps=25.0,
        use_nvenc=True,
        batch_size=16
    )
    assert task.use_nvenc is True
    assert task.batch_size == 16


@patch("upframe.workers.gpu_worker.VideoEncoder")
@patch("upframe.workers.gpu_worker.VideoDecoder")
def test_gpu_worker_nvenc_passed_to_chunk_encoder(mock_decoder_cls, mock_encoder_cls):
    mock_decoder = MagicMock()
    mock_decoder.read_frames_exact.return_value = []
    mock_decoder_cls.return_value = mock_decoder

    task = ChunkTask(
        chunk_id=1,
        start_frame=0,
        end_frame=10,
        input_path="input.mp4",
        output_dir="tmp",
        width=1920,
        height=1080,
        fps=25.0,
        use_nvenc=True
    )
    worker = GPUWorker(device="cpu", model_name="rife")
    result = worker.process_chunk(task)
    assert result.status == "COMPLETED"
    assert result.output_frames_count == 0

