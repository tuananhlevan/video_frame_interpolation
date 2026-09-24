import os
import tempfile
import pytest
from upframe.cli.args import parse_cli_args
from upframe.core.config import PipelineConfig


def test_default_log_file_in_upframe_log():
    inp, out, config = parse_cli_args(["test_videos/test.mp4", "test_videos/test_interpolated.mp4"])
    assert config.log_dir == os.path.join("upframe_log", "test_rife")
    assert config.log_file == os.path.join("upframe_log", "test_rife", "test_rife.log")
    assert "test_videos" not in os.path.dirname(config.log_file)


def test_custom_log_dir():
    inp, out, config = parse_cli_args([
        "test_videos/test.mp4",
        "test_videos/test_interpolated.mp4",
        "--log-dir", "custom_log_dir"
    ])
    assert config.log_dir == os.path.join("custom_log_dir", "test_rife")
    assert config.log_file == os.path.join("custom_log_dir", "test_rife", "test_rife.log")


def test_model_in_log_folder():
    inp, out, config = parse_cli_args([
        "test_videos/short_1.mp4",
        "test_videos/short_1_amt.mp4",
        "--model", "amt"
    ])
    assert config.log_dir == os.path.join("upframe_log", "short_1_amt")
    assert config.log_file == os.path.join("upframe_log", "short_1_amt", "short_1_amt.log")


def test_explicit_log_file():
    inp, out, config = parse_cli_args([
        "test_videos/test.mp4",
        "test_videos/test_interpolated.mp4",
        "--log-file", "specific/path/run.log"
    ])
    assert config.log_file == "specific/path/run.log"


def test_yaml_config_log_dir():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("log_dir: special_log_dir\n")
        tmp_cfg = f.name

    try:
        inp, out, config = parse_cli_args([
            "match.mp4",
            "output.mp4",
            "--config", tmp_cfg
        ])
        assert config.log_dir == os.path.join("special_log_dir", "match_rife")
        assert config.log_file == os.path.join("special_log_dir", "match_rife", "match_rife.log")
    finally:
        if os.path.exists(tmp_cfg):
            os.unlink(tmp_cfg)


def test_bwdif_and_deinterlace_args():
    inp, out, config = parse_cli_args([
        "match.mp4", "output_50p.mp4",
        "--model", "bwdif",
        "--deinterlace", "bwdif"
    ])
    assert config.model == "bwdif"
    assert config.deinterlace == "bwdif"
    assert config.log_dir == os.path.join("upframe_log", "match_bwdif")

    inp2, out2, config2 = parse_cli_args([
        "match.mp4", "output_50p.mp4",
        "--model", "auto"
    ])
    assert config2.model == "auto"
    assert config2.deinterlace == "auto"
