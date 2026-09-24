"""Unit tests for centralized weights discovery, validation, and auto-downloading."""

import os
import tempfile
import pytest
from upframe.models.weights import (
    CACHE_DIR,
    MODEL_WEIGHT_CANDIDATES,
    is_valid_candidate,
    resolve_checkpoint,
)


def test_candidates_defined():
    """Verify key VFI models have candidates configured."""
    expected_models = ["rife", "amt", "amt-s", "amt-l", "amt-g", "ifrnet", "gmfss", "ema-vfi", "interpany"]
    for model in expected_models:
        assert model in MODEL_WEIGHT_CANDIDATES
        assert len(MODEL_WEIGHT_CANDIDATES[model]) > 0


def test_is_valid_candidate():
    """Verify validation logic for files and directory-based backbones."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Non-existent file
        assert not is_valid_candidate("rife", os.path.join(tmpdir, "missing.pth"))

        # 2. Empty file
        empty_file = os.path.join(tmpdir, "empty.pth")
        with open(empty_file, "wb") as f:
            pass
        assert not is_valid_candidate("rife", empty_file)

        # 3. Valid non-empty file
        valid_file = os.path.join(tmpdir, "valid.pth")
        with open(valid_file, "wb") as f:
            f.write(b"checkpoint_data")
        assert is_valid_candidate("rife", valid_file)

        # 4. GMFSS directory validation (requires 4 pkl files)
        gmfss_dir = os.path.join(tmpdir, "gmfss")
        os.makedirs(gmfss_dir, exist_ok=True)
        assert not is_valid_candidate("gmfss", gmfss_dir)

        for pkl in ["flownet.pkl", "metric.pkl", "feat.pkl"]:
            with open(os.path.join(gmfss_dir, pkl), "wb") as f:
                f.write(b"data")
        assert not is_valid_candidate("gmfss", gmfss_dir)

        # Add the 4th required file
        with open(os.path.join(gmfss_dir, "fusionnet.pkl"), "wb") as f:
            f.write(b"data")
        assert is_valid_candidate("gmfss", gmfss_dir)


def test_resolve_checkpoint_local():
    """Verify local models resolve to absolute paths."""
    rife_path = resolve_checkpoint("rife")
    if rife_path:
        assert os.path.isabs(rife_path)
        assert os.path.exists(rife_path)

    gmfss_path = resolve_checkpoint("gmfss")
    if gmfss_path:
        assert os.path.isabs(gmfss_path)
        assert os.path.exists(gmfss_path)


def test_resolve_checkpoint_explicit_override():
    """Verify explicit path overrides candidate search."""
    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
        tmp.write(b"custom_weights")
        tmp.flush()
        resolved = resolve_checkpoint("rife", explicit_path=tmp.name)
        assert resolved == os.path.abspath(tmp.name)


def test_resolve_nonexistent_no_download():
    """Verify nonexistent model returns None when auto_download is disabled."""
    resolved = resolve_checkpoint("non_existent_model_xyz", auto_download=False)
    assert resolved is None
