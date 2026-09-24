"""Tests for GMFSS and EMA-VFI backbone adapters."""

import os
import pytest
import torch
import numpy as np

from upframe.models.base import ModelRegistry
import upframe.models.gmfss  # noqa: F401
import upframe.models.ema_vfi  # noqa: F401
import upframe.models.amt  # noqa: F401


@pytest.fixture
def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("fp16", [False, True])
def test_gmfss_adapter(device, fp16):
    """Test GMFSS model loading and inference with FP32 and FP16/AMP."""
    model_cls = ModelRegistry.get("gmfss")
    assert model_cls is not None

    model = model_cls()
    model.load(device=device, fp16=fp16)
    assert model.is_loaded

    h, w = 256, 256
    frame_a = np.zeros((h, w, 3), dtype=np.uint8)
    frame_b = np.full((h, w, 3), 255, dtype=np.uint8)

    interp = model.interpolate(frame_a, frame_b)
    assert interp.shape == (h, w, 3)
    assert interp.dtype == np.uint8

    model.unload()
    assert model._model is None


@pytest.mark.parametrize("fp16", [False, True])
def test_ema_vfi_adapter(device, fp16):
    """Test EMA-VFI model loading and inference with FP32 and FP16/AMP."""
    model_cls = ModelRegistry.get("ema-vfi")
    assert model_cls is not None

    model = model_cls()
    model.load(device=device, fp16=fp16)
    assert model.is_loaded

    h, w = 256, 256
    frame_a = np.zeros((h, w, 3), dtype=np.uint8)
    frame_b = np.full((h, w, 3), 255, dtype=np.uint8)

    interp = model.interpolate(frame_a, frame_b)
    assert interp.shape == (h, w, 3)
    assert interp.dtype == np.uint8

    model.unload()
    assert model._ema_model is None


def test_sequential_gmfss_and_ema_vfi(device):
    """Test loading GMFSS and EMA-VFI sequentially without namespace clashes."""
    gmfss_cls = ModelRegistry.get("gmfss")
    ema_cls = ModelRegistry.get("ema-vfi")

    gmfss = gmfss_cls()
    gmfss.load(device=device, fp16=True)
    assert gmfss.is_loaded

    h, w = 128, 128
    frame_a = np.zeros((h, w, 3), dtype=np.uint8)
    frame_b = np.full((h, w, 3), 128, dtype=np.uint8)
    out_gmfss = gmfss.interpolate(frame_a, frame_b)
    assert out_gmfss.shape == (h, w, 3)
    gmfss.unload()

    ema = ema_cls()
    ema.load(device=device, fp16=True)
    assert ema.is_loaded
    out_ema = ema.interpolate(frame_a, frame_b)
    assert out_ema.shape == (h, w, 3)
    ema.unload()


@pytest.mark.parametrize("fp16", [False, True])
def test_amt_g_adapter(device, fp16):
    """Test AMT-G model loading and inference with FP32 and FP16/AMP."""
    model_cls = ModelRegistry.get("amt-g")
    assert model_cls is not None

    model = model_cls()
    model.load(device=device, fp16=fp16)
    assert model.is_loaded

    h, w = 256, 256
    frame_a = np.zeros((h, w, 3), dtype=np.uint8)
    frame_b = np.full((h, w, 3), 255, dtype=np.uint8)

    interp = model.interpolate(frame_a, frame_b)
    assert interp.shape == (h, w, 3)
    assert interp.dtype == np.uint8

    model.unload()
    assert model.model is None


def test_ema_vfi_tta(device):
    """Test EMA-VFI inference with TTA enabled."""
    model_cls = ModelRegistry.get("ema-vfi")
    assert model_cls is not None

    model = model_cls()
    model.load(device=device, fp16=True, tta=True)
    assert model.is_loaded

    h, w = 128, 128
    frame_a = np.zeros((h, w, 3), dtype=np.uint8)
    frame_b = np.full((h, w, 3), 255, dtype=np.uint8)

    interp = model.interpolate(frame_a, frame_b)
    assert interp.shape == (h, w, 3)
    assert interp.dtype == np.uint8

    model.unload()


@pytest.mark.parametrize("mname", ["interpany-vgg", "interpany-pro"])
def test_interpany_adapter(device, mname):
    """Test InterpAny model loading and inference."""
    import upframe.models.interpany  # noqa: F401

    model_cls = ModelRegistry.get(mname)
    assert model_cls is not None

    model = model_cls()
    model.load(device=device, fp16=True)
    assert model.is_loaded

    h, w = 128, 128
    frame_a = np.zeros((h, w, 3), dtype=np.uint8)
    frame_b = np.full((h, w, 3), 255, dtype=np.uint8)

    interp = model.interpolate(frame_a, frame_b)
    assert interp.shape == (h, w, 3)
    assert interp.dtype == np.uint8

    model.unload()
    assert model.model is None


@pytest.mark.parametrize("mname", ["bwdif", "deinterlace"])
def test_bwdif_adapter(device, mname):
    """Test BWDIF adapter loading and interpolation."""
    import upframe.models.bwdif  # noqa: F401

    model_cls = ModelRegistry.get(mname)
    assert model_cls is not None

    model = model_cls()
    model.load(device=device)
    assert model.is_loaded

    h, w = 64, 64
    frame_a = np.zeros((h, w, 3), dtype=np.uint8)
    frame_b = np.full((h, w, 3), 200, dtype=np.uint8)

    interp_np = model.interpolate(frame_a, frame_b)
    assert interp_np.shape == (h, w, 3)
    assert interp_np.dtype == np.uint8
    assert np.allclose(interp_np, 100)

    tensor_a = torch.zeros((1, 3, h, w), dtype=torch.float32)
    tensor_b = torch.ones((1, 3, h, w), dtype=torch.float32)
    interp_torch = model.interpolate(tensor_a, tensor_b)
    assert interp_torch.shape == (1, 3, h, w)
    assert torch.allclose(interp_torch, torch.tensor(0.5))

    model.unload()
    assert not model.is_loaded


