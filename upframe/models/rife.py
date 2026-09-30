"""RIFE (Real-Time Intermediate Flow Estimation) model adapter for production upframing."""

import logging
import os
import sys
from typing import Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from upframe.models.base import BaseVFIModel, ModelRegistry
from upframe.models.weights import PROJECT_ROOT, resolve_checkpoint
from upframe.utils.tensor import (
    frame_to_tensor,
    tensor_to_frame,
    pad_to_multiple,
    unpad,
    warp
)

logger = logging.getLogger(__name__)


def conv(in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def conv_leaky(in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.2, True)
    )


class Head(nn.Module):
    """Context encoder head for RIFE HDv2/HDv3 architectures."""

    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, feat: bool = False):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3


class ResConv(nn.Module):
    """Residual convolution unit for RIFE HDv2/HDv3."""

    def __init__(self, c: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock_HD(nn.Module):
    """Multi-scale optical flow and fusion block for RIFE HDv2/HDv3."""

    def __init__(self, in_planes: int, c: int = 64):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv_leaky(in_planes, c // 2, 3, 2, 1),
            conv_leaky(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1),
            nn.PixelShuffle(2)
        )

    def forward(self, x: torch.Tensor, flow: Optional[torch.Tensor] = None, scale: float = 1.0):
        if scale != 1.0:
            x = nn.functional.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = nn.functional.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) * (1.0 / scale)
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        if scale != 1.0:
            tmp = nn.functional.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow, mask, feat


class IFNet_HD(nn.Module):
    """Complete multi-scale IFNet architecture for RIFE HDv2/HDv3 checkpoints.
    
    Supports both 4-block (Daydream/Scope v4.25 mirror) and 5-block (upstream Practical-RIFE) configurations.
    """

    def __init__(self, block_channels: tuple[int, ...] = (192, 128, 64, 32)):
        super().__init__()
        self.blocks = nn.ModuleList()
        # block 0 takes 15 channels (img0: 3, img1: 3, f0: 4, f1: 4, timestep: 1)
        self.blocks.append(IFBlock_HD(15, c=block_channels[0]))
        # subsequent blocks take 28 channels (warped0: 3, warped1: 3, wf0: 4, wf1: 4, timestep: 1, mask: 1, feat: 12)
        for c in block_channels[1:]:
            self.blocks.append(IFBlock_HD(28, c=c))
        # Expose individual block attributes block0, block1, etc. for state_dict compatibility
        for i, b in enumerate(self.blocks):
            setattr(self, f"block{i}", b)
        self.encode = Head()

    def forward(
        self,
        x: torch.Tensor,
        timestep: Union[float, torch.Tensor] = 0.5,
        scale_list: Optional[list[float]] = None
    ) -> torch.Tensor:
        num_blocks = len(self.blocks)
        if scale_list is None:
            scale_list = [float(2 ** (num_blocks - 1 - i)) for i in range(num_blocks)]

        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]
        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])

        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        flow: Optional[torch.Tensor] = None
        mask: Optional[torch.Tensor] = None
        feat: Optional[torch.Tensor] = None
        warped_img0 = img0
        warped_img1 = img1

        for i, block in enumerate(self.blocks):
            s = scale_list[i]
            if flow is None:
                flow, mask, feat = block(torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1), None, scale=s)
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, m0, feat = block(torch.cat((warped_img0[:, :3], warped_img1[:, :3], wf0, wf1, timestep, mask, feat), 1), flow, scale=s)
                mask = m0
                flow = flow + fd
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])

        mask = torch.sigmoid(mask)
        merged = warped_img0 * mask + warped_img1 * (1.0 - mask)
        return torch.clamp(merged, 0.0, 1.0)

    def inference(self, img0: torch.Tensor, img1: torch.Tensor, timestep: float = 0.5, scale: float = 1.0) -> torch.Tensor:
        imgs = torch.cat((img0, img1), 1)
        num_blocks = len(self.blocks)
        scale_list = [(2.0 ** (num_blocks - 1 - i)) / scale for i in range(num_blocks)]
        return self(imgs, timestep=timestep, scale_list=scale_list)


class IFBlock_v4(nn.Module):
    """Multi-scale optical flow and fusion estimation block for RIFE v4."""

    def __init__(self, in_planes: int, c: int = 64):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x: torch.Tensor, flow: Optional[torch.Tensor], scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        if scale != 1:
            x = nn.functional.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = nn.functional.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) * (1.0 / scale)
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat) + feat
        tmp = self.lastconv(feat)
        tmp = nn.functional.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        flow_delta = tmp[:, :4] * scale * 2
        mask_delta = tmp[:, 4:5]
        return flow_delta, mask_delta


class IFNet_v4(nn.Module):
    """Complete multi-scale IFNet architecture for RIFE v4+."""

    def __init__(self):
        super().__init__()
        self.block0 = IFBlock_v4(7, c=192)
        self.block1 = IFBlock_v4(18, c=128)
        self.block2 = IFBlock_v4(18, c=96)
        self.block3 = IFBlock_v4(18, c=64)

    def forward(self, x: torch.Tensor, timestep: float = 0.5, scale_list: tuple = (8, 4, 2, 1)) -> torch.Tensor:
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        timestep = (x[:, :1].clone() * 0 + 1) * timestep

        flow: Optional[torch.Tensor] = None
        mask: Optional[torch.Tensor] = None
        blocks = [self.block0, self.block1, self.block2, self.block3]

        for i in range(4):
            if flow is None:
                flow, mask = blocks[i](torch.cat((img0, img1, timestep), 1), None, scale=scale_list[i])
            else:
                f0 = warp(img0, flow[:, :2])
                f1 = warp(img1, flow[:, 2:4])
                f_res, m_res = blocks[i](
                    torch.cat((img0, img1, timestep, mask, f0, f1), 1),
                    flow,
                    scale=scale_list[i]
                )
                flow = flow + f_res
                mask = mask + m_res

        mask = torch.sigmoid(mask)
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        merged = warped_img0 * mask + warped_img1 * (1 - mask)
        return torch.clamp(merged, 0.0, 1.0)

    def inference(self, img0: torch.Tensor, img1: torch.Tensor, timestep: float = 0.5, scale: float = 1.0) -> torch.Tensor:
        x = torch.cat([img0, img1], dim=1)
        scale_list = (8 / scale, 4 / scale, 2 / scale, 1 / scale)
        return self(x, timestep=timestep, scale_list=scale_list)


# Backward compatibility aliases
IFBlock = IFBlock_v4
IFNet = IFNet_v4


@ModelRegistry.register("rife")
class RIFEModel(BaseVFIModel):
    """RIFE / Practical-RIFE production adapter with auto-detection for HDv3 and v4."""

    def __init__(self) -> None:
        super().__init__(name="rife")
        self.net: Optional[nn.Module] = None
        self.rife_model: Optional[Any] = None
        self.half_precision = False

    def load(
        self,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        fp16: bool = False,
        **kwargs: Any
    ) -> None:
        """Loads RIFE weights onto the target device."""
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(f"CUDA requested ('{device}') but not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        resolved_path = resolve_checkpoint("rife", checkpoint_path)
        loaded = False

        if resolved_path:
            logger.info(f"Loading RIFE weights from: {resolved_path}")
            try:
                try:
                    ckpt = torch.load(resolved_path, map_location=self.device, weights_only=False)
                except TypeError:
                    ckpt = torch.load(resolved_path, map_location=self.device)

                state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                clean_state = {}
                for k, v in state_dict.items():
                    name = k.replace("module.", "")
                    if name.startswith("flownet."):
                        name = name[len("flownet."):]
                    clean_state[name] = v

                # Auto-detect model architecture from checkpoint layer dimensions:
                w0 = clean_state.get("block0.conv0.0.0.weight")
                if w0 is not None and w0.shape[1] == 15:
                    # RIFE HDv2/HDv3 architecture (15 input channels on block0)
                    if "block4.conv0.0.0.weight" in clean_state or (
                        "block2.conv0.0.0.weight" in clean_state and clean_state["block2.conv0.0.0.weight"].shape[0] == 48
                    ):
                        logger.info("Detected RIFE HDv3 (5 flow blocks) architecture.")
                        self.net = IFNet_HD(block_channels=(192, 128, 96, 64, 32)).to(self.device)
                    else:
                        logger.info("Detected RIFE HDv3 (4 flow blocks) architecture.")
                        self.net = IFNet_HD(block_channels=(192, 128, 64, 32)).to(self.device)
                else:
                    # RIFE v4 architecture (7 input channels on block0)
                    logger.info("Detected RIFE v4 architecture.")
                    self.net = IFNet_v4().to(self.device)

                self.net.load_state_dict(clean_state, strict=False)
                logger.info(f"Successfully loaded RIFE weights ({self.net.__class__.__name__}).")
                loaded = True
            except Exception as e:
                logger.warning(f"Failed to load weights from {resolved_path}: {e}. Using initialized weights.")

        if not loaded:
            self.net = IFNet_HD().to(self.device)
            logger.warning("No RIFE checkpoint loaded. Using initialized network.")

        if self.net is not None:
            self.net.eval()
            if fp16 and self.device.type == "cuda":
                self.net.half()
                self.half_precision = True

        self.is_loaded = True

    def unload(self) -> None:
        self.net = None
        self.rife_model = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def interpolate(
        self,
        frame_a: Union[torch.Tensor, np.ndarray],
        frame_b: Union[torch.Tensor, np.ndarray],
        **kwargs: Any
    ) -> Union[torch.Tensor, np.ndarray]:
        if not self.is_loaded or self.net is None:
            raise RuntimeError("RIFE model is not loaded. Call model.load() first.")

        is_numpy = isinstance(frame_a, np.ndarray)
        if is_numpy:
            ta = frame_to_tensor(frame_a, self.device, half=self.half_precision)
            tb = frame_to_tensor(frame_b, self.device, half=self.half_precision)
        else:
            ta = frame_a.to(self.device)
            tb = frame_b.to(self.device)
            if self.half_precision:
                ta = ta.half()
                tb = tb.half()

        orig_h, orig_w = ta.shape[-2:]
        pad_mult = 64 if isinstance(self.net, IFNet_HD) else 32
        ta, _ = pad_to_multiple(ta, multiple=pad_mult)
        tb, _ = pad_to_multiple(tb, multiple=pad_mult)

        with torch.no_grad():
            scale = kwargs.get("scale", 1.0)
            timestep = kwargs.get("timestep", 0.5)
            if hasattr(self.net, "inference"):
                pred = self.net.inference(ta, tb, timestep=timestep, scale=scale)
            elif self.rife_model is not None:
                pred = self.rife_model.inference(ta, tb, timestep=timestep, scale=scale)
            else:
                x = torch.cat([ta, tb], dim=1)
                pred = self.net(x, timestep=timestep)

        pred = unpad(pred, orig_h, orig_w)

        if is_numpy:
            return tensor_to_frame(pred)
        return pred

    def interpolate_batch(
        self,
        batch_a: torch.Tensor,
        batch_b: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        if not self.is_loaded or self.net is None:
            raise RuntimeError("RIFE model is not loaded.")

        batch_a = batch_a.to(self.device)
        batch_b = batch_b.to(self.device)
        if self.half_precision:
            batch_a = batch_a.half()
            batch_b = batch_b.half()

        orig_h, orig_w = batch_a.shape[-2:]
        pad_mult = 64 if isinstance(self.net, IFNet_HD) else 32
        batch_a, _ = pad_to_multiple(batch_a, multiple=pad_mult)
        batch_b, _ = pad_to_multiple(batch_b, multiple=pad_mult)

        with torch.no_grad():
            scale = kwargs.get("scale", 1.0)
            timestep = kwargs.get("timestep", 0.5)
            if hasattr(self.net, "inference"):
                pred = self.net.inference(batch_a, batch_b, timestep=timestep, scale=scale)
            elif self.rife_model is not None:
                pred = self.rife_model.inference(batch_a, batch_b, timestep=timestep, scale=scale)
            else:
                x = torch.cat([batch_a, batch_b], dim=1)
                pred = self.net(x, timestep=timestep)

        return unpad(pred, orig_h, orig_w)
