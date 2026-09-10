"""RIFE (Real-Time Intermediate Flow Estimation) model adapter for production upframing."""

import logging
from typing import Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from upframe.models.base import BaseVFIModel, ModelRegistry
from upframe.models.weights import resolve_checkpoint
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


class IFBlock(nn.Module):
    """Multi-scale optical flow and fusion estimation block."""

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


class IFNet(nn.Module):
    """Complete multi-scale IFNet architecture for RIFE v4+."""

    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7, c=192)
        self.block1 = IFBlock(18, c=128)
        self.block2 = IFBlock(18, c=96)
        self.block3 = IFBlock(18, c=64)

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


@ModelRegistry.register("rife")
class RIFEModel(BaseVFIModel):
    """RIFE / Practical-RIFE production adapter."""

    def __init__(self) -> None:
        super().__init__(name="rife")
        self.net: Optional[nn.Module] = None
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

        self.net = IFNet().to(self.device)

        resolved_path = resolve_checkpoint("rife", checkpoint_path)
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
                self.net.load_state_dict(clean_state, strict=False)
                logger.info("Successfully loaded RIFE weights.")
            except Exception as e:
                logger.warning(f"Failed to load weights from {resolved_path}: {e}. Using initialized weights.")
        else:
            logger.warning("No RIFE checkpoint specified or found. Using initialized network.")

        self.net.eval()
        if fp16 and self.device.type == "cuda":
            self.net.half()
            self.half_precision = True

        self.is_loaded = True

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
        ta, _ = pad_to_multiple(ta, multiple=32)
        tb, _ = pad_to_multiple(tb, multiple=32)

        with torch.no_grad():
            x = torch.cat([ta, tb], dim=1)
            pred = self.net(x, timestep=0.5)

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
        batch_a, _ = pad_to_multiple(batch_a, multiple=32)
        batch_b, _ = pad_to_multiple(batch_b, multiple=32)

        with torch.no_grad():
            x = torch.cat([batch_a, batch_b], dim=1)
            pred = self.net(x, timestep=0.5)

        return unpad(pred, orig_h, orig_w)
