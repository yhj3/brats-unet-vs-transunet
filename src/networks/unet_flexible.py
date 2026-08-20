# networks/unet_flexible.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


NormType = Literal["bn", "gn", "in", "ln", "none"]


def make_norm(norm: NormType, c: int, gn_groups: int = 32) -> nn.Module:
    """
    norm:
      - bn: BatchNorm2d
      - gn: GroupNorm (good for small batch)
      - in: InstanceNorm2d
      - ln: LayerNorm on channels (implemented via GroupNorm(1, C))
      - none: Identity
    """
    if norm == "bn":
        return nn.BatchNorm2d(c)
    if norm == "gn":
        g = min(gn_groups, c)
        # make g divide c if possible
        while g > 1 and (c % g) != 0:
            g -= 1
        return nn.GroupNorm(g, c)
    if norm == "in":
        return nn.InstanceNorm2d(c, affine=True)
    if norm == "ln":
        # LN over channels for 2D feature maps: GN(1, C) is common trick
        return nn.GroupNorm(1, c)
    return nn.Identity()


class ConvBlock(nn.Module):
    """
    Conv -> Norm -> Act  (x2)
    Optional residual connection.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm: NormType = "gn",
        act: Literal["relu", "silu", "gelu"] = "relu",
        dropout: float = 0.0,
        residual: bool = False,
        gn_groups: int = 32,
    ):
        super().__init__()
        self.residual = residual and (in_ch == out_ch)

        if act == "relu":
            act_fn = nn.ReLU(inplace=True)
        elif act == "silu":
            act_fn = nn.SiLU(inplace=True)
        else:
            act_fn = nn.GELU()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = make_norm(norm, out_ch, gn_groups)
        self.act1 = act_fn

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = make_norm(norm, out_ch, gn_groups)
        self.act2 = act_fn

        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.drop(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = self.act2(y)

        if self.residual:
            y = y + x
        return y


class DownBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm: NormType = "gn",
        act: Literal["relu", "silu", "gelu"] = "relu",
        dropout: float = 0.0,
        residual: bool = False,
        gn_groups: int = 32,
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, norm=norm, act=act, dropout=dropout, residual=residual, gn_groups=gn_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        bilinear: bool = True,
        norm: NormType = "gn",
        act: Literal["relu", "silu", "gelu"] = "relu",
        dropout: float = 0.0,
        residual: bool = False,
        gn_groups: int = 32,
    ):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            up_out_ch = in_ch
        else:
            # transposed conv halves channels
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            up_out_ch = in_ch // 2

        self.conv = ConvBlock(
            in_ch=up_out_ch + skip_ch,
            out_ch=out_ch,
            norm=norm,
            act=act,
            dropout=dropout,
            residual=residual,
            gn_groups=gn_groups,
        )

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self._match_size(x, skip)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


@dataclass
class UNetConfig:
    in_ch: int = 4
    n_classes: int = 4
    base: int = 32
    depth: int = 4
    bilinear: bool = True
    norm: NormType = "gn"
    act: Literal["relu", "silu", "gelu"] = "relu"
    dropout: float = 0.0
    residual: bool = False
    gn_groups: int = 32


class UNetFlexible(nn.Module):
    """
    Flexible UNet:
      - depth controls how many down/up levels
      - norm: bn/gn/in/ln/none
      - bilinear or transposed conv
      - residual conv blocks (only when in_ch == out_ch inside block)
    """
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        assert cfg.depth >= 2, "depth too small, try >=2"

        self.cfg = cfg
        self.n_channels = cfg.in_ch
        self.n_classes  = cfg.n_classes
        self.bilinear   = cfg.bilinear

        # encoder
        chs = [cfg.base * (2 ** i) for i in range(cfg.depth)]  # e.g., [32,64,128,256] if depth=4
        self.stem = ConvBlock(cfg.in_ch, chs[0], norm=cfg.norm, act=cfg.act, dropout=cfg.dropout,
                              residual=False, gn_groups=cfg.gn_groups)

        self.downs = nn.ModuleList()
        for i in range(1, cfg.depth):
            self.downs.append(
                DownBlock(
                    in_ch=chs[i - 1],
                    out_ch=chs[i],
                    norm=cfg.norm,
                    act=cfg.act,
                    dropout=cfg.dropout,
                    residual=cfg.residual,
                    gn_groups=cfg.gn_groups,
                )
            )

        # bottleneck
        bott_in = chs[-1]
        bott_out = chs[-1] * 2
        self.bottleneck = ConvBlock(
            bott_in, bott_out,
            norm=cfg.norm, act=cfg.act, dropout=cfg.dropout, residual=False, gn_groups=cfg.gn_groups
        )

        # decoder
        self.ups = nn.ModuleList()
        cur = bott_out
        for i in reversed(range(cfg.depth - 1)):
            skip_ch = chs[i]
            out_ch = chs[i]
            self.ups.append(
                UpBlock(
                    in_ch=cur,
                    skip_ch=skip_ch,
                    out_ch=out_ch,
                    bilinear=cfg.bilinear,
                    norm=cfg.norm,
                    act=cfg.act,
                    dropout=cfg.dropout,
                    residual=cfg.residual,
                    gn_groups=cfg.gn_groups,
                )
            )
            cur = out_ch

        self.head = nn.Conv2d(cfg.base, cfg.n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        x = self.stem(x)

        for down in self.downs:
            skips.append(x)     # ✅ 下采样前存 skip
            x = down(x)

        x = self.bottleneck(x)

        for up in self.ups:
            skip = skips.pop()  # ✅ 对应上一个尺度
            x = up(x, skip)

        return self.head(x)

