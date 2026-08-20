# networks/model_factory.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple, Callable

import torch
import torch.nn as nn

from networks.unet_flexible import UNetFlexible, UNetConfig
from networks.unet_original import UNet as UNetOriginal
from networks.transunet_wrapper import build_transunet  # 你已有的话就用


Builder = Callable[[Dict[str, Any]], Tuple[nn.Module, Dict[str, Any]]]


def _build_flex(cfg_dict: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    cfg = UNetConfig(**cfg_dict)
    model = UNetFlexible(cfg)
    meta = {
        "model_name": "flex",
        "cfg": asdict(cfg),
        "tag": cfg_dict.get("tag", "flex"),   # 用于区分实验
    }
    return model, meta


def _build_orig(cfg_dict: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    in_ch = int(cfg_dict.get("in_ch", 4))
    n_classes = int(cfg_dict.get("n_classes", 4))
    bilinear = bool(cfg_dict.get("bilinear", True))
    model = UNetOriginal(n_channels=in_ch, n_classes=n_classes, bilinear=bilinear)
    meta = {
        "model_name": "orig",
        "cfg": {"in_ch": in_ch, "n_classes": n_classes, "bilinear": bilinear},
        "tag": cfg_dict.get("tag", "orig"),
    }
    return model, meta


def _build_transunet(cfg_dict: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    model, meta = build_transunet(cfg_dict)
    # 保证 meta 有 tag
    meta.setdefault("tag", cfg_dict.get("tag", "transunet"))
    return model, meta


# ===== Registry: 每个实验一个 key =====
REGISTRY: Dict[str, Builder] = {
    # 基础模型
    "orig": _build_orig,
    "flex": _build_flex,
    "transunet": _build_transunet,

    # 下面是“实验别名”(推荐你这么做)
    # 例子：深度/归一化/残差的一套固定配置，直接命令行 --model flex_gn_d5_res
    "flex_gn_d5_res": lambda _: _build_flex({
        "in_ch": 4, "n_classes": 4,
        "base": 32, "depth": 5, "norm": "gn",
        "bilinear": True, "dropout": 0.0, "residual": True,
        "tag": "flex_gn_d5_res",
    }),

    "flex_gn_d4": lambda _: _build_flex({
        "in_ch": 4, "n_classes": 4,
        "base": 32, "depth": 4, "norm": "gn",
        "bilinear": True, "dropout": 0.0, "residual": False,
        "tag": "flex_gn_d4",
    }),

    # 你以后每加一个新方法，就加一行 entry，不影响旧实验
}


def build_model(model_name: str, cfg_dict: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    model_name: registry key
    cfg_dict: for default builders (flex/transunet/orig) used directly
             for preset experiment keys, cfg_dict can be empty {}
    """
    key = model_name.lower()
    if key not in REGISTRY:
        raise ValueError(f"Unknown model_name={model_name}. Available: {sorted(REGISTRY.keys())}")
    return REGISTRY[key](cfg_dict)
