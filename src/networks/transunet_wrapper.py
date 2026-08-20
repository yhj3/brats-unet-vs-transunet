# src/networks/transunet_wrapper.py
import copy
import torch
import torch.nn as nn
import os
import numpy as np
from networks.transunet.vit_seg_modeling import VisionTransformer, CONFIGS

class TransUNet(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, img_size: int, vit_name: str, pretrained_path: str, n_skip: int):
        super().__init__()


        self.n_classes = int(n_classes)      # ✅ 让 eval.py 的 get_n_classes() 识别到
        self.num_classes = self.n_classes    # ✅ 保险起见也加上
        config = copy.deepcopy(CONFIGS[vit_name])

        # --- important: make grid consistent with img_size for hybrid (R50-*) ---
        if config.patches.get("grid", None) is not None:
            g = int(img_size) // 16
            if g <= 0:
                raise ValueError(f"img_size={img_size} too small for hybrid encoder (needs >=16).")
            config.patches.grid = (g, g)

        # override runtime cfg
        config.n_classes = int(n_classes)
        if hasattr(config, "n_skip"):
            config.n_skip = int(n_skip)

        if pretrained_path:
            config.pretrained_path = pretrained_path

        # --- for your 4-channel MRI: project to 3 channels to reuse ImageNet-pretrained backbone ---
        self.in_proj = nn.Identity()
        if in_ch != 3:
            self.in_proj = nn.Conv2d(in_ch, 3, kernel_size=1, bias=False)

        self.backbone = VisionTransformer(config, img_size=img_size, num_classes=config.n_classes, zero_head=True, vis=False)

        ckpt = getattr(config, "pretrained_path", "")
        if ckpt and os.path.isfile(ckpt):
            w = np.load(ckpt)
            self.backbone.load_from(w)
            print(f"[TransUNet] Loaded pretrained weights from: {ckpt}")
        else:
            print(f"[TransUNet] No pretrained weights found at: {ckpt} (training from scratch)")

    def forward(self, x):
        x = self.in_proj(x)
        return self.backbone(x)


def build_transunet(cfg: dict):
    in_ch = int(cfg.get("in_ch", 4))
    n_classes = int(cfg.get("n_classes", 4))
    img_size = int(cfg.get("img_size", 224))
    vit_name = str(cfg.get("vit_name", "R50-ViT-B_16"))
    pretrained_path = str(cfg.get("pretrained_path", ""))
    n_skip = int(cfg.get("n_skip", 3))

    model = TransUNet(in_ch=in_ch, n_classes=n_classes, img_size=img_size,
                      vit_name=vit_name, pretrained_path=pretrained_path, n_skip=n_skip)

    meta = {
        "model_name": "transunet",
        "cfg": {
            "in_ch": in_ch, "n_classes": n_classes, "img_size": img_size,
            "vit_name": vit_name, "pretrained_path": pretrained_path, "n_skip": n_skip,
        }
    }
    return model, meta
