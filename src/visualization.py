# src/viz_ckpt.py
import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from unet_v3 import BrainSegmentationDataset, CropOptimizer
from networks.model_factory import build_model


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        out[k[len("module."):]] = v if k.startswith("module.") else v
        if not k.startswith("module."):
            out[k] = v
    # 上面写法会重复加一次非 module 的 key，修一下
    clean = {}
    for k, v in out.items():
        clean[k] = v
    return clean


def load_ckpt_any(path: str, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    # 2.4+ 有 weights_only，可以减少 pickle 风险/警告；不行就 fallback
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        meta = {
            "model_name": ckpt.get("model_name", ""),
            "cfg": ckpt.get("cfg", {}),
            "crop_coords": ckpt.get("crop_coords", None),
            "epoch": ckpt.get("epoch", None),
            "run_id": ckpt.get("run_id", None),
        }
        state = strip_module_prefix(ckpt["state_dict"])
        return meta, state

    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return {}, strip_module_prefix(ckpt)

    raise RuntimeError(f"Unrecognized checkpoint format: {path}")


@torch.inference_mode()
def visualize_predictions_png(
    model: torch.nn.Module,
    dataset: BrainSegmentationDataset,
    device: torch.device,
    out_dir: str,
    num_samples: int,
    seed: int,
    amp: bool,
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.eval()

    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False).tolist()

    cmap = ListedColormap(["black", "red", "green", "blue"])  # 0/1/2/3

    for k, idx in enumerate(idxs):
        x, y = dataset[idx]  # x:[4,H,W], y:[H,W]
        x_b = x.unsqueeze(0).to(device, non_blocking=True)
        y_np = y.numpy()

        with torch.autocast("cuda", enabled=amp and (device.type == "cuda")):
            logits = model(x_b)          # [1,C,H,W]
            pred = logits.argmax(dim=1)[0].detach().cpu().numpy()

        flair = x[0].numpy()

        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(2, 4, 1); ax.imshow(x[0], cmap="gray"); ax.set_title("flair"); ax.axis("off")
        ax = fig.add_subplot(2, 4, 2); ax.imshow(x[1], cmap="gray"); ax.set_title("t1"); ax.axis("off")
        ax = fig.add_subplot(2, 4, 3); ax.imshow(x[2], cmap="gray"); ax.set_title("t1ce"); ax.axis("off")
        ax = fig.add_subplot(2, 4, 4); ax.imshow(x[3], cmap="gray"); ax.set_title("t2"); ax.axis("off")

        ax = fig.add_subplot(2, 4, 5); ax.imshow(y_np, cmap=cmap, vmin=0, vmax=3); ax.set_title("GT mask"); ax.axis("off")
        ax = fig.add_subplot(2, 4, 6); ax.imshow(pred, cmap=cmap, vmin=0, vmax=3); ax.set_title("Pred mask"); ax.axis("off")

        ax = fig.add_subplot(2, 4, 7)
        ax.imshow(flair, cmap="gray")
        ax.imshow(y_np, cmap=cmap, vmin=0, vmax=3, alpha=0.35)
        ax.set_title("Overlay GT on flair")
        ax.axis("off")

        ax = fig.add_subplot(2, 4, 8)
        ax.imshow(flair, cmap="gray")
        ax.imshow(pred, cmap=cmap, vmin=0, vmax=3, alpha=0.35)
        ax.set_title("Overlay Pred on flair")
        ax.axis("off")

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"sample_{k:03d}_idx{idx}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print(f"[viz] saved {len(idxs)} pngs to: {os.path.abspath(out_dir)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./viz_out")
    ap.add_argument("--num_samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no_img_proc", action="store_true")
    ap.add_argument("--crop_from_csv", type=str, default="")
    ap.add_argument("--force_out_size", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[viz] device={device}")

    meta, state = load_ckpt_any(args.ckpt, device)

    model_name = meta.get("model_name", "") or "flex"
    cfg = meta.get("cfg", {}) or {}
    crop_coords = meta.get("crop_coords", None)

    # out_size: 优先强制；否则用 ckpt cfg 里的 img_size（TransUNet 必须一致）
    out_size = None
    if args.force_out_size > 0:
        out_size = int(args.force_out_size)
    elif isinstance(cfg, dict) and "img_size" in cfg:
        out_size = int(cfg["img_size"])

    # crop: ckpt 没存就从 csv 算
    if crop_coords is None and args.crop_from_csv.strip():
        crop_coords = CropOptimizer(args.crop_from_csv).find_optimal_crop()
        print(f"[viz] crop_from_csv => {crop_coords}")

    print(f"[viz] model_name={model_name}")
    print(f"[viz] cfg={cfg}")
    print(f"[viz] out_size={out_size}, crop_coords={crop_coords}")

    model, _ = build_model(model_name, cfg)
    model.load_state_dict(state, strict=False)
    model = model.to(device)

    ds = BrainSegmentationDataset(
        csv_path=args.csv,
        crop_coords=crop_coords,
        use_img_proc=(not args.no_img_proc),
        out_size=out_size,
        cache_volumes=False,
    )

    visualize_predictions_png(
        model=model,
        dataset=ds,
        device=device,
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        amp=bool(args.amp),
    )


if __name__ == "__main__":
    main()
