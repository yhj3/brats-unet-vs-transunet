# src/eval.py
import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 直接复用 unet_v3 的 Dataset / CropOptimizer
from unet_v3 import BrainSegmentationDataset, CropOptimizer

# 用 factory 重建模型
from networks.model_factory import build_model


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            out[k[len("module.") :]] = v
        else:
            out[k] = v
    return out


def load_ckpt_any(path: str, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """
    Returns: (meta_dict, state_dict)
    meta_dict may be empty for old checkpoints.
    """
    ckpt = torch.load(path, map_location=device)

    # new format: dict with state_dict
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        meta = {
            "model_name": ckpt.get("model_name", ""),
            "cfg": ckpt.get("cfg", {}),
            "crop_coords": ckpt.get("crop_coords", None),
            "epoch": ckpt.get("epoch", None),
            "run_id": ckpt.get("run_id", None),
            "val_fg_dice": ckpt.get("val_fg_dice", None),
        }
        state = ckpt["state_dict"]
        return meta, strip_module_prefix(state)

    # old format: pure state_dict
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return {}, strip_module_prefix(ckpt)

    raise RuntimeError(f"Unrecognized checkpoint format: {path}")


def build_fallback_cfg(args) -> Dict[str, Any]:
    # 用于 old checkpoint（只有 state_dict 没有 cfg）时，必须你手动告诉它模型结构
    cfg = dict(
        in_ch=args.fallback_in_ch,
        n_classes=args.fallback_n_classes,
        base=args.fallback_base,
        depth=args.fallback_depth,
        norm=args.fallback_norm,
        bilinear=args.fallback_bilinear,
        dropout=args.fallback_dropout,
        residual=args.fallback_residual,
    )
    return cfg


def _fast_confusion_matrix(pred: torch.Tensor, true: torch.Tensor, C: int) -> torch.Tensor:
    """
    pred/true: [B,H,W] long on GPU
    returns conf: [C,C] long on GPU, conf[t, p] counts
    """
    pred = pred.reshape(-1).to(torch.int64)
    true = true.reshape(-1).to(torch.int64)

    # safety: clamp labels into range
    true = true.clamp_(0, C - 1)
    pred = pred.clamp_(0, C - 1)

    idx = true * C + pred
    conf = torch.bincount(idx, minlength=C * C).reshape(C, C)
    return conf


def _dice_from_confmat(conf: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    conf: [C,C] (counts), conf[t,p]
    returns dice: [C] float
    """
    conf = conf.to(torch.float32)
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    return dice


def evaluate_one_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = False,
) -> Dict[str, float]:
    model.eval()
    device_type = "cuda" if device.type == "cuda" else "cpu"

    ce_sum = 0.0
    n_samples = 0

    conf_total: Optional[torch.Tensor] = None  # [C,C] on GPU

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.long, non_blocking=True)

            with torch.amp.autocast(device_type=device_type, enabled=amp):
                logits = model(x)  # [B,C,H,W] usually

                # 若尺寸不一致，强行对齐（避免某些模型输出和 label 不一致导致报错）
                if logits.ndim == 4 and logits.shape[-2:] != y.shape[-2:]:
                    logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

                ce = F.cross_entropy(logits, y)

            B = int(x.shape[0])
            ce_sum += float(ce.item()) * B
            n_samples += B

            # 直接用 logits 推 C（不再依赖 get_n_classes(model)，避免你之前 TransUNet 那个报错）
            C = int(logits.shape[1])
            pred = logits.argmax(dim=1)  # [B,H,W]

            conf = _fast_confusion_matrix(pred, y, C)
            if conf_total is None:
                conf_total = conf
            else:
                conf_total += conf

    if n_samples == 0 or conf_total is None:
        return {"mean_ce": np.nan, "fg_dice": np.nan}

    dice = _dice_from_confmat(conf_total)  # [C]
    # foreground mean dice (classes 1..C-1)
    if dice.numel() > 1:
        fg_dice = float(dice[1:].mean().item())
    else:
        fg_dice = float(dice.mean().item())

    out: Dict[str, float] = {
        "mean_ce": ce_sum / max(1, n_samples),
        "fg_dice": fg_dice,
    }

    # per-class dice (exclude background)
    for c in range(1, int(dice.numel())):
        out[f"dice_cls{c}"] = float(dice[c].item())

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="./eval_compare.csv")

    # 你可以直接给两个 ckpt，也可以给目录
    ap.add_argument("--ckpts", type=str, nargs="*", default=[])
    ap.add_argument("--ckpt_dirs", type=str, nargs="*", default=[])

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--no_img_proc", action="store_true")

    # crop：优先用 ckpt 里的 crop_coords；否则用 --crop_from_csv 重新算；否则不 crop
    ap.add_argument("--crop_from_csv", type=str, default="")

    # resize：优先用 ckpt cfg 的 img_size (transunet)；否则用 --force_out_size；否则 None
    ap.add_argument("--force_out_size", type=int, default=0)

    # DataLoader speed knobs
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--cache_volumes", action="store_true")  # 如果 IO 是瓶颈，这个会很快，但吃内存

    # cudnn benchmark（输入 size 固定时一般更快）
    ap.add_argument("--cudnn_benchmark", action="store_true")

    # fallback（给旧 ckpt 用）
    ap.add_argument("--fallback_model", type=str, default="flex")
    ap.add_argument("--fallback_in_ch", type=int, default=4)
    ap.add_argument("--fallback_n_classes", type=int, default=4)
    ap.add_argument("--fallback_base", type=int, default=32)
    ap.add_argument("--fallback_depth", type=int, default=4)
    ap.add_argument("--fallback_norm", type=str, default="gn")
    ap.add_argument("--fallback_bilinear", action="store_true")
    ap.add_argument("--fallback_dropout", type=float, default=0.0)
    ap.add_argument("--fallback_residual", action="store_true")

    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[eval] device = {device}")

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # collect checkpoints
    ckpt_paths: List[str] = []
    ckpt_paths += list(args.ckpts)

    for d in args.ckpt_dirs:
        d = Path(d)
        if not d.exists():
            print(f"[eval] WARN: ckpt_dir not found: {d}")
            continue
        for p in sorted(d.rglob("*.pth")) + sorted(d.rglob("*.pt")):
            ckpt_paths.append(str(p))

    ckpt_paths = [p for p in ckpt_paths if p]
    ckpt_paths = sorted(list(dict.fromkeys(ckpt_paths)))  # unique preserve order-ish

    if len(ckpt_paths) == 0:
        raise RuntimeError("No checkpoints found. Pass --ckpts or --ckpt_dirs correctly.")

    # optional crop coords fallback
    crop_fallback = None
    if args.crop_from_csv.strip():
        crop_fallback = CropOptimizer(args.crop_from_csv).find_optimal_crop()
        print(f"[eval] crop_from_csv => {crop_fallback}")

    rows = []

    # 关键：复用 DataLoader（避免每个 ckpt 都重建 workers）
    loader_cache: Dict[Tuple, DataLoader] = {}

    for ckpt_path in ckpt_paths:
        print(f"\n[eval] loading: {ckpt_path}")
        meta, state = load_ckpt_any(ckpt_path, device)

        model_name = meta.get("model_name", "") or args.fallback_model
        cfg = meta.get("cfg", {}) or {}
        crop_coords = meta.get("crop_coords", None)

        # old ckpt: cfg missing => must use fallback cfg
        if (not cfg) and (model_name.lower() == args.fallback_model.lower()):
            cfg = build_fallback_cfg(args)
            print(f"[eval] old ckpt detected, use fallback cfg: {cfg}")

        # decide out_size
        out_size = None
        if int(args.force_out_size) > 0:
            out_size = int(args.force_out_size)
        elif isinstance(cfg, dict) and "img_size" in cfg:
            try:
                out_size = int(cfg["img_size"])
            except Exception:
                out_size = None

        # decide crop
        if crop_coords is None:
            crop_coords = crop_fallback

        # build model
        model, _meta2 = build_model(model_name, cfg)
        model.load_state_dict(state, strict=False)
        model = model.to(device)

        # dataset+loader (cached)
        ds_key = (
            str(Path(args.test_csv).resolve()),
            str(crop_coords),
            bool(not args.no_img_proc),
            int(out_size) if out_size is not None else None,
            int(args.batch_size),
            int(args.num_workers),
            int(args.prefetch_factor),
            bool(args.persistent_workers),
            bool(args.cache_volumes),
        )

        if ds_key in loader_cache:
            loader = loader_cache[ds_key]
        else:
            ds = BrainSegmentationDataset(
                csv_path=args.test_csv,
                crop_coords=crop_coords,
                use_img_proc=(not args.no_img_proc),
                out_size=out_size,
                cache_volumes=bool(args.cache_volumes),
            )

            loader_args = dict(
                dataset=ds,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(args.num_workers),
                pin_memory=True,
            )
            if int(args.num_workers) > 0:
                loader_args["persistent_workers"] = bool(args.persistent_workers)
                loader_args["prefetch_factor"] = int(args.prefetch_factor)

            loader = DataLoader(**loader_args)
            loader_cache[ds_key] = loader

        metrics = evaluate_one_model(model, loader, device, amp=bool(args.amp))

        row = {
            "ckpt": ckpt_path,
            "model_name": model_name,
            "epoch": meta.get("epoch", None),
            "run_id": meta.get("run_id", None),
            "val_fg_dice_in_ckpt": meta.get("val_fg_dice", None),
            "out_size": out_size,
            "crop_coords": str(crop_coords),
            "cfg": str(cfg),
        }
        row.update(metrics)
        rows.append(row)

        print(f"[eval] fg_dice={row.get('fg_dice'):.4f}  mean_ce={row.get('mean_ce'):.4f}")

        # 释放显存（评很多 ckpt 时更稳）
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\n[eval] Saved CSV => {out_csv.resolve()}")
    print(df[["model_name", "epoch", "fg_dice", "mean_ce", "ckpt"]].to_string(index=False))


if __name__ == "__main__":
    main()
