#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def infer_run_name_from_ckpt(ckpt_path: str) -> str:
    """
    Try to infer a run name from checkpoint path when run_id is missing.
    Examples:
      checkpoints/flex/checkpoint_flex_epoch10.pth -> flex
      checkpoints/norm_depth/flex_epoch1.pth -> norm_depth
      checkpoints/runs/jobA/checkpoint_jobA_epoch1.pth -> jobA
      checkpoints/transunet/transunet_xxx/checkpoint_... -> transunet_xxx
      checkpoints/unet_orig_baseline/checkpoint_... -> unet_orig_baseline
    """
    if not isinstance(ckpt_path, str) or ckpt_path.strip() == "":
        return "unknown"

    p = Path(ckpt_path)
    parts = list(p.parts)

    # find ".../checkpoints/..."
    if "checkpoints" in parts:
        i = parts.index("checkpoints")
        after = parts[i + 1:]  # segments after "checkpoints"
        if len(after) == 0:
            return "checkpoints"

        # special: checkpoints/runs/<run_id>/...
        if after[0] == "runs" and len(after) >= 2:
            return after[1]

        # common: checkpoints/<something>/...
        return after[0]

    # fallback: parent folder name
    return p.parent.name if p.parent else "unknown"


def make_group_label(row) -> str:
    model = str(row.get("model_name", "unknown"))
    run_id = row.get("run_id", "")
    run_id = "" if pd.isna(run_id) else str(run_id).strip()

    if run_id == "":
        run_id = infer_run_name_from_ckpt(str(row.get("ckpt", "")))

    return f"{model}:{run_id}"


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path, title: str = ""):
    # Keep only rows with valid epoch & metric
    d = df.copy()
    d = d[pd.notna(d["epoch"])]
    d = d[pd.notna(d[metric])]

    if len(d) == 0:
        print(f"[skip] no data for metric={metric}")
        return

    # sort and group
    d = d.sort_values(["group", "epoch"])

    plt.figure()
    for g, sub in d.groupby("group"):
        sub = sub.sort_values("epoch")
        plt.plot(sub["epoch"].values, sub[metric].values, marker="o", linewidth=1, markersize=3, label=g)

    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(title or metric)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[ok] saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to eval csv")
    ap.add_argument("--out_dir", default="./curve_plots", help="output directory for plots")
    ap.add_argument("--metrics", default="fg_dice,mean_ce,dice_cls1,dice_cls2,dice_cls3,val_fg_dice_in_ckpt",
                    help="comma-separated metrics to plot")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)

    df = pd.read_csv(csv_path)

    # basic cleanup
    if "epoch" not in df.columns:
        raise ValueError("CSV missing required column: epoch")

    # ensure numeric epoch
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")

    # make group label
    df["group"] = df.apply(make_group_label, axis=1)

    df = df[~df["ckpt"].astype(str).str.contains("/checkpoints/norm_depth/", na=False)].copy()

    # convert metric columns to numeric if exist
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip() != ""]
    for m in metrics:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")
        else:
            print(f"[warn] metric column not found in CSV: {m}")

    # Print quick best summary for fg_dice if available
    if "fg_dice" in df.columns:
        tmp = df[pd.notna(df["fg_dice"])].copy()
        if len(tmp) > 0:
            best = tmp.sort_values("fg_dice", ascending=False).groupby("group").head(1)
            best = best.sort_values("fg_dice", ascending=False)
            print("\n=== Best fg_dice per group ===")
            for _, r in best.iterrows():
                print(f"{r['group']:<30} best_fg_dice={r['fg_dice']:.4f} at epoch={int(r['epoch'])}  ckpt={r.get('ckpt','')}")
            print("================================\n")

    # plot
    for m in metrics:
        if m not in df.columns:
            continue
        plot_metric(
            df=df,
            metric=m,
            out_path=out_dir / f"{m}.png",
            title=f"{m} vs epoch (from {csv_path.name})"
        )


if __name__ == "__main__":
    main()
