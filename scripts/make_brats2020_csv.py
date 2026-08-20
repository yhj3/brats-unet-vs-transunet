# make_brats2020_csv.py
from pathlib import Path
import os
import pandas as pd

ROOT = Path("/home/yihangj3/brain-mri-segmentation/data/brats/training/MICCAI_BraTS2020_TrainingData")
OUT  = Path("/home/yihangj3/brain-mri-segmentation/data/training_detailed_summary_2020.csv")

# 同时支持 .nii.gz 和 .nii
modalities = [
    ("flair", ["_flair.nii.gz", "_flair.nii"]),
    ("t1",    ["_t1.nii.gz",    "_t1.nii"]),
    ("t1ce",  ["_t1ce.nii.gz",  "_t1ce.nii"]),
    ("t2",    ["_t2.nii.gz",    "_t2.nii"]),
    ("seg",   ["_seg.nii.gz",   "_seg.nii"]),
]

def find_modality_file(subject_dir: Path, sid: str, suffixes: list[str]) -> Path | None:
    """
    优先找 {sid}{suffix} 精确文件名；找不到再用 glob 兜底匹配。
    """
    # 1) exact match
    for suf in suffixes:
        f = subject_dir / f"{sid}{suf}"
        if f.exists():
            return f

    # 2) fallback glob
    for suf in suffixes:
        matches = sorted(subject_dir.glob(f"*{suf}"))
        if matches:
            return matches[0]

    return None


def main():
    print("ROOT:", ROOT)
    if not ROOT.exists():
        raise FileNotFoundError(f"ROOT does not exist: {ROOT}")

    subjects = sorted([p for p in ROOT.glob("BraTS20_Training_*") if p.is_dir()])
    print("Found subject dirs:", len(subjects))

    rows = []
    missing_log = []   # 收集缺失情况，最后汇总打印

    for sd in subjects:
        sid = sd.name
        missing_this_subject = []

        for scan, suffixes in modalities:
            f = find_modality_file(sd, sid, suffixes)
            if f is None:
                missing_this_subject.append(scan)
                missing_log.append((sid, scan))
                continue
            rows.append({"Subject ID": sid, "Scan Type": scan, "File Path": str(f)})

        # 如果你希望强制每个 subject 必须 5 个都齐全，否则直接丢弃 subject：
        # （可选：更严格，更适合训练）
        # if missing_this_subject:
        #     # remove any rows already appended for this subject
        #     rows = [r for r in rows if r["Subject ID"] != sid]

    df = pd.DataFrame(rows, columns=["Subject ID", "Scan Type", "File Path"])

    # --- 防呆：不要写空文件 ---
    if df.shape[0] == 0:
        # 不写 OUT，避免留下空文件迷惑你
        print("\nERROR: No rows collected. Check ROOT path and filename suffixes.\n")
        print("Example subject dirs under ROOT (first 5):", subjects[:5])
        raise RuntimeError("No rows collected; aborting without writing CSV.")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    # 原子写入：先写临时文件，再 replace，避免写到一半被中断变空文件
    tmp = OUT.with_suffix(OUT.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, OUT)

    # --- 写完立即验证 ---
    print("Wrote:", OUT)
    print("File size bytes:", OUT.stat().st_size)
    print("Rows:", len(df), "Subjects:", df["Subject ID"].nunique())
    print("Scan Type counts:\n", df["Scan Type"].value_counts())

    if missing_log:
        print("\nMissing summary (showing up to 20):")
        for sid, scan in missing_log[:20]:
            print("  MISSING:", sid, scan)
        print("Total missing entries:", len(missing_log))
    else:
        print("\nAll modalities found for all subjects (based on your matching rules).")


if __name__ == "__main__":
    main()
