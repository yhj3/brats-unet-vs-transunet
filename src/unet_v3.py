# src/unet_v3.py
"""
Brain MRI (BraTS-style) segmentation training script.

Key features:
- One CLI entrypoint to run different models (flex / transunet / orig)
- Checkpoints save BOTH weights + cfg (so eval can reconstruct the model)
- Experiments separated automatically via run_id (and you can override with --run_name)
- Crop optimizer that returns None if crop can't be found (so dataset won't crash)
- Optional resize (for fair comparison / TransUNet fixed input size)
- Mixed precision (--amp), weighted CE, foreground Dice loss (with warmup)
- NEW: fg-slice oversampling via WeightedRandomSampler (helps class imbalance a lot)
- NEW: Dice loss = multi-class + only-present-classes (avoids empty-class inflation)
- NEW: Gradient accumulation (--accum_steps) to reduce GPU usage (slower, but less memory/compute peak)
"""

import argparse
import json
import logging
import os
import re
import subprocess
from hashlib import md5
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm

# Your UNet-flexible implementation
from networks.unet_flexible import UNetFlexible, UNetConfig

# Optional: model_factory (preferred)
try:
    from networks.model_factory import build_model as build_model_factory  # (model_name, cfg_dict) -> (model, meta)
except Exception:
    build_model_factory = None

# Optional: transunet wrapper (if you don't use factory)
try:
    from networks.transunet_wrapper import build_transunet
except Exception:
    build_transunet = None


# -------------------------
# utils
# -------------------------
def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def seed_everything(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def get_n_classes(model: nn.Module) -> int:
    m = unwrap_model(model)
    if hasattr(m, "n_classes"):
        return int(getattr(m, "n_classes"))
    if hasattr(m, "num_classes"):
        return int(getattr(m, "num_classes"))
    raise AttributeError("Model does not expose n_classes/num_classes; please add it for dice/onehot.")


def parse_csv_floats(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [float(x) for x in s.split(",")]


def safe_run_name(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)


def short_cfg_hash(cfg: Dict[str, Any]) -> str:
    try:
        blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    except Exception:
        blob = str(cfg).encode("utf-8")
    return md5(blob).hexdigest()[:8]


def get_git_short_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        sha = out.decode("utf-8").strip()
        return sha if sha else None
    except Exception:
        return None


def optimizer_to(optimizer: optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


# -------------------------
# Dice
# -------------------------
def dice_coeff(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    prob/target: [B, C, H, W] float
    returns mean dice over batch & channels
    """
    assert prob.shape == target.shape
    dims = tuple(range(2, prob.ndim))  # sum over H,W
    inter = 2.0 * (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    dice = (inter + eps) / (denom + eps)
    return dice.mean()


def dice_loss(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - dice_coeff(prob, target)


def dice_loss_mc_present(prob_fg: torch.Tensor, true_fg: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    prob_fg/true_fg: [B, K, H, W] , here K=3 for labels 1,2,3
    Only compute dice on classes that appear in target within the batch.
    This prevents empty classes from inflating dice to ~1.
    """
    assert prob_fg.shape == true_fg.shape
    dims = (0, 2, 3)  # sum over B,H,W

    inter = 2.0 * (prob_fg * true_fg).sum(dims)          # [K]
    denom = prob_fg.sum(dims) + true_fg.sum(dims)        # [K]
    dice = (inter + eps) / (denom + eps)                 # [K]

    present = (true_fg.sum(dims) > 0).float()            # [K]
    dice_mean = (dice * present).sum() / present.sum().clamp(min=1.0)
    return 1.0 - dice_mean


# -------------------------
# Original UNet baseline
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetOriginal(nn.Module):
    def __init__(self, n_channels=4, n_classes=4, bilinear=True):
        super().__init__()
        self.n_classes = int(n_classes)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# -------------------------
# Crop optimizer
# -------------------------
class CropOptimizer:
    def __init__(self, csv_path: str, modalities: Optional[List[str]] = None):
        self.data_summary = pd.read_csv(csv_path)
        self.modalities = modalities or ["flair", "t1", "t1ce", "t2"]

    def find_optimal_crop(self) -> Optional[Tuple[int, int, int, int]]:
        crop_coords: List[List[int]] = []
        for sid in self.data_summary["Subject ID"].unique():
            sub = self.data_summary[self.data_summary["Subject ID"] == sid]
            for mod in self.modalities:
                rows = sub[sub["Scan Type"] == mod]
                if len(rows) == 0:
                    continue
                fpath = rows["File Path"].values[0]
                if not isinstance(fpath, str) or (not os.path.exists(fpath)):
                    continue
                try:
                    data = nib.load(fpath).get_fdata()
                except Exception:
                    continue
                nz = np.argwhere(data > 0)
                if nz.size == 0:
                    continue
                crop_coords.append([
                    int(np.min(nz[:, 0])), int(np.max(nz[:, 0])) + 1,  # exclusive
                    int(np.min(nz[:, 1])), int(np.max(nz[:, 1])) + 1,
                ])

        crop_coords = np.array(crop_coords)
        if crop_coords.size == 0:
            logging.warning("[CropOptimizer] No nonzero voxels found for crop. Disable cropping.")
            return None

        min_x = int(np.min(crop_coords[:, 0]))
        max_x = int(np.max(crop_coords[:, 1]))  # already exclusive
        min_y = int(np.min(crop_coords[:, 2]))
        max_y = int(np.max(crop_coords[:, 3]))  # already exclusive

        if max_x <= min_x or max_y <= min_y:
            logging.warning("[CropOptimizer] Degenerate crop found. Disable cropping.")
            return None

        return (min_x, max_x, min_y, max_y)


# -------------------------
# Dataset
# -------------------------
class _VolumeCache:
    def __init__(self, max_items: int = 32):
        self.max_items = int(max_items)
        self._keys: List[str] = []
        self._store: Dict[str, np.ndarray] = {}

    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self._store:
            self._keys.remove(key)
            self._keys.append(key)
            return self._store[key]
        return None

    def put(self, key: str, value: np.ndarray) -> None:
        if self.max_items <= 0:
            return
        if key in self._store:
            self._store[key] = value
            self._keys.remove(key)
            self._keys.append(key)
            return
        if len(self._keys) >= self.max_items:
            old = self._keys.pop(0)
            self._store.pop(old, None)
        self._store[key] = value
        self._keys.append(key)


class BrainSegmentationDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        crop_coords: Optional[Tuple[int, int, int, int]] = None,
        use_img_proc: bool = True,
        out_size: Optional[int] = None,
        cache_volumes: bool = False,
        cache_max_items: int = 32,
    ):
        self.data_summary = pd.read_csv(csv_path)
        self.crop_coords = crop_coords
        self.use_img_proc = use_img_proc
        self.out_size = out_size

        self.modalities = ["flair", "t1", "t1ce", "t2"]

        # filter valid subjects (must have flair/t1/t1ce/t2 + seg)
        need = set(self.modalities + ["seg"])
        self.subjects: List[str] = []
        for sid in self.data_summary["Subject ID"].unique():
            sub = self.data_summary[self.data_summary["Subject ID"] == sid]
            have = set(sub["Scan Type"].tolist())
            if not need.issubset(have):
                logging.warning(f"[Dataset] skip subject {sid}, has={sorted(list(have))}")
                continue
            self.subjects.append(sid)

        self.cache_volumes = bool(cache_volumes)
        self.cache = _VolumeCache(max_items=cache_max_items) if self.cache_volumes else None

        # slice index + fg flags
        self.slice_info = self._create_slice_index()

    @staticmethod
    def Img_proc(image: np.ndarray, _lambda: float = -0.8, eps: float = 1e-6) -> np.ndarray:
        if np.isnan(image).any() or np.isinf(image).any():
            return np.zeros_like(image, dtype=np.float32)
        mn = float(np.min(image))
        mx = float(np.max(image))
        if mx <= mn:
            return np.zeros_like(image, dtype=np.float32)

        x = (image - mn) / (mx - mn + eps)

        m = float(np.max(x))
        img1 = (m / np.log(m + 1 + eps)) * np.log(x + 1 + eps)
        img2 = 1.0 - np.exp(-x)
        img3 = (img1 + img2) / (_lambda + (img1 * img2) + eps)

        try:
            from scipy.special import erf
            img4 = erf(_lambda * np.arctan(np.exp(img3)) - 0.5 * img3)
        except Exception:
            import math
            img4 = np.vectorize(math.erf)(_lambda * np.arctan(np.exp(img3)) - 0.5 * img3)

        mn2, mx2 = float(np.min(img4)), float(np.max(img4))
        if mx2 <= mn2:
            return np.zeros_like(img4, dtype=np.float32)
        return ((img4 - mn2) / (mx2 - mn2 + eps)).astype(np.float32)

    def _load_volume(self, fpath: str) -> np.ndarray:
        if self.cache is not None:
            cached = self.cache.get(fpath)
            if cached is not None:
                return cached
        vol = nib.load(fpath).get_fdata().astype(np.float32)
        if self.cache is not None:
            self.cache.put(fpath, vol)
        return vol

    def _create_slice_index(self) -> List[Tuple[str, int]]:
        """
        Also builds self.has_fg (bool per slice) for sampler usage.
        has_fg=True means the corresponding seg slice has any tumor voxel (>0).
        """
        slice_info: List[Tuple[str, int]] = []
        has_fg: List[bool] = []

        min_x = max_x = min_y = max_y = None
        if self.crop_coords is not None:
            min_x, max_x, min_y, max_y = self.crop_coords

        for sid in self.subjects:
            sub = self.data_summary[self.data_summary["Subject ID"] == sid]
            flair_path = sub[sub["Scan Type"] == "flair"]["File Path"].values[0]
            seg_path = sub[sub["Scan Type"] == "seg"]["File Path"].values[0]

            if (not os.path.exists(flair_path)) or (not os.path.exists(seg_path)):
                continue

            try:
                flair = nib.load(flair_path).get_fdata().astype(np.float32)
                seg = nib.load(seg_path).get_fdata().astype(np.uint8)
            except Exception:
                continue

            seg[seg == 4] = 3

            if self.crop_coords is not None:
                flair = flair[min_x:max_x, min_y:max_y, :]
                seg = seg[min_x:max_x, min_y:max_y, :]

            depth = flair.shape[2]
            for z in range(15, max(15, depth - 12)):
                if not np.any(flair[:, :, z] > 0):
                    continue
                slice_info.append((sid, z))
                has_fg.append(bool(np.any(seg[:, :, z] > 0)))

        self.has_fg = has_fg
        if len(has_fg) > 0:
            pos = sum(has_fg)
            logging.info(f"[Dataset] slices={len(has_fg)}, fg_slices={pos} ({pos/len(has_fg):.3f})")

        return slice_info

    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, idx):
        sid, z = self.slice_info[idx]
        sub = self.data_summary[self.data_summary["Subject ID"] == sid]

        min_x = max_x = min_y = max_y = None
        if self.crop_coords is not None:
            min_x, max_x, min_y, max_y = self.crop_coords

        imgs: List[np.ndarray] = []
        for mod in self.modalities:
            fpath = sub[sub["Scan Type"] == mod]["File Path"].values[0]
            vol = self._load_volume(fpath) if self.cache_volumes else nib.load(fpath).get_fdata().astype(np.float32)
            if self.crop_coords is not None:
                vol = vol[min_x:max_x, min_y:max_y, :]
            sl = vol[:, :, z]

            mu = float(np.mean(sl))
            sd = float(np.std(sl)) + 1e-6
            sl = (sl - mu) / sd

            if self.use_img_proc:
                sl = self.Img_proc(sl)
            imgs.append(sl)

        x = np.stack(imgs, axis=0).astype(np.float32)  # [C,H,W]

        seg_path = sub[sub["Scan Type"] == "seg"]["File Path"].values[0]
        seg = self._load_volume(seg_path) if self.cache_volumes else nib.load(seg_path).get_fdata().astype(np.float32)
        if self.crop_coords is not None:
            seg = seg[min_x:max_x, min_y:max_y, :]
        seg = seg.astype(np.uint8)
        seg[seg == 4] = 3
        y = seg[:, :, z].astype(np.int64)

        x_t = torch.from_numpy(x).float()
        y_t = torch.from_numpy(y).long()

        if self.out_size is not None and int(self.out_size) > 0:
            out = int(self.out_size)
            x_t = F.interpolate(x_t.unsqueeze(0), size=(out, out), mode="bilinear", align_corners=False).squeeze(0)
            y_t = F.interpolate(y_t.unsqueeze(0).unsqueeze(0).float(), size=(out, out), mode="nearest").squeeze(0).squeeze(0).long()

        return x_t, y_t


# -------------------------
# checkpoint helpers
# -------------------------
def find_latest_checkpoint(ckpt_dir: str, run_id: str) -> Optional[Path]:
    p = Path(ckpt_dir)
    if not p.exists():
        return None
    cands = list(p.glob(f"checkpoint_{run_id}_epoch*.pth"))
    if not cands:
        return None
    best = None
    best_epoch = -1
    for c in cands:
        m = re.search(rf"checkpoint_{re.escape(run_id)}_epoch(\d+)", c.name)
        if m:
            ep = int(m.group(1))
            if ep > best_epoch:
                best_epoch = ep
                best = c
    return best


def load_state_dict_safely(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError:
        pass

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=False)


def save_checkpoint(
    path: Path,
    model_name: str,
    cfg: Dict[str, Any],
    crop_coords: Optional[Tuple[int, int, int, int]],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    epoch: int,
    extra: Optional[Dict[str, Any]] = None,
):
    to_save = {
        "model_name": model_name,
        "cfg": cfg,
        "crop_coords": crop_coords,
        "epoch": epoch,
        "state_dict": unwrap_model(model).state_dict(),
    }
    if optimizer is not None:
        to_save["optimizer"] = optimizer.state_dict()
    if extra is not None:
        to_save.update(extra)
    torch.save(to_save, path)


# -------------------------
# eval (val dice)
# -------------------------
@torch.no_grad()
def evaluate_val(net: nn.Module, dataloader: DataLoader, device: torch.device, amp: bool) -> float:
    """
    Foreground dice across labels 1,2,3, computed as:
    - one-hot pred/true
    - only-present-classes dice to avoid empty-class inflation
    """
    net.eval()
    dice_sum = 0.0
    n = 0

    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=amp):
        for x, y in dataloader:
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.long, non_blocking=True)

            logits = net(x)
            pred = logits.argmax(dim=1)

            C = get_n_classes(net)
            pred_oh = F.one_hot(pred, num_classes=C).permute(0, 3, 1, 2).float()
            true_oh = F.one_hot(y, num_classes=C).permute(0, 3, 1, 2).float()

            pred_fg = pred_oh[:, 1:]
            true_fg = true_oh[:, 1:]

            d = 1.0 - float(dice_loss_mc_present(pred_fg, true_fg).item())
            dice_sum += d
            n += 1

    net.train()
    return dice_sum / max(n, 1)


# -------------------------
# training
# -------------------------
def train_model(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    model_name: str,
    run_id: str,
    cfg_meta: Dict[str, Any],
    crop_coords: Optional[Tuple[int, int, int, int]],
    epochs: int,
    batch_size: int,
    lr: float,
    checkpoint_dir: str,
    val_percent: float,
    amp: bool,
    dice_w: float,
    dice_warmup_epochs: int,
    class_weights: Optional[List[float]],
    num_workers: int,
    max_steps_per_epoch: int,
    resume: bool,
    accum_steps: int,
    use_sampler: bool,
    fg_oversample: float,
):
    ckpt_root = Path(checkpoint_dir)
    ckpt_dir = ckpt_root / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)

    start_epoch = 1
    if resume:
        latest = find_latest_checkpoint(str(ckpt_dir), run_id)
        if latest is not None:
            logging.info(f"Resume from {latest}")
            ckpt = torch.load(latest, map_location=device)
            load_state_dict_safely(unwrap_model(model), ckpt["state_dict"])
            if "optimizer" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                    optimizer_to(optimizer, device)
                except Exception as e:
                    logging.warning(f"Optimizer load failed, ignore: {e}")
            start_epoch = int(ckpt.get("epoch", 0)) + 1
        else:
            logging.info("No checkpoint found, train from scratch")

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    nw = int(num_workers)
    train_loader_args = dict(batch_size=batch_size, num_workers=nw, pin_memory=True, drop_last=False)
    val_loader_args = dict(batch_size=batch_size, num_workers=nw, pin_memory=True, shuffle=False, drop_last=True)

    if nw > 0:
        train_loader_args.update(dict(persistent_workers=True, prefetch_factor=4))
        val_loader_args.update(dict(persistent_workers=True, prefetch_factor=2))

    # -------- sampler (oversample fg slices) --------
    sampler = None
    if use_sampler and hasattr(dataset, "has_fg"):
        try:
            weights = []
            for idx in train_set.indices:
                weights.append(float(fg_oversample) if dataset.has_fg[idx] else 1.0)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            logging.info(f"[Train] Using WeightedRandomSampler (fg weight={fg_oversample}).")
        except Exception as e:
            logging.warning(f"[Train] sampler build failed, fallback shuffle=True: {e}")
            sampler = None

    if sampler is not None:
        train_loader = DataLoader(train_set, sampler=sampler, shuffle=False, **train_loader_args)
    else:
        train_loader = DataLoader(train_set, shuffle=True, **train_loader_args)

    val_loader = DataLoader(val_set, **val_loader_args)

    # loss
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32, device=device)
        ce_criterion = nn.CrossEntropyLoss(weight=w)
        logging.info(f"Use weighted CE: {class_weights}")
    else:
        ce_criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler(enabled=amp)
    device_type = "cuda" if device.type == "cuda" else "cpu"

    accum_steps = max(1, int(accum_steps))
    logging.info(
        f"Start training: run_id={run_id}, model={model_name}, epochs={epochs}, batch={batch_size}, "
        f"lr={lr}, amp={amp}, workers={nw}, accum_steps={accum_steps}, sampler={sampler is not None}"
    )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sum = 0.0
        ce_sum = 0.0
        dice_sum = 0.0

        steps = 0
        total_steps = len(train_loader)
        if max_steps_per_epoch and max_steps_per_epoch > 0:
            total_steps = min(total_steps, int(max_steps_per_epoch))

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(total=total_steps, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for x, y in train_loader:
            steps += 1
            if max_steps_per_epoch and max_steps_per_epoch > 0 and steps > max_steps_per_epoch:
                break

            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.long, non_blocking=True)

            with torch.amp.autocast(device_type=device_type, enabled=amp):
                logits = model(x)
                ce = ce_criterion(logits, y)

                C = get_n_classes(model)
                prob = torch.softmax(logits, dim=1)
                true_oh = F.one_hot(y, num_classes=C).permute(0, 3, 1, 2).float()

                prob_fg = prob[:, 1:]
                true_fg = true_oh[:, 1:]

                dloss = dice_loss_mc_present(prob_fg, true_fg)

                cur_dice_w = 0.0 if epoch <= dice_warmup_epochs else float(dice_w)
                loss = ce + cur_dice_w * dloss

                # gradient accumulation (scale down each step)
                loss_to_backprop = loss / float(accum_steps)

            scaler.scale(loss_to_backprop).backward()

            do_step = (steps % accum_steps == 0)
            if do_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            loss_sum += float(loss.item())
            ce_sum += float(ce.item())
            dice_sum += float(dloss.item())

            pbar.update(1)
            pbar.set_postfix(
                loss=f"{loss.item():.3f}", ce=f"{ce.item():.3f}", dice=f"{dloss.item():.3f}", dw=f"{cur_dice_w:.2f}"
            )

        # if last batch didn't trigger optimizer step
        if (steps % accum_steps) != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        pbar.close()

        logging.info(f"Epoch {epoch} train: loss={loss_sum/max(steps,1):.4f}, ce={ce_sum/max(steps,1):.4f}, dice={dice_sum/max(steps,1):.4f}")

        val_d = evaluate_val(model, val_loader, device, amp)
        logging.info(f"Epoch {epoch} val foreground dice={val_d:.4f}")

        ckpt_path = ckpt_dir / f"checkpoint_{run_id}_epoch{epoch}.pth"
        save_checkpoint(
            ckpt_path,
            model_name=model_name,
            cfg=cfg_meta,
            crop_coords=crop_coords,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            extra={"val_fg_dice": val_d, "run_id": run_id},
        )
        logging.info(f"Saved: {ckpt_path}")


# -------------------------
# build model + run_id
# -------------------------
def build_model_and_meta(args) -> Tuple[nn.Module, str, Dict[str, Any], str]:
    model_key = args.model.lower()

    # 1) Prefer factory if you have it
    if build_model_factory is not None:
        if model_key == "flex":
            cfg_dict = dict(
                in_ch=args.in_ch,
                n_classes=args.n_classes,
                base=args.base,
                depth=args.depth,
                norm=args.norm,
                bilinear=args.bilinear,
                dropout=args.dropout,
                residual=args.residual,
            )
        elif model_key == "transunet":
            cfg_dict = dict(
                in_ch=args.in_ch,
                n_classes=args.n_classes,
                img_size=args.img_size,
                vit_name=args.vit_name,
                pretrained_path=args.pretrained_path,
                n_skip=args.n_skip,
            )
        elif model_key == "orig":
            cfg_dict = dict(in_ch=args.in_ch, n_classes=args.n_classes, bilinear=args.bilinear)
        else:
            cfg_dict = {}

        model, meta = build_model_factory(args.model, cfg_dict)
        model_name = str(meta.get("model_name", model_key))
        cfg_meta = dict(meta.get("cfg", cfg_dict))

    else:
        # 2) fallback (no factory)
        if model_key == "flex":
            cfg = UNetConfig(
                in_ch=args.in_ch,
                n_classes=args.n_classes,
                base=args.base,
                depth=args.depth,
                norm=args.norm,
                bilinear=args.bilinear,
                dropout=args.dropout,
                residual=args.residual,
            )
            model = UNetFlexible(cfg)
            model_name = "flex"
            cfg_meta = {
                "in_ch": args.in_ch, "n_classes": args.n_classes, "base": args.base, "depth": args.depth,
                "norm": args.norm, "bilinear": args.bilinear, "dropout": args.dropout, "residual": args.residual
            }

        elif model_key == "transunet":
            if build_transunet is None:
                raise RuntimeError("transunet requested but networks.transunet_wrapper.build_transunet not found.")
            cfg_meta = dict(
                in_ch=args.in_ch, n_classes=args.n_classes, img_size=args.img_size, vit_name=args.vit_name,
                pretrained_path=args.pretrained_path, n_skip=args.n_skip
            )
            model, meta = build_transunet(cfg_meta)
            model_name = str(meta.get("model_name", "transunet"))
            cfg_meta = dict(meta.get("cfg", cfg_meta))

        elif model_key == "orig":
            model = UNetOriginal(n_channels=args.in_ch, n_classes=args.n_classes, bilinear=args.bilinear)
            model_name = "orig"
            cfg_meta = dict(in_ch=args.in_ch, n_classes=args.n_classes, bilinear=args.bilinear)
        else:
            raise ValueError(f"Unknown model={args.model}")

    # ensure n_classes exists for dice/val
    if not hasattr(model, "n_classes"):
        setattr(model, "n_classes", int(cfg_meta.get("n_classes", args.n_classes)))

    # run_id
    if args.run_name and args.run_name.strip():
        run_id = safe_run_name(args.run_name)
    else:
        sha = get_git_short_sha()
        h = short_cfg_hash(cfg_meta)
        run_id = safe_run_name(f"{model_name}_{sha+'_' if sha else ''}{h}")

    return model, model_name, cfg_meta, run_id


# -------------------------
# main
# -------------------------
def main():
    setup_logger()
    torch.backends.cudnn.benchmark = True

    ap = argparse.ArgumentParser()

    ap.add_argument("--model", type=str, default="flex", help="orig/flex/transunet (or any your factory supports)")
    ap.add_argument("--run_name", type=str, default="", help="Optional experiment name (forces separate folder)")

    ap.add_argument("--train_csv", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    ap.add_argument("--in_ch", type=int, default=4)
    ap.add_argument("--n_classes", type=int, default=4)
    ap.add_argument("--bilinear", action="store_true")

    # flex
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--norm", type=str, default="gn")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--residual", action="store_true")

    # training tricks
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--val_percent", type=float, default=0.1)
    ap.add_argument("--dice_w", type=float, default=0.5)
    ap.add_argument("--dice_warmup_epochs", type=int, default=2)
    ap.add_argument("--class_weights", type=str, default=None,
                    help="comma floats, e.g. '0.05,1,1,1' (len must = n_classes)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_steps_per_epoch", type=int, default=0,
                    help="0 means full epoch; >0 means only run that many steps (quick test)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no_img_proc", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    # perf
    ap.add_argument("--cache_volumes", action="store_true", help="Cache NIfTI volumes in RAM (faster, more memory).")
    ap.add_argument("--cache_max_items", type=int, default=32)

    # sampler / imbalance
    ap.add_argument("--no_sampler", action="store_true", help="Disable fg slice oversampling sampler.")
    ap.add_argument("--fg_oversample", type=float, default=5.0, help="Sampler weight for fg slices (default 5.0).")

    # gradient accumulation (reduce GPU peak)
    ap.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps (>=1).")

    # transunet
    ap.add_argument("--img_size", type=int, default=224, help="TransUNet input size, divisible by 16")
    ap.add_argument("--vit_name", type=str, default="R50-ViT-B_16")
    ap.add_argument("--pretrained_path", type=str, default="")
    ap.add_argument("--n_skip", type=int, default=3)

    # optional: resize for non-transunet too (0 = disable)
    ap.add_argument("--resize_to", type=int, default=0,
                    help="If >0, resize all samples to resize_to (for fair comparison). "
                         "TransUNet will always use img_size.")

    args = ap.parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    # crop
    crop_optimizer = CropOptimizer(args.train_csv)
    crop_coords = crop_optimizer.find_optimal_crop()
    logging.info(f"Crop coords: {crop_coords}")

    # dataset out_size logic
    out_size: Optional[int] = None
    if args.model.lower() == "transunet":
        out_size = int(args.img_size)
    elif args.resize_to and int(args.resize_to) > 0:
        out_size = int(args.resize_to)

    dataset = BrainSegmentationDataset(
        csv_path=args.train_csv,
        crop_coords=crop_coords,
        use_img_proc=(not args.no_img_proc),
        out_size=out_size,
        cache_volumes=args.cache_volumes,
        cache_max_items=args.cache_max_items,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset has 0 slices after filtering. Check CSV paths and data integrity.")

    x0, y0 = dataset[0]
    logging.info(f"Sample image: {tuple(x0.shape)}, mask: {tuple(y0.shape)}")
    logging.info(f"Device={device.type}, num_gpus={num_gpus}")

    model, model_name, cfg_meta, run_id = build_model_and_meta(args)
    logging.info(f"Model={model_name}, run_id={run_id}, cfg={cfg_meta}")

    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    class_weights = parse_csv_floats(args.class_weights)
    if class_weights is not None and len(class_weights) != int(args.n_classes):
        raise ValueError(f"--class_weights length must be {args.n_classes}, got {len(class_weights)}")

    train_model(
        model=model,
        dataset=dataset,
        device=device,
        model_name=model_name,
        run_id=run_id,
        cfg_meta=cfg_meta,
        crop_coords=crop_coords,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        checkpoint_dir=str(args.checkpoint_dir),
        val_percent=float(args.val_percent),
        amp=bool(args.amp),
        dice_w=float(args.dice_w),
        dice_warmup_epochs=int(args.dice_warmup_epochs),
        class_weights=class_weights,
        num_workers=int(args.num_workers),
        max_steps_per_epoch=int(args.max_steps_per_epoch),
        resume=bool(args.resume),
        accum_steps=int(args.accum_steps),
        use_sampler=(not args.no_sampler),
        fg_oversample=float(args.fg_oversample),
    )


if __name__ == "__main__":
    main()
