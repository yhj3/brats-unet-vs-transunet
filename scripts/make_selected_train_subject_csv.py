# src/visualization.py
from scipy.special import erf
from scipy.ndimage import laplace
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

print("Scipy imports successful")

import nibabel as nib
print("Nibabel imports successful")

from tqdm import tqdm
import logging
from pathlib import Path
import pandas as pd
import numpy as np
print("Training Phase Visualization and data writers imports successful")

import random
import copy
import re
import os
import argparse
print("Other supplies successful")

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.nn import MaxPool2d
import torch.nn.functional as F
print("Torch supplies successful")

import torchvision
print("Line 1 supplies successful")
from torchvision import transforms
print("Line 2 supplies successful")
import torchvision.transforms.functional as TF
print("Line 3 supplies successful")

print("Torchvision imports successful")

# optional: faster inference
torch.backends.cudnn.benchmark = True


# ---------------------------
# your existing functions (kept)
# ---------------------------
def dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: torch.Tensor, target: torch.Tensor, multiclass: bool = False):
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = TF.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

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
        logits = self.outc(x)
        return logits


def Img_proc(image, _lambda=-0.8, epsilon=1e-6):
    if np.isnan(image).any() or np.isinf(image).any():
        raise ValueError("Input image contains NaN or infinity values.")

    I_img = image
    min_val = np.min(I_img)
    max_val = np.max(I_img)

    if max_val == min_val:
        return np.zeros_like(I_img)

    I_img_norm = (I_img - min_val) / (max_val - min_val + epsilon)

    max_I_img = np.max(I_img_norm)
    IMG1 = (max_I_img / np.log(max_I_img + 1 + epsilon)) * np.log(I_img_norm + 1)
    IMG2 = 1 - np.exp(-I_img_norm)
    IMG3 = (IMG1 + IMG2) / (_lambda + (IMG1 * IMG2))
    IMG4 = erf(_lambda * np.arctan(np.exp(IMG3)) - 0.5 * IMG3)

    min_IMG4 = np.min(IMG4)
    max_IMG4 = np.max(IMG4)
    if max_IMG4 == min_IMG4:
        return np.zeros_like(IMG4)

    IMG5 = (IMG4 - min_IMG4) / (max_IMG4 - min_IMG4 + epsilon)
    return IMG5


class BrainSegmentationDataset(Dataset):
    """
    ✅ debug: 增加 crop_coords/out_size/use_img_proc
    - crop_coords: (x1,x2,y1,y2)
    - out_size: int -> resize 到 (out_size,out_size)
    - use_img_proc: bool
    """
    def __init__(self, csv_path, transform=None, crop_coords=None, out_size=None, use_img_proc=True):
        self.data_summary = pd.read_csv(csv_path)
        self.subjects = self.data_summary['Subject ID'].unique()
        self.transform = transform
        self.slice_info = self._create_slice_index()
        self.crop_coords = crop_coords
        self.out_size = int(out_size) if (out_size is not None and int(out_size) > 0) else None
        self.use_img_proc = bool(use_img_proc)

    def Img_proc(image, _lambda=-0.8, epsilon=1e-6):
        if np.isnan(image).any() or np.isinf(image).any():
            raise ValueError("Input image contains NaN or infinity values.")
        I_img = image
        min_val = np.min(I_img)
        max_val = np.max(I_img)
        if max_val == min_val:
            return np.zeros_like(I_img)
        I_img_norm = (I_img - min_val) / (max_val - min_val + epsilon)
        max_I_img = np.max(I_img_norm)
        IMG1 = (max_I_img / np.log(max_I_img + 1 + epsilon)) * np.log(I_img_norm + 1)
        IMG2 = 1 - np.exp(-I_img_norm)
        IMG3 = (IMG1 + IMG2) / (_lambda + (IMG1 * IMG2))
        IMG4 = erf(_lambda * np.arctan(np.exp(IMG3)) - 0.5 * IMG3)
        min_IMG4 = np.min(IMG4)
        max_IMG4 = np.max(IMG4)
        if max_IMG4 == min_IMG4:
            return np.zeros_like(IMG4)
        IMG5 = (IMG4 - min_IMG4) / (max_IMG4 - min_IMG4 + epsilon)
        return IMG5

    def _create_slice_index(self):
        slice_info = []
        for subject_id in self.subjects:
            subject_data = self.data_summary[self.data_summary['Subject ID'] == subject_id]
            flair_path = subject_data[subject_data['Scan Type'] == 'flair']['File Path'].values[0]
            nii = nib.load(flair_path)
            depth = nii.shape[2]
            slice_info.extend([(subject_id, z) for z in range(depth)])
        return slice_info

    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, idx):
        subject_id, slice_idx = self.slice_info[idx]
        subject_data = self.data_summary[self.data_summary['Subject ID'] == subject_id]

        modalities = ['flair', 't1', 't1ce', 't2']
        slices = []

        for modality in modalities:
            file_path = subject_data[subject_data['Scan Type'] == modality]['File Path'].values[0]
            nii = nib.load(file_path)

            # ✅ debug: 只取 2D slice，避免 get_fdata() 整个 3D 读入（更快）
            image2d = np.asarray(nii.dataobj[:, :, slice_idx], dtype=np.float32)

            # normalize (avoid std=0)
            mu = float(np.mean(image2d))
            sd = float(np.std(image2d))
            if sd < 1e-6:
                image2d = image2d - mu
            else:
                image2d = (image2d - mu) / sd

            if self.use_img_proc:
                image2d = Img_proc(image2d)

            slices.append(image2d)

        images = np.stack(slices, axis=0)  # [4,H,W]

        seg_data = subject_data[subject_data['Scan Type'] == 'seg']
        if seg_data.empty:
            raise ValueError(f"Missing segmentation mask for subject {subject_id}")
        seg_path = seg_data['File Path'].values[0]
        seg_nii = nib.load(seg_path)
        seg_slice = np.asarray(seg_nii.dataobj[:, :, slice_idx], dtype=np.uint8)
        seg_slice[seg_slice == 4] = 3

        # ✅ debug: apply crop if provided
        if self.crop_coords is not None:
            x1, x2, y1, y2 = self.crop_coords
            images = images[:, x1:x2, y1:y2]
            seg_slice = seg_slice[x1:x2, y1:y2]

        # ✅ debug: resize to out_size if provided (needed for TransUNet img_size=320 etc.)
        if self.out_size is not None:
            img_t = torch.from_numpy(images).unsqueeze(0)  # [1,4,H,W]
            img_t = F.interpolate(img_t, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
            images = img_t.squeeze(0).numpy()

            m_t = torch.from_numpy(seg_slice).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
            m_t = F.interpolate(m_t, size=(self.out_size, self.out_size), mode="nearest")
            seg_slice = m_t.squeeze(0).squeeze(0).to(torch.long).numpy()

        if self.transform:
            images, seg_slice = self.transform(images, seg_slice)

        return torch.tensor(images, dtype=torch.float32), torch.tensor(seg_slice, dtype=torch.long)


# ---------------------------
# keep your training/eval funcs (unchanged)
# ---------------------------
def train_model(
    model,
    dataset,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    amp: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
    pin_memory=False,
    checkpoint_dir: str = "./checkpoints"
):
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    logging.info(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{epochs}", unit="batch") as pbar:
            for images, masks in train_loader:
                images = images.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    predictions = model(images)
                    if model.n_classes == 1:
                        loss = criterion(predictions.squeeze(1), masks.float())
                        loss += dice_loss(torch.sigmoid(predictions.squeeze(1)), masks.float(), multiclass=False)
                    else:
                        loss = criterion(predictions, masks)
                        loss += dice_loss(
                            torch.softmax(predictions, dim=1),
                            torch.nn.functional.one_hot(masks, num_classes=model.n_classes)
                                .permute(0, 3, 1, 2)
                                .float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(1)
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        logging.info(f"Epoch {epoch} - Training loss: {epoch_loss:.4f}")

        val_score = evaluate(model, val_loader, device, amp)
        logging.info(f"Epoch {epoch} - Validation Dice Score: {val_score:.4f}")

        if save_checkpoint:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path / f"checkpoint_epoch{epoch}.pth")
            logging.info(f"Checkpoint saved at epoch {epoch}")


def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)


def load_model_ckpt(model, ckpt_path, device):
    # ✅ debug: try weights_only=True first (reduce warning), fallback if not supported
    try:
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(ckpt_path, map_location=device)

    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd, strict=False)
    return model


@torch.inference_mode()
def visualize_predictions_png(
    model,
    dataset,
    device,
    out_dir="./viz",
    num_samples=8,
    seed=0,
    amp=True,
):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False).tolist()

    cmap = ListedColormap(["black", "red", "green", "blue"])

    for k, idx in enumerate(idxs):
        x, y = dataset[idx]              # x: [4,H,W], y: [H,W]
        x_b = x.unsqueeze(0).to(device, non_blocking=True)  # [1,4,H,W]
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


# ---------------------------
# debug helpers (NEW, not deleting your functions)
# ---------------------------
def strip_module_prefix(state):
    out = {}
    for k, v in state.items():
        out[k[len("module."):]] = v if k.startswith("module.") else v
        if not k.startswith("module."):
            out[k] = v
    # fix duplicate writing logic
    clean = {}
    for k, v in out.items():
        clean[k] = v
    return clean


def load_ckpt_any(path: str, device: torch.device):
    """
    Returns: (meta_dict, state_dict)
    meta_dict may be empty for old checkpoints.
    """
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


def smoke_test_dataset(csv_path: str):
    dataset = BrainSegmentationDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    for batch_idx, (images, masks) in enumerate(dataloader):
        print(f"Batch {batch_idx+1}")
        print(f"Images shape: {images.shape}")
        print(f"Masks shape: {masks.shape}")
        break


# ---------------------------
# MAIN (fixed)
# ---------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./viz")
    ap.add_argument("--num_samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no_img_proc", action="store_true")
    ap.add_argument("--force_out_size", type=int, default=0)
    ap.add_argument("--crop_coords", type=str, default="")  # "40,197,29,223"
    ap.add_argument("--smoke_test", action="store_true")
    args = ap.parse_args()

    csv_path = args.csv
    CKPT = args.ckpt

    if args.smoke_test:
        smoke_test_dataset(csv_path)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[viz] device = {device}")

    meta, state = load_ckpt_any(CKPT, device)

    # parse crop_coords from arg if provided
    crop_coords = meta.get("crop_coords", None)
    if crop_coords is None and args.crop_coords.strip():
        parts = [int(x.strip()) for x in args.crop_coords.split(",")]
        if len(parts) != 4:
            raise ValueError("--crop_coords must be like '40,197,29,223'")
        crop_coords = tuple(parts)

    model_name = meta.get("model_name", "")
    cfg = meta.get("cfg", {}) or {}

    # out_size: prefer CLI force; else ckpt cfg img_size (TransUNet needs this)
    out_size = None
    if int(args.force_out_size) > 0:
        out_size = int(args.force_out_size)
    elif isinstance(cfg, dict) and "img_size" in cfg:
        try:
            out_size = int(cfg["img_size"])
        except Exception:
            out_size = None

    print(f"[viz] ckpt={CKPT}")
    print(f"[viz] meta.model_name={model_name}  cfg={cfg}  crop_coords={crop_coords}  out_size={out_size}")

    # ✅ build correct model if possible (TransUNet/flex/orig etc.)
    model = None
    if model_name and isinstance(cfg, dict) and len(cfg) > 0:
        try:
            from networks.model_factory import build_model
            model, _ = build_model(model_name, cfg)
            model.load_state_dict(state, strict=False)
            model = model.to(device)
            print(f"[viz] built model via build_model('{model_name}', cfg)")
        except Exception as e:
            print(f"[viz] WARN: build_model failed ({e}), fallback to UNet(4,4)")

    # fallback: your UNet
    if model is None:
        model = UNet(n_channels=4, n_classes=4, bilinear=False).to(device)
        model = load_model_ckpt(model, CKPT, device)

    dataset = BrainSegmentationDataset(
        csv_path=csv_path,
        crop_coords=crop_coords,
        out_size=out_size,
        use_img_proc=(not args.no_img_proc),
    )

    visualize_predictions_png(
        model=model,
        dataset=dataset,
        device=device,
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        amp=bool(args.amp),
    )
