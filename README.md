# BraTS Brain Tumor Segmentation: U-Net vs. TransUNet

A unified 2D segmentation pipeline for BraTS-style multi-modal brain MRI, built so that
**architectural effects can be isolated from preprocessing and loss design**. One training
entrypoint (`src/unet_v3.py`) instantiates three model families over an identical data path,
split rule, and metric:

| `--model` | Architecture |
|---|---|
| `orig` | Classic U-Net, BatchNorm double-convolution blocks |
| `flex` | Configurable U-Net — tunable depth, base width, normalization (Batch/Group/Instance/Layer/none), bilinear vs. transposed-conv upsampling, optional residual blocks |
| `transunet` | Hybrid ResNet50 + ViT-B/16 encoder with a convolutional decoder, plus an optional lightweight self-attention block that refines one skip connection |

Input is 4-channel axial slices (FLAIR, T1, T1-CE, T2) with 4 classes; the BraTS label `4` is
remapped to `3` at load time.

## Results

Best checkpoints on the BraTS validation slices. **FG Dice** is averaged over labels 1–3 and
computed only over slices where the class is present (see [Evaluation](#evaluation) — this
matters). Every row uses the same split rule and the same metric.

| Method | Input res. | mean CE ↓ | FG Dice ↑ | Dice 1 | Dice 2 | Dice 3 |
|---|---|---|---|---|---|---|
| Original U-Net | crop | 0.0437 | 0.7326 | 0.6826 | 0.6800 | 0.8351 |
| Flex U-Net (norm + depth only) | crop | 0.3850 | 0.6323 | 0.5395 | 0.6475 | 0.7100 |
| Flex U-Net (loss improved) | crop | 0.1952 | 0.7033 | 0.6048 | 0.7100 | 0.7952 |
| **TransUNet baseline** | 320² | 0.0455 | **0.7575** | 0.6750 | 0.7652 | 0.8322 |
| TransUNet (skip refinement + aug) | 224² | 0.0411 | 0.7347 | 0.6493 | 0.7455 | 0.8094 |

Three things worth reading off this table:

1. **Loss and sampling design matter as much as the backbone.** Making the U-Net deeper and
   swapping BatchNorm for GroupNorm *with the loss untouched* dropped FG Dice to 0.6323 — worse
   than the plain baseline, with mean CE rising to 0.3850. Reworking only the objective and
   sampler on that same architecture (present-class foreground Dice with warmup, class
   weighting, foreground oversampling) recovered roughly **+7 Dice points**.
2. **The transformer's advantage is local to one subregion.** TransUNet wins overall (+2.5
   points), but on the compact structures the tuned U-Net is competitive or better (Dice 3:
   0.8351 vs. 0.8322). Essentially all of the gain sits in label 2 — the diffuse subregion —
   at 0.7652 vs. 0.6800.
3. **Better calibration can cost Dice.** The skip-refinement variant improved mean CE
   (0.0455 → 0.0411) while *lowering* FG Dice (0.7575 → 0.7347): better calibrated on average,
   more conservative on small lesions.

Training curves for these runs are in `curve_plots/` and `curve_plots_v2/`; the per-epoch
numbers behind them are the CSVs in the repository root. Qualitative predictions are in
`viz/jobA_epoch19/`.

## Repository layout

```
src/unet_v3.py                  training entrypoint (all three model families)
src/eval.py                     evaluation; reconstructs a model from checkpoint metadata
src/data_preprocessing.py       cropping, slice indexing, normalization, sampling
src/visualization.py            overlay renders of predictions
src/networks/model_factory.py   --model string -> module
src/networks/unet_flexible.py   configurable U-Net
src/networks/unet_original.py   U-Net baseline
src/networks/transunet_wrapper.py   4->3 channel projection, patch-grid fix, skip refinement
src/networks/transunet/         upstream TransUNet modules (see Attribution)
scripts/make_brats2020_csv.py   build the slice-index CSV from a BraTS directory tree
scripts/make_selected_train_subject_csv.py   subject-level subset selection
plot_curves.py                  regenerate the figures in curve_plots/
examples/basic_inference.py     single-volume inference example
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or requirements_no_torch.txt if torch is preinstalled
```

Trained on 4× A100-40GB with mixed precision; a single 40GB card is enough for the TransUNet
run at 320² with gradient accumulation.

## Data preparation

The BraTS data is **not** in this repository. Download it from the
[BraTS challenge](https://www.med.upenn.edu/cbica/brats2020/data.html) (registration required),
then build the slice index:

```bash
python scripts/make_brats2020_csv.py --help    # expected directory layout and options
```

Each subject needs all four modalities plus `seg`. The script filters invalid subjects, keeps
only slices containing non-zero brain signal, and stores a per-slice foreground flag used later
for oversampling.

## Reproducing the table

```bash
# U-Net baseline
python src/unet_v3.py --model orig --train_csv data/brats_train.csv \
  --epochs 15 --amp --dice_w 0.5 --dice_warmup_epochs 2 --fg_oversample 5.0 \
  --run_name unet_orig

# Configurable U-Net, with the improved objective
python src/unet_v3.py --model flex --train_csv data/brats_train.csv \
  --epochs 15 --amp --norm gn --base 32 --depth 4 \
  --dice_w 0.5 --dice_warmup_epochs 2 --fg_oversample 5.0 \
  --run_name unet_flex_loss

# TransUNet baseline — the 0.7575 row
python src/unet_v3.py --model transunet --train_csv data/brats_train.csv \
  --epochs 15 --amp --img_size 320 --vit_name R50-ViT-B_16 --n_skip 3 \
  --dice_w 0.5 --dice_warmup_epochs 2 --fg_oversample 5.0 --accum_steps 2 \
  --run_name transunet_base
```

Run `python src/unet_v3.py --help` for the full flag list. The knobs that turned out to matter
most: `--dice_warmup_epochs`, `--fg_oversample`, `--class_weights`, `--norm`, and `--img_size`
(TransUNet requires a multiple of 16).

**Reproducibility.** Each checkpoint stores the model configuration *and* the crop coordinates
used during training alongside the weights, so evaluation reconstructs the exact model without
manual bookkeeping:

```bash
python src/eval.py --checkpoint checkpoints/transunet_base/best.pth
python plot_curves.py            # regenerate curve_plots/ from the CSVs
```

This is why the ablation rows above are comparable: the differences between them are exactly the
flags named in each row, and nothing else.

## Evaluation

Foreground Dice is computed over **present classes only** — classes that actually appear in the
ground truth of the batch:

```
Dice_k = 2·Σ p_k t_k / (Σ p_k + Σ t_k),   m_k = 1[Σ t_k > 0]
L_dice = 1 − Σ_k m_k · Dice_k / Σ_k m_k
```

Including absent classes inflates the score: a model gets credit for correctly predicting
"nothing here" on slices where a class does not exist, which is most slices. The reported
numbers are not comparable to Dice computed over all classes.

Training loss is class-weighted cross-entropy plus this Dice term, with the Dice weight warmed
up over the first `--dice_warmup_epochs` epochs for stability.

## Pretrained weights

The TransUNet checkpoint is hosted on the Hugging Face Hub (too large for GitHub):

**https://huggingface.co/yihangj3/brats-transunet**

```bash
hf download yihangj3/brats-transunet checkpoint_improved_transunet_epoch20.pth
```

## Attribution

- `src/networks/transunet/` (`vit_seg_modeling.py`, `vit_seg_modeling_resnet_skip.py`,
  `vit_seg_config*.py`) is adapted from the official
  [TransUNet](https://github.com/Beckschen/TransUNet) implementation by Chen et al. (2021).
  My additions are in `src/networks/transunet_wrapper.py`: the learnable 1×1 projection from
  4 MRI channels to the 3-channel pretrained backbone, patch-grid consistency for the hybrid
  encoder at non-default input sizes, and the optional decoder-side skip refinement block.
- The U-Net baseline follows Ronneberger et al. (2015); the hybrid encoder uses ResNet50
  (He et al., 2016) and ViT (Dosovitskiy et al., 2021).
- Everything else in `src/`, `scripts/`, and `plot_curves.py` is my own.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Yihang Jiao — yihangj3@illinois.edu ·
[Project write-up with full analysis](https://yhj3.github.io/projects/brats.html)
