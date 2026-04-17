# DR-NRT — Deep Codebase Research Report

> Generated: 2026-04-17. Covers everything in the repository as of the last commit on `main`.

---

## 1. Project Overview

**DR-NRT** (Diabetic Retinopathy — Ordinal-aware Representation & Training) is a deep learning research project for automated severity grading of Diabetic Retinopathy (DR) from retinal fundus photographs. It is a thesis project.

The task is a 5-class **ordinal classification** problem:

| Grade | Label | Clinical meaning |
|-------|-------|-----------------|
| 0 | No DR | No diabetic retinopathy |
| 1 | Mild DR | Microaneurysms only |
| 2 | Moderate DR | More than microaneurysms but less than severe |
| 3 | Severe DR | Any one of: 20+ intraretinal hemorrhages, venous beading, IRMA |
| 4 | Proliferative DR | Neovascularisation, vitreous/pre-retinal hemorrhage |

The "ordinal" aspect is crucial: grades are not independent categories but represent a clinical progression. A Grade 2 prediction for a true Grade 3 case is less harmful than a Grade 0 prediction for the same case. This ordinal structure motivates several design choices throughout the codebase.

**Primary metric**: Quadratic Weighted Kappa (QWK) — penalises errors by the square of the distance between predicted and true grade.

**Stack**: Python 3.x, PyTorch ≥ 2.0, torchvision, Albumentations, coral-pytorch, scikit-learn, OpenCV, scipy.

---

## 2. Research Goals & Hypotheses

The project investigates three central questions:

1. **Does ordinal-aware contrastive pretraining (OrdSupCon) improve downstream grading performance?** The hypothesis: encoding the ordinal structure of DR grades directly into the representation space — so that features of Grade 2 sit between features of Grade 1 and Grade 3 in embedding space — improves classification, especially for the minority classes (Severe, Proliferative) whose small sample count makes them hard to separate.

2. **Can cross-dataset pretraining on EyePACS (35 K images) transfer to APTOS (3 K images)?** EyePACS is much larger but has an even more extreme class imbalance and a slight domain shift. The question is whether scale compensates for shift.

3. **What combination of loss function, regularisation, sampling strategy, and inference tricks maximises both QWK and per-class F1?** These two objectives are in tension: QWK rewards ordinal closeness and can be gamed by always predicting the central class (Moderate), while per-class Macro F1 requires correctly identifying every grade.

---

## 3. Datasets

### 3.1 APTOS 2019 (primary)

- **Train**: 3,112 PNG images (after dropping 2 corrupt files from the original 3,114)
- **Test**: 550 PNG images
- **Split used in code**: stratified 85 / 15 train-val on the 3,112 training images
- **Class distribution (train set)**:

| Grade | Count | % |
|-------|-------|---|
| 0 — No DR | 1,534 | 49.3% |
| 1 — Mild | 314 | 10.1% |
| 2 — Moderate | 849 | 27.3% |
| 3 — Severe | 164 | 5.3% |
| 4 — Proliferative | 251 | 8.1% |

- **Test distribution**: 271 / 56 / 150 / 29 / 44 = 550 total
- **Image format**: PNG, variable resolution (~2000×2000 on average)
- **Preprocessing**: Ben Graham's method (see §5.2)

### 3.2 EyePACS (contrastive pretraining only)

- **Size**: 35,108 JPEG images (35,074 valid after filtering 34 blank images)
- **Grade distribution**: ~73.5% Grade 0, heavy long tail
- **Purpose**: large-scale dataset for OrdSupCon pretraining in Phase B experiments
- **Domain shift**: slightly different imaging equipment and camera settings vs APTOS
- **Note**: EyePACS test set has relatively fewer normal cases than APTOS — distribution mismatch

### 3.3 IDRiD Disease Grading (supplement)

- **Size**: 516 images total (413 train + 103 test)
- **Grade distribution**: 168 G0, 168 G2, 93 G3, 62 G4 (notably more balanced than APTOS)
- **Purpose**: inject additional Grade 3+4 images to combat APTOS data scarcity for these grades
- **Selection**: 155 Grade 3+4 images selected for experiment C1
- **Extra labels**: DME (Diabetic Macular Edema) risk flags available but unused

### 3.4 Offline Oversampling

A separate pipeline (`scripts/offline_oversample.py`) generates synthetic training images by applying Level 1.5 augmentations to minority-class originals until each class reaches 1,000 samples. The output lives in `data/train_oversampled/`. This is "offline" (pre-generated and saved to disk) rather than online (on-the-fly at training time), which makes oversampling reproducible and allows visual inspection.

---

## 4. Directory Structure

```
dr-nrt/
│
├── src/                          # All core Python modules
│   ├── __init__.py
│   ├── config.py                 # 656 lines: experiment registry + ExpConfig dataclass
│   ├── dataset.py                # 294 lines: preprocessing, dataset classes, data loaders
│   ├── models.py                 # ~143 lines: GeM, ProjectionHead, build_model(), build_contrastive_model()
│   ├── losses.py                 # 171 lines: FocalLoss, CORN, CumLink, EMD, OrdSupConLoss
│   ├── train.py                  # ~800 lines: contrastive pretraining, supervised training, evaluation
│   ├── evaluate.py               # 251 lines: metrics, OptimizedRounder, ordinal decoders, plots
│   ├── transforms.py             # 81 lines: Albumentations pipelines for 4 augmentation levels
│   ├── ensemble.py               # 131 lines: multi-backbone inference and averaging
│   ├── tta.py                    # 81 lines: TTA prediction, standard batch prediction
│   └── pseudo_label.py           # 110 lines: test-set pseudo-labeling and fine-tuning
│
├── scripts/
│   ├── offline_oversample.py     # generate synthetic minority-class images
│   ├── threshold_optimize.py     # optimize decision thresholds on val set (general)
│   ├── threshold_optimize_exp501.py  # threshold opt specific to exp 501
│   ├── mc_dropout_eval.py        # Monte Carlo Dropout uncertainty estimation
│   ├── preprocess_idrid.py       # preprocess IDRiD images to standard format
│   └── append_results_section.py
│
├── notebooks/                    # Jupyter notebooks (EDA, feature visualisation, analyses)
│
├── docs/
│   ├── 01-ideas.md               # ~55 KB: baseline experiment design
│   ├── 02-ideas-next.md          # ~56 KB: extended experiment ideas
│   ├── 03-ordinal-supcon.md      # ~97 KB: core OrdSupCon research document
│   ├── 04-coral.md               # ~35 KB: ordinal loss experiments (CORN/CumLink/EMD)
│   ├── 05-phase-h.md             # ~37 KB: LP-FT strategy
│   ├── thesis-draft.md           # ~39 KB: thesis manuscript draft
│   ├── analysis_exp501_results.md    # deep post-hoc analysis of exp 501 (F2)
│   ├── qwk_moderate_collapse_analysis.md  # analysis of the QWK centrality bias problem
│   ├── section12_results.md      # full results table for Phases D & F (up to 2026-04-16)
│   └── BÁO CÁO TIẾN ĐỘ 1.md     # Vietnamese-language progress report
│
├── data/                         # (gitignored) datasets
│   ├── data_split/
│   │   ├── train_label.csv       # id_code, diagnosis
│   │   ├── test_label.csv        # id_code, diagnosis
│   │   ├── train_split/          # 3,112 PNG training images
│   │   └── test_split/           # 550 PNG test images
│   ├── eyepacs_processed/        # preprocessed EyePACS images
│   ├── train_oversampled/        # offline-augmented minority class images
│   └── idrid_processed/          # preprocessed IDRiD images
│
├── checkpoints/                  # (gitignored) model weights per experiment
│   └── exp{ID}_{name}/
│       ├── *_best.pth            # best val-QWK checkpoint
│       ├── *_backbone.pth        # contrastive pretrain backbone only
│       └── *_swa.pth             # SWA-averaged weights
│
├── results/                      # (gitignored) outputs per experiment
│   └── exp{ID}_{name}/
│       ├── *.log
│       ├── *_cm.png              # confusion matrix
│       ├── *_cls_report.txt      # classification report
│       ├── *_training_curves.png
│       ├── *_preds.csv           # per-sample predictions
│       └── *_contrastive_log.csv # contrastive pretraining loss curve
│
├── Eyepacs/
│   └── resized_train/            # 35 K resized JPEG images
│
├── B_Disease_Grading/            # IDRiD raw images
│
├── .agents/thesis-write/SKILL.md # Claude agent skill definition for thesis writing
├── run_experiment.py             # 220 lines: main CLI entry point
├── run_all.sh                    # batch runner for experiments 12-14
├── run_phase_subA.sh             # batch runner for Phase A v2 (102-103)
└── requirements.txt              # 12 top-level dependencies
```

---

## 5. Source Code — Module by Module

### 5.1 `src/config.py` (656 lines)

The configuration hub. Every experiment is defined here and nowhere else.

**Constants**:
```
IMAGE_SIZE = 512
NUM_CLASSES = 5
CLASS_COUNTS = [1534, 314, 849, 164, 251]   # per-class train counts
TOTAL_TRAIN = sum(CLASS_COUNTS) = 3112
ImageNet mean = [0.485, 0.456, 0.406]
ImageNet std  = [0.229, 0.224, 0.225]
```

**`ExpConfig` dataclass** — 50+ fields covering every training knob:

| Category | Fields |
|----------|--------|
| Identity | `exp_id`, `exp_name` |
| Architecture | `backbone` (resnet50/efficientnet_b4/convnext_small), `use_gem`, `gem_p`, `num_outputs`, `head_dropout` |
| Training | `total_epochs`, `freeze_epochs`, `batch_size`, `lr_head`, `lr_finetune`, `weight_decay` |
| LR schedule | `lr_schedule` (step/cosine/cosine_warm), `step_size`, `step_gamma`, `cosine_T0`, `cosine_Tmult` |
| Loss | `loss_type` (ce/focal/smoothl1/corn/cumlink/emd), `focal_gamma`, `label_smoothing`, `use_class_weights` |
| Augmentation | `aug_level` (0/1/2), `mixup_alpha`, `cutmix_alpha` |
| Sampling | `use_weighted_sampler`, `oversample_dir` |
| Contrastive pretraining | `use_contrastive_pretrain`, `contrastive_data` (aptos/eyepacs), `contrastive_epochs`, `contrastive_lr`, `contrastive_temperature`, `contrastive_proj_dim` |
| Joint contrastive | `use_joint_contrastive`, `contrastive_lambda`, `joint_warmup_epochs`, `detach_backbone_for_proj` |
| Extra data | `extra_img_dirs`, `use_eyepacs_pretrain_ckpt` |
| SWA | `use_swa`, `swa_start_epoch` |
| Pseudo-label | `use_pseudo_label`, `pseudo_weight`, `pseudo_epochs` |
| Inference | `use_tta`, `use_optimized_thresholds`, `load_checkpoint` |
| LP-FT | `layerwise_lr_decay` |

**`EXPERIMENTS` dict**: maps `exp_id → ExpConfig`. ~50 defined experiments across all phases.

**`get_config(exp_id)`**: returns a deep-copied `ExpConfig` and sets derived paths (`ckpt_dir`, `results_dir`).

---

### 5.2 `src/dataset.py` (294 lines)

#### Ben Graham Preprocessing

```python
def ben_graham_preprocess(img: np.ndarray, image_size: int = 512) -> np.ndarray:
    # 1. Grayscale + threshold to find retinal content
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # 2. Find largest contour → crop to bounding box
    contours, _ = cv2.findContours(mask, ...)
    x, y, w, h = cv2.boundingRect(max_contour)
    img = img[y:y+h, x:x+w]
    # 3. Resize to square
    img = cv2.resize(img, (image_size, image_size))
    # 4. Local contrast enhancement (unsharp masking)
    gauss = cv2.GaussianBlur(img, (image_size // 30 * 2 + 1, ...), image_size / 30)
    img = cv2.addWeighted(img, 4, gauss, -4, 128)
    return img
```

The `addWeighted(img, 4, gauss, -4, 128)` step subtracts a blurred version of the image from itself (amplified 4×), leaving only local high-frequency features (microvasculature, hemorrhages, exudates), then adds 128 as a neutral gray bias. This dramatically improves visibility of pathological features.

#### Dataset Classes

**`DRDataset`**:
- Reads `id_code → label` from CSV
- Supports `extra_img_dirs` (list of additional image directories, used for IDRiD supplement)
- Supports `oversample_dir` (pre-generated augmented images)
- Applies Ben Graham preprocessing + transforms + ImageNet normalisation
- `is_regression=True` mode: returns label as float for SmoothL1 regression head

**`ContrastiveDRDataset`**:
- Wraps `DRDataset`
- For each sample, applies the training augmentation **twice independently**
- Returns `(view1, view2, label, id_code)` — the two views are used as the positive pair in OrdSupCon

**`PseudoLabelDataset`**:
- Concatenates real training samples + pseudo-labeled test samples
- Each sample carries a `weight` (real: 1.0, pseudo: `cfg.pseudo_weight`, default 0.5)
- The weight is used in the training loop to scale individual sample losses

**`build_datasets(cfg)`**:
- Loads train/val/test CSVs
- Stratified 85/15 split on training set
- Returns `(train_dataset, val_dataset, test_dataset)`
- Optionally injects IDRiD images via `extra_img_dirs`

**`build_eyepacs_dataset(cfg)`**:
- Scans `Eyepacs/resized_train/` directory
- Filters 34 blank images detected during preprocessing
- Returns dataset for contrastive pretraining

---

### 5.3 `src/models.py` (143 lines)

#### GeM Pooling

```python
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        self.p = nn.Parameter(torch.ones(1) * p)  # learnable

    def forward(self, x):
        return avg_pool2d(x.clamp(min=eps).pow(self.p), (H, W)).pow(1/self.p)
```

Generalised mean pooling: `p=1` is average pooling, `p→∞` is max pooling. Learnable `p` lets the model find the optimal interpolation. For fine-grained recognition tasks (like detecting small retinal lesions), `p≈3` has been found empirically better than average pooling.

Replaces `model.avgpool` in ResNet-50 and `model.avgpool` in EfficientNet-B4.

#### ProjectionHead (for contrastive learning)

```
Linear(2048 → 512, bias=False) → BatchNorm1d → ReLU → Linear(512 → 128, bias=False) → BatchNorm1d → L2-normalise
```

Used only during contrastive pretraining and joint contrastive fine-tuning. The projector is discarded after pretraining — only the backbone weights are kept.

#### `build_model(cfg)` factory

- **ResNet-50**: IMAGENET1K_V2 weights (better than V1). Optional GeM. Optional `head_dropout`. FC output dim = 5 for classification, 4 for CORN/CumLink (K-1 outputs), 1 for regression.
- **EfficientNet-B4**: IMAGENET1K_V1 weights. Optional GeM on `model.avgpool`. No CORN/CumLink support (raises `NotImplementedError`).
- **ConvNeXt-Small**: IMAGENET1K_V1 weights. Replaces `model.classifier[2]`. No CORN/CumLink support.

#### `build_contrastive_model(cfg)`

Returns `(backbone, projector)` where:
- `backbone` = ResNet-50 with `fc = nn.Identity()` (outputs 2048-dim feature vector)
- `projector` = `ProjectionHead(2048 → 512 → cfg.contrastive_proj_dim)`

Only ResNet-50 is supported for contrastive pretraining.

#### Freeze/Unfreeze Utilities

- `freeze_backbone(model, backbone)`: freezes all params except `fc` (ResNet-50) or `classifier` (EfficientNet/ConvNeXt)
- `unfreeze_all(model)`: re-enables all parameters

---

### 5.4 `src/losses.py` (171 lines)

#### FocalLoss

```python
log_probs = F.log_softmax(logits, dim=1)
probs = log_probs.exp()
focal_weight = (1 - probs).pow(gamma)          # down-weight easy examples
loss = -focal_weight * one_hot * log_probs      # per-class, per-sample
if alpha is not None:
    loss = loss * alpha.unsqueeze(0)             # class reweighting
return loss.sum(dim=1).mean()
```

- `gamma=2.0` default. Supports label smoothing by blending one-hot with uniform prior.
- Class weights computed as `(TOTAL_TRAIN / CLASS_COUNTS[k]) / sum * NUM_CLASSES` — inverse frequency, renormalised to preserve scale.

#### CORNLoss

Thin wrapper around `coral_pytorch.corn_loss(logits, targets, num_classes=5)`. CORN (Consistent Rank Logits) models ordinal classification as a chain of binary decisions: P(grade≥1), P(grade≥2|grade≥1), ..., P(grade≥4|grade≥3). The model outputs 4 logits (K-1) instead of 5.

#### CumulativeLinkLoss

```python
levels = torch.arange(1, K)         # [1, 2, 3, 4]
binary_targets = (targets >= levels) # [N, K-1], e.g. grade 3 → [1, 1, 1, 0]
loss = F.binary_cross_entropy_with_logits(logits, binary_targets.float())
```

K-1 independent binary classifiers: "is grade ≥ k?". Preserves ordinal monotonicity assumption. Model outputs 4 logits.

#### EMDLoss

```python
cdf_pred = torch.cumsum(F.softmax(logits, dim=1), dim=1)
cdf_true = torch.cumsum(F.one_hot(targets, K).float(), dim=1)
emd = (cdf_pred - cdf_true).pow(2).sum(dim=1).mean()
```

Earth Mover's Distance between predicted and true cumulative distribution functions. Penalises distributional mismatch in a way that respects ordinal structure: predicting Grade 2 for a true Grade 3 case is less penalised than predicting Grade 0 for the same case.

#### OrdSupConLoss (the core innovation)

```python
# Ordinal weight matrix
dist = |g_i - g_j|             # absolute grade difference, [2N, 2N]
W = 1.0 - dist / (K - 1)       # W(0,4)=0.0, W(1,2)=0.75, W(2,2)=1.0

# Cosine similarity (features already L2-normalised)
sim = features @ features.T / temperature   # temperature=0.07

# Mask diagonal (self-similarity)
sim.masked_fill_(eye_mask, -1e4)   # -1e4 is FP16-safe (avoids -inf NaN)

# Weighted log-softmax contrastive loss
log_prob = sim - logsumexp(sim, dim=1)
Z_i = sum_{j≠i} W(i,j)            # per-anchor normalisation
loss = -mean( sum_j W(i,j) * log_prob(i,j) / Z_i )
```

The weight matrix `W(i,j)` is the key design: same-grade pairs have W=1.0 (pull strongly together), adjacent-grade pairs have W=0.75 (pull moderately), grade-4-apart pairs have W=0.0 (no pull, act as free negatives). This differs from standard SupCon where all same-class pairs have W=1 and all cross-class pairs have W=0 — here the boundaries are soft and ordinal.

**Critical implementation detail**: the diagonal masking uses `-1e4` not `-inf`. This is because with FP16 mixed precision, `masked_fill_(-inf)` followed by `logsumexp` can produce NaN (the `0 * (-inf)` problem was the source of a real bug fixed in commit `36cdabe`).

**Another detail**: `log_prob.clamp(min=-100)` before weighting prevents `0 * (-inf) = NaN` when W(i,j)=0 for maximally distant grade pairs.

---

### 5.5 `src/train.py` (~800 lines)

#### `run_contrastive_pretraining(cfg, device)`

```
1. Build backbone + projector via build_contrastive_model(cfg)
2. Load ContrastiveDRDataset (APTOS or EyePACS depending on cfg.contrastive_data)
3. For each epoch:
   a. Sample batch of (view1, view2, labels)
   b. Concatenate: features = [view1, view2] → shape [2N, C, H, W]
   c. Forward through backbone → [2N, 2048]
   d. Forward through projector → [2N, 128], L2-normalised
   e. Compute OrdSupConLoss(features, labels_repeated_2x)
   f. Backprop through both backbone and projector
4. Save backbone.state_dict() only (projector discarded)
5. Return path to saved backbone checkpoint
```

Optimiser: Adam with `lr=cfg.contrastive_lr` (default 1e-3). Schedule: CosineAnnealingLR(T_max=epochs).

#### `run_training(cfg, device, pretrained_backbone=None)`

**Stage 1 — Head training (frozen backbone):**
- Load `pretrained_backbone` weights if provided
- `freeze_backbone(model)` — only FC/classifier trainable
- Optimiser: `Adam([fc_params], lr=cfg.lr_head)` (default 1e-3)
- Runs for `cfg.freeze_epochs` epochs (typically 5–7)

**Stage 2 — Fine-tuning (unfrozen backbone):**
- `unfreeze_all(model)`
- New optimiser: `Adam(all_params, lr=cfg.lr_finetune)` (default 1e-4)
  - Optional: layer-wise LR decay (`layerwise_lr_decay` factor, e.g. 0.9 per layer group from head to stem)
- LR schedule: `StepLR`, `CosineAnnealingLR`, or `CosineAnnealingWarmRestarts`
- Mixed-precision training via `torch.amp.autocast()` + `GradScaler`
- Optional Mixup/CutMix (incompatible with CORN/CumLink — they need hard integer labels)

**Joint contrastive path (when `use_joint_contrastive=True`):**
- A forward hook is registered on `model.avgpool` to capture backbone features
- For each batch, both `view1` and `view2` are passed through the model
- Classification loss is computed on `view1` predictions only
- OrdSupCon loss is computed on the avgpool features of both views, passed through the separate projector
- Combined: `total_loss = cls_loss + cfg.contrastive_lambda * con_loss`
- Optional warmup: `lambda` ramped from 0 to `cfg.contrastive_lambda` over `joint_warmup_epochs`
- Optional detach: `detach_backbone_for_proj=True` trains projector independently (no gradient to backbone)

**Checkpoint saving**:
```python
if val_metrics["qwk"] > best_qwk:
    best_qwk = val_metrics["qwk"]
    torch.save(model.state_dict(), ckpt_path)
```
Saves based on **val QWK only**. This is a known limitation (see §9.1).

**SWA** (`use_swa=True`):
- From `swa_start_epoch` (typically 80), accumulate model parameter averages
- After training ends, update BatchNorm statistics on train set with SWA-averaged model
- Save as separate `*_swa.pth` checkpoint

**Pseudo-labeling** (calls `pseudo_label.py`):
- Generate predictions on test set with the best checkpoint
- Create `PseudoLabelDataset` mixing real + pseudo-labeled test images
- Fine-tune for `pseudo_epochs` with lower learning rate

---

### 5.6 `src/evaluate.py` (251 lines)

#### Metrics computed on every evaluation

| Metric | How computed |
|--------|-------------|
| QWK | `sklearn.metrics.cohen_kappa_score(..., weights='quadratic')` |
| Accuracy | standard |
| Macro F1 | unweighted average F1 across all 5 classes |
| Sensitivity | macro-average recall |
| Specificity | macro-average TNR |
| AUC-ROC | one-vs-rest, macro-average, `sklearn.metrics.roc_auc_score` |
| ECE | 15-bin calibration error |
| Per-class F1 | F1 for grades 0-4 individually |

#### OptimizedRounder

For regression-mode experiments (SmoothL1 head) and for threshold refinement:

```python
# scipy.optimize.minimize with Nelder-Mead
minimize(
    fun = lambda thresholds: -qwk(targets, regression_to_class(preds, thresholds)),
    x0 = [0.5, 1.5, 2.5, 3.5],
    method = 'Nelder-Mead'
)
```

Also used post-hoc on softmax classification outputs in three variants:
- **argmax**: standard argmax (no optimisation)
- **expected_grade**: optimise thresholds on the expected (weighted average) grade = `sum(k * p_k for k in 0..4)`
- **cumulative_opt**: optimise thresholds on `sum(p_k for k > level)` per level

#### Ordinal Decoders

- `corn_logits_to_probs(logits)`: chain-rule decomposition from `coral_pytorch`
- `cumlink_logits_to_probs(logits)`: `sigmoid(logits)` → K-1 cumulative probs → K class probs
- `cumlink_to_class(logits)`: `sum(sigmoid(logit_k) > 0.5)` for k=0..K-2
- `regression_to_class(pred, thresholds)`: compare scalar to 4 thresholds

---

### 5.7 `src/transforms.py` (81 lines)

All augmentations use **Albumentations**.

**Level 0**: `Compose([Resize(512, 512), ToTensorV2()])` — for evaluation

**Level 1 (standard)**:
- `HorizontalFlip(p=0.5)`, `VerticalFlip(p=0.5)`, `RandomRotate90(p=0.5)`
- `ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=180, p=0.7)`
- `RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)`
- `HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3)`

**Level 2 (advanced)**: Level 1 plus:
- `CLAHE(clip_limit=4.0, p=0.3)` — adaptive histogram equalisation
- `GaussianBlur(blur_limit=(3,7), p=0.2)`
- (CoarseDropout and ElasticTransform are present but commented out — they destroy retinal pathology markers)

**Level 1.5 (offline oversampling)**: Same as Level 1. Named "1.5" to indicate it is safe for DR images — strong enough for diversity, without destructive transforms that could erase diagnostic features.

**TTA Transforms** (6 views):
- Original, HFlip, VFlip, Rot90, Rot180, Rot270

---

### 5.8 `src/tta.py` (81 lines)

**`predict_with_tta(model, dataset, transforms, device)`**:
- For each image: apply all 6 TTA transforms, run 6 forward passes, average softmax probabilities
- Returns: `(averaged_probs, id_codes)`

**`predict_no_tta(model, loader, device)`**:
- Standard batch inference
- Returns: `(logits_or_preds, targets, id_codes)`

---

### 5.9 `src/ensemble.py` (131 lines)

**`build_ensemble_configs(base_cfg)`**: creates 3 derived configs — one per backbone (ResNet-50, EfficientNet-B4, ConvNeXt-Small) — pointing to their respective checkpoint files.

**`run_ensemble_inference(base_cfg, device)`**:
1. For each backbone: load checkpoint → run `predict_no_tta` (or with TTA if enabled)
2. Average all three models' softmax probabilities
3. Optional: apply `OptimizedRounder` to ensemble probabilities using val set
4. Compute and log ensemble metrics
5. Save ensemble predictions separately

---

### 5.10 `src/pseudo_label.py` (110 lines)

**`generate_pseudo_labels(model, test_dataset, cfg, device)`**:
- Predicts on the test set (with optional TTA)
- Returns `(id_codes, pseudo_labels, pseudo_probs)` — hard labels for simplicity

**`finetune_with_pseudo(model, train_dataset, pseudo_labels, cfg, device)`**:
- Creates `PseudoLabelDataset` combining real train + pseudo-labeled test images
- Trains for `pseudo_epochs` (default 10) at a lower LR
- The per-sample `weight` field scales the loss contribution of pseudo samples vs real samples

---

## 6. Training Pipeline — End-to-End Flow

```
python run_experiment.py --exp <ID> [--device cuda] [--workers 4]

run_experiment.py (220 lines)
│
├─ get_config(exp_id)
├─ build_datasets(cfg) → train_ds, val_ds, test_ds
│   └─ if cfg.extra_img_dirs: inject IDRiD images
│
├─ if cfg.use_contrastive_pretrain:
│   └─ run_contrastive_pretraining(cfg, device)
│       ├─ build_contrastive_model() → (backbone, projector)
│       ├─ ContrastiveDRDataset(APTOS or EyePACS)
│       ├─ Train: Adam + CosineAnnealingLR, OrdSupConLoss
│       └─ Save backbone only → cfg.ckpt_dir/*_backbone.pth
│
├─ run_training(cfg, device, pretrained_backbone=ckpt_path_or_None)
│   ├─ build_model(cfg) → model
│   ├─ build_loss(cfg, device) → criterion
│   ├─ Stage 1: freeze_backbone → train head (lr_head, freeze_epochs)
│   ├─ Stage 2: unfreeze_all → fine-tune all (lr_finetune, LR schedule)
│   │   └─ [joint] forward hook on avgpool
│   │      → cls_loss(view1) + λ * OrdSupConLoss(both views)
│   ├─ [swa] accumulate + BN update → save *_swa.pth
│   └─ Save best val-QWK checkpoint → *_best.pth
│
├─ if cfg.use_pseudo_label:
│   ├─ generate_pseudo_labels(model, test_ds)
│   └─ finetune_with_pseudo(model, train_ds, pseudo_labels, cfg)
│
├─ evaluate_on_test(model, test_ds, val_ds, cfg, device)
│   ├─ [tta] predict_with_tta or predict_no_tta
│   ├─ [optimized_thresholds] OptimizedRounder on val → apply to test
│   ├─ Compute all metrics
│   └─ Save: confusion matrix PNG, classification report, predictions CSV, curves PNG
│
└─ [exp 14 only] run_ensemble_inference(cfg, device)
    ├─ Load 3 backbone checkpoints
    ├─ Average predictions
    └─ Save ensemble results
```

---

## 7. Experiment Taxonomy

### 7.1 Phase Overview

| Phase | IDs | Focus |
|-------|-----|-------|
| Baseline | 0–13 | Progressive feature engineering |
| Ensemble | 14 | Multi-backbone aggregation |
| A | 100–103 | Ordinal baseline re-eval, sampling, OrdSupCon on APTOS |
| B | 200–201 | OrdSupCon pretraining on EyePACS |
| C | 301 | IDRiD Grade 3+4 data supplement |
| D | 300 | Regularisation recipe |
| F | 501–504 | Joint contrastive fine-tuning ablations |
| G | 600–604 | Ordinal loss functions (CORN/CumLink/EMD) on pretrained backbone |
| H | 700–704 | Linear Probe + Fine-Tune (LP-FT) with layer-wise LR decay |

### 7.2 Baseline Experiments (0–13)

Each experiment adds one capability to a ResNet-50 + CE baseline:

| Exp | Key change |
|-----|-----------|
| 0 | Bare baseline (ResNet-50, CE, no aug) |
| 1 | Aug Level 1 |
| 2 | Focal loss |
| 3 | Focal + class weights |
| 4 | Aug Level 2 |
| 5 | Weighted sampler |
| 6 | Mixup |
| 7 | Regression head (SmoothL1) |
| 8 | **GeM pooling** ← best baseline, becomes "A0" |
| 9 | Cosine LR schedule |
| 10 | SWA + GeM + Cosine |
| 11 | TTA + SWA + GeM |
| 12 | EfficientNet-B4 backbone |
| 13 | Pseudo-labeling on top of exp 8 |
| 14 | Ensemble (ResNet + EfficientNet + ConvNeXt) |

### 7.3 Phase A — Ordinal Baselines

| Exp | ID | Description |
|-----|----|-------------|
| A0 | 8 | Load exp 8 checkpoint, eval with ECE |
| A0b | 101 | + weighted sampler (inverse frequency) |
| A0c | 102 | + offline oversampling (1K/class) |
| A1-v2 | 103 | OrdSupCon pretraining on APTOS (20 ep) → fine-tune |

### 7.4 Phase B — EyePACS Pretraining

| Exp | ID | Description |
|-----|----|-------------|
| A2 | 200 | OrdSupCon on 35K EyePACS → fine-tune on APTOS |
| A2-v2 | 201 | EyePACS pretrain + longer freeze (BN recalibration) |

### 7.5 Phases C, D

| Exp | ID | Description |
|-----|----|-------------|
| C1 | 301 | D1 recipe + inject 155 IDRiD Grade 3+4 images |
| D1 | 300 | ResNet-50 + Focal + GeM + Dropout(0.3) + CosineDecay + 80 ep + offline oversample |

### 7.6 Phase F — Joint Contrastive Ablations

| Exp | ID | Description |
|-----|----|-------------|
| F2 | 501 | D1 recipe + joint OrdSupCon (λ=0.1, batch=24, lr=5e-5, freeze=7) |
| F3 | 502 | F2 but with D1's exact hyperparams (batch=32, lr=1e-4, freeze=5) |
| F4 | 503 | F2 + warmup λ schedule (0→0.1 over epochs 8–20) |
| F5 | 504 | F2 + detach_backbone_for_proj=True |

### 7.7 Phase G — Ordinal Losses

CORN, CumLink, EMD tested on pretrained backbone:

| Exp | ID | Loss |
|-----|----|------|
| G1 | 600 | CORN (random init, isolates loss effect) |
| G2 | 601 | CORN (pretrained backbone) |
| G3 | 602 | EMD |
| G4 | 603 | CumLink |
| G5 | 604 | CumLink + pretrained |

### 7.8 Phase H — LP-FT

Linear Probe + Fine-Tune strategy:

| Exp | ID | Description |
|-----|----|-------------|
| H0 | 700 | Long linear probe (20 ep) then fine-tune (10 ep, lr=1e-5) |
| H1 | 701 | LP-FT with layer-wise LR decay |
| H2 | 702 | LP-FT on OrdSupCon backbone |
| H3 | 703 | LP-FT + longer fine-tune |
| H4 | 704 | LP-FT + IDRiD supplement |

---

## 8. Experimental Results

### 8.1 Master Results Table (as of 2026-04-16)

| Experiment | Description | QWK | Macro F1 | Severe F1 | Prolif F1 | Moderate F1 | AUC-ROC | ECE |
|-----------|-------------|-----|----------|----------|----------|------------|--------|-----|
| A0 (exp8) | Focal+GeM, 50ep, StepLR | 0.9127 | **0.7040** | 0.4444 | **0.7013** | 0.7911 | 0.9462 | **0.0420** |
| A0c-v2 (102) | + Offline oversample | 0.9073 | 0.6990 | 0.4530 | 0.6020 | **0.8132** | 0.9484 | 0.0503 |
| A1-v2 (103) | OrdSupCon APTOS pretrain | 0.9056 | 0.6866 | 0.4286 | 0.6667 | 0.7603 | 0.9493 | 0.0466 |
| A2 (200) | OrdSupCon EyePACS pretrain | 0.8932 | 0.6580 | 0.4167 | 0.6279 | 0.7127 | 0.9383 | 0.0520 |
| D1 (300) | Dropout(0.3)+Cosine+80ep | 0.9159 | 0.6945 | 0.3830 | 0.6667 | 0.8221 | 0.9518 | 0.0482 |
| **D1+thresh** | D1 + cumulative threshold opt | **0.9175** | 0.6893 | 0.4231 | 0.6494 | 0.8182 | 0.9518 | 0.0482 |
| F2 (501) | Joint Focal+OrdSupCon λ=0.1 | 0.8987 | 0.6746 | 0.4186 | 0.6024 | 0.7823 | **0.9520** | 0.0499 |
| **F2+thresh** | F2 + expected-grade opt | 0.9100 | 0.6946 | **0.4839** | 0.6154 | 0.7774 | **0.9520** | 0.0499 |
| F3 (502) | F2 with D1 hyperparams (collapsed) | 0.8792 | 0.6763 | 0.4074 | 0.6000 | 0.7687 | 0.9507 | 0.0344 |

**Winners by metric**:
- **QWK**: D1+thresh (0.9175)
- **Macro F1**: A0 (0.7040)
- **Severe F1**: F2+thresh (0.4839, Severe recall = 52%)
- **Prolif F1**: A0 (0.7013)
- **AUC-ROC**: F2 / F2+thresh (0.9520)
- **ECE**: A0 (0.0420)

### 8.2 D1 Confusion Matrix (best QWK model — cumulative thresholds)

| True \ Pred | No DR | Mild | Moderate | Severe | Prolif |
|-------------|:-----:|:----:|:--------:|:------:|:------:|
| **No DR (271)** | 265 | 5 | 1 | 0 | 0 |
| **Mild (56)** | 4 | 29 | 23 | 0 | 0 |
| **Moderate (150)** | 0 | 8 | 135 | 5 | 2 |
| **Severe (29)** | 0 | 0 | 12 | **11** | 6 |
| **Prolif (44)** | 0 | 3 | 9 | 7 | 25 |

Severe recall = 37.9% (11/29). Moderate recall = 90% (135/150) — best across all experiments.

### 8.3 F2 Confusion Matrix (best Severe recall — expected-grade thresholds)

| True \ Pred | No DR | Mild | Moderate | Severe | Prolif |
|-------------|:-----:|:----:|:--------:|:------:|:------:|
| **No DR (271)** | 264 | 4 | 3 | 0 | 0 |
| **Mild (56)** | 4 | 37 | 15 | 0 | 0 |
| **Moderate (150)** | 0 | 18 | 117 | 12 | 3 |
| **Severe (29)** | 0 | 1 | 7 | **15** | 6 |
| **Prolif (44)** | 0 | 3 | 7 | 10 | 24 |

Severe recall = 51.7% (15/29) — **best across all experiments**. Cost: Moderate recall drops to 78% (117/150).

### 8.4 D1 Training Dynamics

- Best val QWK: **0.9570** at epoch 61
- Val→test gap: 0.041 (val 0.957 → test 0.916/0.9175)
- Train loss: 0.715 → 0.008 (smooth cosine decay, no StepLR kink)
- Val loss stabilises at ~0.11–0.12 from epoch 30 — no overfitting

### 8.5 F2 Training Dynamics

- Best val checkpoint: epoch 75, val QWK = **0.9516**, val Macro F1 = 0.822
- Val→test gap: **0.053** (0.952 → 0.899) — significantly larger than D1
- Cause: dual-view augmentation inflates apparent sample diversity (same image seen twice per batch with correlated views)

---

## 9. Key Findings & Insights

### 9.1 Regularisation beats ordinal pretraining on QWK

The strongest model by QWK is D1: `Dropout(0.3) + CosineAnnealingLR + 80 epochs + offline oversampling`. This beats both OrdSupCon pretraining variants (A1-v2: 0.9056, A2: 0.8932) and the joint contrastive approach (F2: 0.8987). Simple regularisation is more QWK-effective than representation learning on this dataset scale.

### 9.2 Joint OrdSupCon wins on Severe recall at the cost of QWK

F2+threshold achieves Severe F1 = 0.484 (recall = 52%), compared to D1+threshold's Severe F1 = 0.423 (recall = 38%). This is a clinically significant difference — 6 additional Severe patients correctly detected. But QWK is −0.008 lower than D1. The statistical metric and the clinical metric diverge.

### 9.3 Gradient competition is the primary failure mode for joint training

F3 applied the same joint contrastive loss as F2 but used D1's exact hyperparameters (batch=32, lr=1e-4, freeze=5 ep). This caused a catastrophic collapse to QWK=0.879. Root cause: at higher learning rate, the contrastive gradient (pushing representations toward ordinal structure) competes destructively with the focal classification gradient (pushing toward sharp category boundaries). The joint approach requires carefully tuned hyperparameters — lower LR, smaller batch, longer freeze — to survive.

### 9.4 The Moderate class attractor

In all joint contrastive experiments, Grade 2 (Moderate, n=150, the largest non-majority class) acts as an attractor in representation space. In F2: 18 Mild, 12 Severe, and 12 Proliferative images are all predicted as Moderate. The OrdSupCon weight matrix partly explains this: W(1,2) = W(2,3) = 0.75, meaning Moderate is pulled toward both its neighbours equally — but as the largest class, it dominates the anchor pool and becomes the gravitational centre of the manifold.

### 9.5 QWK rewards centrality bias

The QWK penalty for predicting Moderate for a true Severe case is `(3-2)² = 1`. For predicting No DR for a true Severe case, it is `(3-0)² = 9`. The model can achieve high QWK by defaulting to the central class when uncertain. This is the **ordinal regression centrality bias**, and it explains why QWK (best: 0.9175) and Macro F1 (best: 0.7040) come from different models.

### 9.6 Threshold optimisation is free performance

Applying Nelder-Mead threshold optimisation on existing val-set predictions (no retraining):
- F2 argmax QWK 0.8987 → F2+expected-grade 0.9100 (+0.011)
- D1 argmax QWK 0.9159 → D1+cumulative 0.9175 (+0.002)
- F2 Severe F1: 0.4186 → 0.4839 (+0.065) via expected-grade thresholds

No experiment has been run without applying threshold optimisation in the final evaluation.

### 9.7 No experiment breaks A0's Macro F1 ceiling

A0 (the original exp 8 baseline: Focal + GeM + StepLR, 50 epochs) has Macro F1 = 0.7040. Every subsequent experiment that improves QWK or Severe F1 regresses on Macro F1. This ceiling exists because Severe and Proliferative F1 are in tension: improving Severe recall (by shifting the Severe/Moderate boundary) causes Proliferative cases to be misclassified as Severe, and vice versa.

### 9.8 AUC-ROC and QWK measure different things

F2 achieves the best AUC-ROC (0.9520) — meaning its softmax probabilities rank cases correctly more often than any other model. But its QWK (0.8987 argmax) is the worst of the completed experiments. AUC-ROC measures ranking quality; QWK measures classification quality. F2's probabilities are good but its decision boundaries are miscalibrated. This is why threshold optimisation recovers substantial QWK for F2 (+0.011) but less for D1 (+0.002).

---

## 10. Known Issues & Implementation Specifics

### 10.1 Checkpoint selection by QWK only (train.py)

```python
if val_metrics["qwk"] > best_qwk:
    torch.save(model.state_dict(), ckpt_path)
```

The best checkpoint is selected purely by val QWK. This incentivises the Moderate collapse pattern (see §9.5). A composite score `0.6*QWK + 0.4*MacroF1` is recommended and discussed in `docs/qwk_moderate_collapse_analysis.md` but not yet implemented in `train.py`.

### 10.2 FP16 overflow bug in OrdSupConLoss (fixed in commit 36cdabe)

Original code used `masked_fill_(-float('inf'))` for the diagonal masking step. With `torch.amp.autocast()` (FP16), this created NaN in the `logsumexp` computation via `exp(-inf) → 0` but `logsumexp` of all `-inf` entries → `-inf` → gradient NaN. Fixed to `masked_fill_(-1e4)` which is FP16-safe (max representable half-precision value ≈ 65504 >> 1e4).

### 10.3 OrdSupCon during the frozen head phase

When `use_joint_contrastive=True` and `freeze_epochs > 0`, the backbone is frozen during Stage 1. The projector still receives gradients via the avgpool hook, but these operate on frozen features. When the backbone unfreezes in Stage 2, the projector has learned to project from a fixed distribution that no longer holds — this likely contributes to instability in the early fine-tuning epochs.

### 10.4 Dual-view augmentation inflates apparent diversity

In joint contrastive training, each sample is augmented twice. The model sees `2N` views per batch, but only `N` unique images. This inflates the effective sample count and can cause the val→test gap to widen (F2: val 0.952, test 0.899, gap = 0.053; D1: val 0.957, test 0.916, gap = 0.041).

### 10.5 EfficientNet/ConvNeXt don't support CORN/CumLink

`_build_efficientnet_b4` and `_build_convnext_small` raise `NotImplementedError` if `loss_type` is `corn` or `cumlink`. These losses require adjusting the output head to K-1 units — only implemented for ResNet-50.

### 10.6 2 missing APTOS training images

Two images referenced in `train_label.csv` do not exist on disk. `DRDataset.__getitem__` handles this gracefully by trying multiple directory patterns.

### 10.7 34 blank EyePACS images

`build_eyepacs_dataset()` filters out images that, after Ben Graham preprocessing, are detected as blank (all pixels equal to 128 — the neutral gray bias of the preprocessing). These 34 images contained no retinal content.

### 10.8 Offline vs online oversampling

Offline oversampling (Level 1.5 augmentations saved to `data/train_oversampled/`) is preferred over online weighted sampling for two reasons: (1) the oversampled images can be visually inspected for quality; (2) it avoids the bias of sampling the same few minority-class originals many times per epoch, which can cause overfitting to those specific images.

---

## 11. Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/offline_oversample.py` | Generate synthetic minority-class images to `data/train_oversampled/`. Target: 1,000/class. Uses Level 1.5 augmentation. |
| `scripts/threshold_optimize.py` | Run Nelder-Mead threshold optimisation on any experiment's val predictions. |
| `scripts/threshold_optimize_exp501.py` | Same but hardcoded for exp 501 (F2), with the three threshold strategies. |
| `scripts/mc_dropout_eval.py` | Monte Carlo Dropout uncertainty: T=20 forward passes with `model.train()` (dropout active), compute variance of predictions as uncertainty estimate. |
| `scripts/preprocess_idrid.py` | Apply Ben Graham preprocessing to IDRiD images and save to `data/idrid_processed/`. |
| `run_all.sh` | Sequential runner for exp 12, 13, 14. Logs to `run_12_14.log`. Archives results + uploads to Google Drive via `rclone`. |
| `run_phase_subA.sh` | Runs exp 102 + 103. First regenerates offline-oversampled images (Level 1.5), then trains both experiments. |

---

## 12. Documentation Files

| File | Purpose |
|------|---------|
| `docs/01-ideas.md` | ~55 KB. Initial experiment progression design (Exp 0–13). Problem framing, dataset overview, metric justification. |
| `docs/02-ideas-next.md` | ~56 KB. Extended experiment ideas beyond the baseline phase. |
| `docs/03-ordinal-supcon.md` | ~97 KB. **Core research document**. Detailed OrdSupCon methodology, literature review (SupCon, CLOC, SCOL), Phases A–H design. |
| `docs/04-coral.md` | ~35 KB. Ordinal loss function experiments (CORN, CumLink, EMD). Mathematical derivations, expected outcomes. |
| `docs/05-phase-h.md` | ~37 KB. Linear Probe + Fine-Tune strategy. Motivation, hyperparameter choices, comparison with standard fine-tuning. |
| `docs/thesis-draft.md` | ~39 KB. Thesis manuscript. Abstract, introduction, methodology, results sections. |
| `docs/analysis_exp501_results.md` | Deep post-hoc analysis of exp 501 (F2). Confusion matrix breakdown, training dynamics, root cause analysis, next-step recommendations. |
| `docs/qwk_moderate_collapse_analysis.md` | Analysis of QWK centrality bias. 5 proposed solutions (composite checkpoint, min-F1 guard, ordinal cost focal, CORN, Macro-F1 primary). |
| `docs/section12_results.md` | Full results tables for Phases D & F. D1 vs F2 comparison, master comparison table, key findings, next steps (updated 2026-04-16). |
| `docs/BÁO CÁO TIẾN ĐỘ 1.md` | Vietnamese-language progress report for thesis supervisor. |

---

## 13. Recommended Next Steps (as of 2026-04-16)

Ordered by priority:

| Priority | Experiment | Expected gain | Rationale |
|:--------:|-----------|--------------|-----------|
| 🔴 1 | **MC Dropout on D1** (T=20) | Uncertainty calibration metrics | Head dropout=0.3 is already in place; no retraining needed. Contributes to thesis safety section. |
| 🟡 2 | **C1 (exp301)** — IDRiD G3+4 supplement | Severe F1 > 0.50 target | Real Grade 3+4 images attack the data scarcity root cause directly (+93 Severe, +67% Severe train data increase). |
| 🟡 3 | **MC Dropout on A0 baseline** | Comparison baseline for uncertainty | Shows improvement D1's dropout provides over non-dropout baseline. |
| 🟢 4 | **F4 (exp503)** — warmup λ | Eliminate gradient competition | Ramps λ from 0 to 0.1 over first 12 epochs post-unfreeze; should prevent F3-style collapse while keeping ordinal benefit. |
| 🟢 5 | **Composite checkpoint criterion** | +Severe F1, minor QWK trade-off | 5-line change in train.py: `composite = 0.6*QWK + 0.4*MacroF1`. Can be applied retroactively to training logs. |
| 🔵 6 | **G1-G5** (ordinal losses) | CORN expected +2–5% QWK | Eliminates centrality bias structurally by replacing 5-class softmax with 4 binary boundary decisions. |
| 🔵 7 | **H0-H4** (LP-FT) | Validate OrdSupCon representations | Linear probe on pretrained backbone without fine-tuning isolates representation quality from fine-tuning quality. |

---

## Summary

DR-NRT is a well-structured, systematically documented medical imaging research pipeline for ordinal DR grading. Its core contribution is the investigation of **OrdSupCon** — an ordinal-aware supervised contrastive loss that shapes the representation space to reflect clinical DR grade proximity. The project has run ~10 completed experiments across the full designed scope of ~50, with the current state-of-the-art being:

- **Best QWK**: D1+threshold = **0.9175** (Dropout + CosineAnnealing + offline oversample + threshold opt)
- **Best Severe F1**: F2+threshold = **0.4839** (Joint OrdSupCon + threshold opt)
- **Best AUC-ROC**: F2/F2+thresh = **0.9520**
- **Best calibration (ECE)**: A0 = **0.0420**

The central unsolved challenge is the tension between QWK and per-class F1: no experiment simultaneously improves both beyond A0's Macro F1 ceiling (0.7040) while matching or exceeding D1's QWK (0.9175).
