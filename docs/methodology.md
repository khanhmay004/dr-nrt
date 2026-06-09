# Methodology

This document describes the data pipeline, model, losses, and training protocol
used in this project. All hyper‑parameters live in [`src/config.py`](../src/config.py)
as a registry of `ExpConfig` dataclasses, so every experiment is fully specified
and reproducible from a single experiment id.

---

## 1. Task

Five‑class **ordinal** classification of diabetic retinopathy (DR) severity from
colour fundus photographs:

| Grade | 0 | 1 | 2 | 3 | 4 |
|-------|---|---|---|---|---|
| Label | No DR | Mild | Moderate | Severe | Proliferative |

Because the grades are ordered, the primary metric is **Quadratic Weighted Kappa
(QWK)**, which penalises errors by the *square* of the grade distance (a
`No DR → Proliferative` mistake counts far more than `No DR → Mild`). Macro‑F1 and
per‑class F1 are tracked to expose minority‑grade behaviour.

---

## 2. Datasets

| Dataset | Role | Size | Notes |
|---------|------|------|-------|
| **APTOS 2019** | train + test (primary benchmark) | 3,662 imgs → 3,112 train / 550 test | Stratified 85 / 15 split, `seed=42` |
| **EyePACS** | OrdSupCon Stage‑1 pre‑training only | ~35k imgs | Never fine‑tuned on directly; strong domain shift, noisy labels |
| **IDRiD** | optional Severe/Proliferative supplement | 516 imgs | Phase C only |
| **Messidor‑2** | external validation | 1,744 imgs | Out‑of‑distribution generalisation check |

Datasets are **not** committed (see `.gitignore`); place them under `data/`,
`Eyepacs/`, etc. as described in the [README](../README.md).

### Class distribution (APTOS train)
No DR ≈ 49%, Moderate ≈ 27%, Mild ≈ 10%, Proliferative ≈ 8%, **Severe ≈ 5%**.
The Severe grade is the binding bottleneck throughout the study.

---

## 3. Preprocessing — Ben Graham method

1. Detect the circular fundus boundary (grayscale threshold + contours) and crop
   the black border.
2. Resize to a square working resolution (512×512).
3. Subtract a Gaussian‑blurred copy (`σ ≈ size/30`) to remove uneven illumination
   and amplify lesion contrast, then re‑centre to mid‑grey.
4. ImageNet channel normalisation (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`).

This step is *table stakes*: removing it costs ≥ 0.03 QWK on every baseline.

### Offline oversampling
Minority classes are deterministically augmented to **1,000 images/class** and
written to `data/train_oversampled/` (`oversample_target=1000`). This is combined
with class‑weighted loss for balanced gradients while keeping the augmented set
inspectable and reproducible.

---

## 4. Model

- **Backbone:** ResNet‑50, ImageNet‑pretrained (`backbone="resnet50"`).
- **Pooling:** Generalised‑Mean (**GeM**, learnable exponent `p`, init 3.0,
  `use_gem=True`) — amplifies small focal activations vs. background.
- **Classification head:** `Dropout(0.3) → Linear(2048 → 5)`. Dropout is kept at
  inference for **MC‑Dropout** uncertainty estimation.
- **Stage‑1 projection head (contrastive only):**
  `Linear(2048→512) → BN → ReLU → Linear(512→128) → BN → L2‑normalise`
  (`contrastive_proj_dim=128`), discarded before fine‑tuning.

Defined in [`src/models.py`](../src/models.py) (`build_model`, `GeM`,
`ProjectionHead`, `OrdinalPrototypeHead`).

---

## 5. Losses ([`src/losses.py`](../src/losses.py))

**Primary — Focal loss + inverse‑frequency class weights** (`loss_type="focal"`,
`focal_gamma=2.0`, `use_class_weights=True`):

```
L_focal = - Σ_c  w_c · (1 - p_c)^γ · log p_c ,   γ = 2,   w_c ∝ 1 / freq(c)
```

**Ordinal‑aware Supervised Contrastive — OrdSupCon** (the method under study).
Standard SupCon treats same‑label samples as positives and everything else as
negatives. OrdSupCon replaces that binary partition with a *continuous ordinal
weight* on every pair:

```
W(i, j) = 1 − |g_i − g_j| / (K − 1) ,   K = 5
```

| grade distance | 0 | 1 | 2 | 3 | 4 |
|----------------|---|---|---|---|---|
| weight W       | 1.00 | 0.75 | 0.50 | 0.25 | 0.00 |

```
L_OrdSupCon = − Σ_{j≠i}  ( W(i,j) / Σ_{k≠i} W(i,k) ) · log  exp(z_i·z_j / τ)
                                                          ─────────────────────
                                                          Σ_{k≠i} exp(z_i·z_k / τ)
```

with temperature `τ = 0.07`. The encoder is taught that adjacent grades cluster
nearby while distant grades repel — an ordinal geometry in embedding space.

**Ordinal‑consistent variants explored** (Phase G/I, all benchmarked against
Focal): CORN, Cumulative‑Link BCE, EMD², SORD, Logit‑Adjusted CE.

---

## 6. Training protocol ([`src/train.py`](../src/train.py))

**Stage 1 — OrdSupCon pre‑training (optional).**
Dual‑view contrastive training (`ContrastiveDRDataset`) on EyePACS (20 ep) or
APTOS (40 ep) with `L_OrdSupCon`; cosine LR from `1e‑3`. The backbone is saved
and loaded via `load_backbone` for Stage 2.

**Stage 2 — supervised fine‑tuning.**
1. **Freeze** backbone for `freeze_epochs=5`, train the head at `lr_head=1e‑3`.
2. **Unfreeze** and train the full network at `lr_finetune=1e‑4`.
3. Focal + class weights, **augmentation level 2** (flips, rotations, CLAHE,
   CoarseDropout, ElasticTransform, GridDistortion, GaussianBlur — see
   [`src/transforms.py`](../src/transforms.py)), offline oversampling.
4. Cosine LR decay, `weight_decay=1e‑4`, `total_epochs=80`, `batch_size=32`.
5. Checkpoint the best epoch by a composite of QWK + Macro‑F1.

**Inference ([`src/tta.py`](../src/tta.py), [`src/evaluate.py`](../src/evaluate.py)).**
- **TTA:** average soft‑max probabilities over flip/rotation views.
- **Decoding:** `argmax`, expected‑grade, or **optimised cumulative thresholds**
  tuned on validation (`OptimizedRounder`).
- **Ensembling** ([`src/ensemble.py`](../src/ensemble.py)): logit averaging across
  members.

---

## 7. The "D1" recipe (single‑model champion, `exp 300`)

| Component | Value |
|-----------|-------|
| Init | ImageNet |
| Loss | Focal (γ=2) + class weights |
| Pooling | GeM |
| Head | Dropout 0.3 → Linear(2048→5) |
| Augmentation | Level 2 (advanced) |
| Oversampling | 1,000 / class (offline) |
| Optimiser / WD | AdamW / `1e‑4` |
| Schedule | cosine decay, 80 epochs, freeze 5 |
| Inference | TTA + optimised cumulative thresholds |

The central empirical finding of the thesis is that **this regularisation recipe —
not loss design or contrastive backbone initialisation — is the binding constraint
on APTOS‑2019.** See [results.md](results.md) and [experiments.md](experiments.md).

---

## 8. Reproducibility

- Fixed `seed=42`; deterministic stratified 85/15 split.
- Every run is one `ExpConfig` in [`src/config.py`](../src/config.py).
- Metrics are recomputed with stratified bootstrap (5,000 resamples, `seed=42`,
  n=550) into `scripts/_canonical_metrics.json`, the single source of truth for
  all reported numbers and confidence intervals.
