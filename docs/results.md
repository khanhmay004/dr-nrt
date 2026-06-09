# Results

All numbers are computed on the **held‑out APTOS test set (n = 550)** by
`scripts/compute_canonical.py` (stratified bootstrap, 5,000 resamples, `seed=42`)
and stored in [`scripts/_canonical_metrics.json`](../scripts/_canonical_metrics.json),
the single source of truth. QWK = Quadratic Weighted Kappa (primary metric).

---

## 1. Phase 1 — building the recipe (single model, argmax)

| Code | Experiment | QWK | Macro‑F1 | Acc | F1 Severe | F1 Prolif |
|------|-----------|-----|----------|-----|-----------|-----------|
| B0 | baseline (CE, no aug) | 0.890 | 0.656 | 0.829 | 0.318 | 0.623 |
| B2 | + advanced aug | 0.908 | 0.703 | 0.856 | 0.400 | 0.707 |
| B3 | + focal + class weights | 0.913 | 0.707 | 0.845 | 0.421 | 0.692 |
| B8 | + GeM pooling (**A0 anchor**) | 0.913 | 0.704 | 0.842 | 0.444 | 0.701 |

Each block adds one ingredient; augmentation and focal loss give the largest
jumps, GeM lifts minority F1 at equal QWK.

---

## 2. Phase A — sampling & OrdSupCon (single model, argmax)

| Code | Experiment | QWK | Macro‑F1 | F1 Severe |
|------|-----------|-----|----------|-----------|
| A0c | offline oversample | 0.907 | 0.699 | 0.453 |
| A1  | OrdSupCon @ **APTOS** (in‑domain) | 0.906 | 0.687 | 0.429 |
| A2  | OrdSupCon @ **EyePACS** (cross‑domain) | 0.893 | 0.658 | 0.417 |

> **Robust null result.** Across two corpora and seven fine‑tuning protocols
> (Phases A, F, G, H), **no OrdSupCon backbone beats ImageNet‑init D1 on QWK
> when used alone.** The representation is not the bottleneck on APTOS‑2019.

---

## 3. Headline models (single, n = 550)

| Code | Model | Decoder | QWK | Macro‑F1 | Acc | F1 Sev | F1 Prolif |
|------|-------|---------|-----|----------|-----|--------|-----------|
| **D1** | exp300, ImageNet | argmax | 0.9159 | 0.6945 | 0.853 | 0.383 | 0.667 |
| **D1** | exp300, ImageNet | cum‑opt | **0.9175** | 0.689 | 0.845 | 0.423 | 0.649 |
| H1 | exp701, OrdSupCon‑EyePACS | argmax | 0.9028 | 0.686 | 0.818 | 0.431 | 0.683 |
| H5 | exp705, OrdSupCon‑APTOS | argmax | 0.8809 | 0.696 | 0.824 | **0.545** | 0.584 |

D1 is the **single‑model champion** (QWK 0.9175). H5 has the best single‑model
**Severe** F1 (0.545).

---

## 4. Headline ensemble — {D1, H1, H5} (logit averaging, argmax)

| Ensemble | QWK | Macro‑F1 | Acc | F1 NoDR | F1 Mild | F1 Mod | F1 Sev | F1 Prolif |
|----------|-----|----------|-----|---------|---------|--------|--------|-----------|
| {D1, H1} | 0.9096 | 0.7052 | 0.851 | 0.985 | 0.655 | 0.814 | 0.431 | 0.640 |
| **{D1, H1, H5}** | **0.9149** | **0.7324** | **0.858** | 0.983 | 0.672 | 0.814 | **0.500** | **0.692** |

Adding the two OrdSupCon members lifts **Macro‑F1 by +0.038** over D1 and sets
the study's best minority F1s (Severe 0.50, Proliferative 0.69) — even though
neither member beats D1 alone. The diversity, from **different pre‑training
corpora**, is what pays off at ensemble time.

---

## 5. Clinical view — referability (binary, grade ≥ 2)

| Model | Sensitivity | Specificity | PPV | NPV |
|-------|-------------|-------------|-----|-----|
| D1 (exp300) | 0.942 | 0.936 | 0.909 | 0.959 |
| **{D1, H1, H5}** | 0.919 | **0.957** | 0.936 | 0.946 |

In the ensemble's 5‑grade confusion matrix, **no Severe or Proliferative case is
ever predicted as No DR** — the most dangerous failure mode does not occur.

---

## 6. Uncertainty & calibration (MC‑Dropout, T = 20)

D1 has the best raw calibration (lowest ECE); H5 gives the strongest
selective‑prediction signal (entropy → error AUROC), making both deployable for
"refer to specialist when uncertain" triage. See
[`notebooks/result_analysis.ipynb`](../notebooks/result_analysis.ipynb).

---

## 7. Negative results catalogue

Documented techniques that **did not** beat the D1 recipe (root causes in the
notebooks): label smoothing, Mixup/CutMix, smooth‑L1 regression, cosine
warm‑restarts, CORN / EMD² / Cumulative‑Link ordinal losses, L2‑SP anchoring,
SWAD, ordinal‑prototype head, joint contrastive fine‑tuning, and standalone
OrdSupCon fine‑tuning (7 protocols × 2 corpora).

**Bottom line:** on APTOS‑2019 the binding constraint is **regularisation and
data scale**, not loss design or contrastive initialisation — but OrdSupCon
backbones remain valuable as *ensemble diversifiers* for minority grades.
