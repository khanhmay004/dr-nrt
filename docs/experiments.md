# Experiments

Every experiment is a single `ExpConfig` entry in
[`src/config.py`](../src/config.py) and is launched by id:

```bash
python run_experiment.py --exp <ID> --device cuda --workers 4
```

Outputs go to `results/<exp_name>/` (predictions CSV, metrics JSON, confusion
matrix, training curves, log) and checkpoints to `checkpoints/<exp_name>/`.

---

## Naming scheme

Experiment ids are grouped into phases. Short codes (B0–B13, A0–A2, D1, F2, G1,
H0–H5, I1–I5) are the labels used in the figures and in
`scripts/_canonical_metrics.json`.

| Id range | Phase | Theme |
|----------|-------|-------|
| `0–14`   | **1 — Ablation** (B0–B13) | Progressive recipe build‑up on the baseline |
| `100–201`| **A — Sampling & OrdSupCon** (A0–A2) | Weighted sampler, offline oversample, in‑/cross‑domain OrdSupCon pre‑training |
| `300–301`| **D — Regularisation** | **D1**: dropout + cosine LR + WD — the champion recipe |
| `501–504`| **F — Joint contrastive** | Auxiliary OrdSupCon loss *during* supervised fine‑tuning |
| `600–605`| **G — Ordinal losses** | CORN, EMD², Cumulative‑Link on the A1 backbone |
| `700–705`| **H — LP‑FT** | Linear‑probe‑then‑finetune on OrdSupCon backbones (H1 = EyePACS, H5 = APTOS) |
| `802–900`| **I — Rescue / Champion** | SORD, SWAD, L2‑SP, prototype head, FLYP; final combined recipe |

### Decoding the sub‑codes
- `a0c_offline_oversample` → sampling strategy variant within Phase A.
- `a1_ordsupcon_aptos` → OrdSupCon pre‑trained **in‑domain** (APTOS).
- `a2_ordsupcon_eyepacs` → OrdSupCon pre‑trained **cross‑domain** (EyePACS, 35k).
- `freeze5` → 5 head‑only warm‑up epochs before unfreezing.
- `d1recipe` → the D1 regularisation recipe applied on top of a given backbone.

---

## The three headline models

| Code | Exp | Backbone init | Why it matters |
|------|-----|---------------|----------------|
| **D1** | `exp300` | ImageNet | Single‑model champion (QWK **0.9175**) |
| **H1** | `exp701` | OrdSupCon @ EyePACS (cross‑domain) + D1 recipe | Ensemble diversifier: lifts **Mild** |
| **H5** | `exp705` | OrdSupCon @ APTOS (in‑domain) + D1 recipe | Best single‑model **Severe** F1; lifts Severe/Proliferative in the ensemble |

> **Note (see also the project memory):** H1 and H5 use *different* OrdSupCon
> initialisations (A2/EyePACS vs A1/APTOS) — they are **not** two runs of one
> backbone.

The final headline result is the **{D1, H1, H5} ensemble** (logit averaging):
two OrdSupCon members pre‑trained on *different corpora* contribute
non‑overlapping minority‑class signal that ImageNet‑init D1 lacks.

---

## Evaluation utilities (`scripts/`)

| Script | Purpose |
|--------|---------|
| `eval_checkpoint.py` | Evaluate a checkpoint on val+test with a chosen decoder |
| `eval_messidor2.py` | External Messidor‑2 generalisation |
| `mc_dropout_eval.py` / `mc_dropout_ensemble.py` | MC‑Dropout uncertainty (T=20) |
| `ncm_eval.py` | Nearest‑class‑mean readout on a frozen backbone |
| `threshold_optimize*.py` | Post‑hoc cumulative‑threshold optimisation |
| `compute_canonical.py` | Recompute all metrics + bootstrap CIs → `_canonical_metrics.json` |
| `offline_oversample.py` | Generate the oversampled minority set |
| `build_*figures.py`, `build_error_pair_distribution.py` | Regenerate paper/thesis figures |

Reproduce the headline ensemble end‑to‑end from
[`notebooks/aptos-ensemble-d1-h1-h5.ipynb`](../notebooks/aptos-ensemble-d1-h1-h5.ipynb).
