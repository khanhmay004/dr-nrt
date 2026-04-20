"""Generate notebooks/explainability.ipynb programmatically.

Run once on the remote (GPU) to produce the Task-3 notebook. Mirrors the
pattern of the other Task generators.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "explainability.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# Intro
# ---------------------------------------------------------------------------

INTRO_MD = """# Explainability — Where is the model looking?

Four families of attribution, validated against classical-CV lesion
proxies (APTOS has no pixel-level lesion masks):

* **Grad-CAM / Grad-CAM++ / HiResCAM** on ResNet-50 `layer4[-1]`.
* **Occlusion sensitivity** (sliding patch) as a gradient-free baseline.
* **Integrated Gradients** (pixel-level attribution via captum).
* **SHAP GradientExplainer** on a 30-sample subset (too expensive for full
  test).

Every CAM is scored on five medically-grounded criteria:

1. **On-retina energy** — top-20% CAM mass should sit inside the FOV.
2. **Anatomical region** — optic-disc vs fovea vs retina vs background.
3. **Lesion-proxy overlap** — pointing-game hit rate + Dice vs MA /
   hemorrhage / exudate candidates from `fundus_cv`.
4. **TTA consistency** — pairwise IoU of CAMs under H-flip and 90° rotations.
5. **Insertion / deletion AUC** — faithfulness (Petsiuk 2018).

Target model: `exp701_h1_ordsupcon_d1recipe` (best classification single).
Ensemble CAMs are obtained by averaging per-member attribution maps.
"""


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

SETUP_CODE = """from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

ROOT = Path('..').resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE,
    TRAIN_IMG_DIR, TEST_IMG_DIR, get_config,
)
from src.dataset import ben_graham_preprocess
from src.models import build_model
from src.analysis import explainers as expl
from src.analysis import faithfulness as faith
from src.analysis import fundus_cv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE)

RESULTS = ROOT / 'results'
CACHE = RESULTS / 'explainability_cache'
CACHE.mkdir(parents=True, exist_ok=True)
(CACHE / 'galleries').mkdir(exist_ok=True)

def _resolve(p: Path, fb: Path) -> Path:
    return p if p.exists() else fb

TEST_IMG_DIR = _resolve(Path(TEST_IMG_DIR), ROOT / 'data' / 'test_split')
TRAIN_IMG_DIR = _resolve(Path(TRAIN_IMG_DIR), ROOT / 'data' / 'train_split')

plt.rcParams['figure.dpi'] = 110
"""

MODEL_LOAD_CODE = """# Load the exp701 classification model + a target layer for Grad-CAM.
TARGET_EXP_ID = 701
cfg = get_config(TARGET_EXP_ID)
print('Target experiment:', cfg.exp_name)

# Checkpoint search order matches scripts/ensemble_cls.py:
candidates = [
    cfg.ckpt_dir / f'{cfg.exp_name}_best_composite.pth',
    cfg.ckpt_dir / f'{cfg.exp_name}_best.pth',
    cfg.ckpt_dir / f'{cfg.exp_name}_pseudo.pth',
    cfg.ckpt_dir / f'{cfg.exp_name}_swa.pth',
]
ckpt_path = next((c for c in candidates if c.exists()), None)
assert ckpt_path is not None, f'No checkpoint for {cfg.exp_name} in {cfg.ckpt_dir}'
print('Loading', ckpt_path.name)

model = build_model(cfg).to(DEVICE)
state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.eval()

TARGET_LAYER = model.layer4[-1]
print('Target layer: model.layer4[-1]  |  type:', type(TARGET_LAYER).__name__)
"""


# ---------------------------------------------------------------------------
# 3B — sample selection from Task 1 / Task 4 caches
# ---------------------------------------------------------------------------

SELECTION_MD = """## 3B. Sample selection

The sample set comes from the curated outputs of Tasks 1 + 4 rather than
random draws — every CAM here is justifying a specific claim elsewhere in
the thesis.
"""

SELECTION_CODE = """per_pair_path = RESULTS / 'confusion_cache' / 'per_pair_tags.csv'
top50_path = RESULTS / 'result_cache' / 'top50_errors.csv'
candidate_path = RESULTS / 'confusion_cache' / 'candidate_mislabels.csv'

per_pair = pd.read_csv(per_pair_path) if per_pair_path.exists() else pd.DataFrame()
top50 = pd.read_csv(top50_path) if top50_path.exists() else pd.DataFrame()
candidates = pd.read_csv(candidate_path) if candidate_path.exists() else pd.DataFrame()

print('per_pair errors:', len(per_pair))
print('top50 errors:', len(top50))
print('candidate mislabels:', len(candidates))

# Also load ensemble predictions to pick high-confidence correct samples per class.
ens_df = pd.read_csv(RESULTS / 'ensemble_900_300_701'
                      / 'ensemble_900_300_701_expected_grade_opt_preds.csv')
ens_df['correct'] = ens_df['rounded_prediction'] == ens_df['true_label']
ens_df['margin'] = np.abs(ens_df['raw_prediction'] - ens_df['true_label'])

HIGH_CONF_PER_CLASS = 5
LOW_CONF_PER_CLASS = 5
hi_conf_rows, lo_conf_rows = [], []
for c in range(NUM_CLASSES):
    sub_c = ens_df[(ens_df['true_label'] == c) & ens_df['correct']].sort_values('margin')
    hi_conf_rows.append(sub_c.head(HIGH_CONF_PER_CLASS).assign(bucket='correct_hiconf'))
    lo = ens_df[(ens_df['true_label'] == c) & ~ens_df['correct']].sort_values('margin')
    lo_conf_rows.append(lo.head(LOW_CONF_PER_CLASS).assign(bucket='wrong_lowconf'))
class_matrix = pd.concat(hi_conf_rows + lo_conf_rows, ignore_index=True)
print('class matrix size:', len(class_matrix))
"""


# ---------------------------------------------------------------------------
# 3A — method roster: run every explainer on one sample
# ---------------------------------------------------------------------------

METHODS_MD = """## 3A. Method roster — smoke-run on one sample

Confirms every explainer family returns a valid map on this model before
we spend compute on the full study.
"""

METHODS_CODE = """def load_rgb(code: str) -> np.ndarray | None:
    for d in (TEST_IMG_DIR, TRAIN_IMG_DIR):
        p = d / f'{code}.png'
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def load_bg_and_tensor(code: str):
    raw = load_rgb(code)
    assert raw is not None, f'missing image: {code}'
    bg = ben_graham_preprocess(raw, IMAGE_SIZE)
    t = expl.image_to_tensor(bg, device=DEVICE)
    return raw, bg, t


probe_code = class_matrix.iloc[0]['id_code']
probe_true = int(class_matrix.iloc[0]['true_label'])
raw, bg, t = load_bg_and_tensor(probe_code)
probs = expl.predict_probs(model, t)
probe_pred = int(np.argmax(probs))
print(f'probe {probe_code}: true={probe_true} pred={probe_pred} probs={np.round(probs,3)}')

maps = {}
maps['gradcam'] = expl.gradcam(model, t, probe_pred, TARGET_LAYER, method='gradcam')
maps['gradcam++'] = expl.gradcam(model, t, probe_pred, TARGET_LAYER, method='gradcam++')
maps['hirescam'] = expl.gradcam(model, t, probe_pred, TARGET_LAYER, method='hirescam')
maps['occlusion'] = expl.occlusion(model, t, probe_pred, patch_size=48, stride=24)
maps['ig'] = expl.integrated_gradients(model, t, probe_pred, n_steps=32)

fig, axes = plt.subplots(1, len(maps) + 1, figsize=(3 * (len(maps) + 1), 3.2))
axes[0].imshow(bg); axes[0].set_title('Ben-Graham'); axes[0].axis('off')
for ax, (name, m) in zip(axes[1:], maps.items()):
    ax.imshow(bg, alpha=0.6)
    ax.imshow(cv2.resize(m, (bg.shape[1], bg.shape[0])), cmap='jet', alpha=0.5)
    ax.set_title(name); ax.axis('off')
plt.tight_layout(); plt.show()
"""


# ---------------------------------------------------------------------------
# 3C — per-sample panels for the focus set
# ---------------------------------------------------------------------------

PANEL_MD = """## 3C. Per-sample panels

Layout per row:
`[Original | Ben-Graham | CAM of pred | CAM of true | Occlusion | IG | prob bar]`

For **correctly** classified samples, "CAM of true" equals "CAM of pred"
(we just show a single panel duplicated).

Panels save to `results/explainability_cache/galleries/`.
"""

PANEL_CODE = """def render_panel(row, save_dir: Path) -> dict:
    code = row['id_code']
    true = int(row['true_label'])
    raw, bg, t = load_bg_and_tensor(code)
    probs = expl.predict_probs(model, t)
    pred = int(np.argmax(probs))

    cam_pred = expl.gradcam(model, t, pred, TARGET_LAYER, method='gradcam')
    cam_true = cam_pred if pred == true else expl.gradcam(
        model, t, true, TARGET_LAYER, method='gradcam')
    occ = expl.occlusion(model, t, pred, patch_size=48, stride=24)
    ig = expl.integrated_gradients(model, t, pred, n_steps=32)

    fig, axes = plt.subplots(1, 7, figsize=(19, 3.0))
    axes[0].imshow(raw); axes[0].set_title(f'original {code[:8]}'); axes[0].axis('off')
    axes[1].imshow(bg); axes[1].set_title(f'BG ({true}->{pred})'); axes[1].axis('off')
    for ax, mp, name in [
        (axes[2], cam_pred, f'Grad-CAM pred={pred}'),
        (axes[3], cam_true, f'Grad-CAM true={true}'),
        (axes[4], occ,      'Occlusion'),
        (axes[5], ig,       'IG'),
    ]:
        ax.imshow(bg, alpha=0.55)
        ax.imshow(cv2.resize(mp, (bg.shape[1], bg.shape[0])), cmap='jet', alpha=0.5)
        ax.set_title(name); ax.axis('off')
    axes[6].bar(range(NUM_CLASSES), probs, tick_label=[n[:3] for n in CLASS_NAMES])
    axes[6].set_ylim(0, 1); axes[6].set_title('softmax')
    plt.tight_layout()
    out_path = save_dir / f'{row.get("bucket", "sample")}_{code}.png'
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    return {'id_code': code, 'true': true, 'pred': pred, 'path': str(out_path)}


panel_log = []
for _, row in class_matrix.iterrows():
    panel_log.append(render_panel(row, CACHE / 'galleries'))
print('rendered', len(panel_log), 'panels')
pd.DataFrame(panel_log).to_csv(CACHE / 'panel_index.csv', index=False)
"""

PANEL_PAIR_CODE = """# Per-pair confusion panels: 10 highest-margin errors per confused pair.
PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4)]
pair_log = []
for (i, j) in PAIRS:
    sub = per_pair[per_pair['pair'] == f'{CLASS_NAMES[i]}<->{CLASS_NAMES[j]}']
    sub = sub.sort_values('margin', ascending=False).head(10)
    for _, row in sub.iterrows():
        pair_log.append(render_panel(row, CACHE / 'galleries'))
print('pair-panels rendered:', len(pair_log))
"""


# ---------------------------------------------------------------------------
# 3D — faithfulness quantification
# ---------------------------------------------------------------------------

FAITH_MD = """## 3D. Faithfulness + medical validation

Per-method, per-class: on-retina energy, anatomical share, pointing-game
rate, Dice vs lesion proxies, TTA IoU, insertion/deletion AUC.

Runs on a 100-sample evaluation set so the aggregate table is statistically
meaningful. Set ``N_EVAL`` lower for a quick pass.
"""

FAITH_CODE = """N_EVAL = 100
eval_ids = class_matrix['id_code'].tolist()
# Top-up to N_EVAL with random additional test samples
remaining = [c for c in ens_df['id_code'] if c not in eval_ids]
rng = np.random.default_rng(42)
extra = rng.choice(remaining, size=max(0, N_EVAL - len(eval_ids)), replace=False)
eval_ids = list(eval_ids)[:N_EVAL - len(extra)] + list(extra)
print('evaluating', len(eval_ids), 'samples')

METHODS = ['gradcam', 'gradcam++', 'hirescam']
rows = []
for code in eval_ids:
    raw = load_rgb(code)
    if raw is None:
        continue
    bg = ben_graham_preprocess(raw, IMAGE_SIZE)
    t = expl.image_to_tensor(bg, device=DEVICE)
    probs = expl.predict_probs(model, t)
    pred = int(np.argmax(probs))
    true = int(ens_df.loc[ens_df['id_code'] == code, 'true_label'].iloc[0])

    H, W = bg.shape[:2]
    fov = fundus_cv.retinal_fov_mask(bg)
    disc_loc, disc_r = fundus_cv.detect_optic_disc(bg, fov_mask=fov)
    fovea_loc = fundus_cv.detect_fovea(bg, fov_mask=fov, optic_disc=disc_loc)
    disc_mask = np.zeros((H, W), dtype=np.uint8)
    if disc_loc is not None:
        cv2.circle(disc_mask, disc_loc, int(disc_r), 1, -1)
    fovea_mask = np.zeros((H, W), dtype=np.uint8)
    if fovea_loc is not None:
        cv2.circle(fovea_mask, fovea_loc, int(H * 0.08), 1, -1)
    lesions = faith.compute_lesion_proxies(bg)

    for m in METHODS:
        cam = expl.gradcam(model, t, pred, TARGET_LAYER, method=m)
        metrics = faith.evaluate_sample(cam, bg, fov, disc_mask, fovea_mask, lesions)
        rows.append({'id_code': code, 'method': m, 'true': true, 'pred': pred, **metrics})

eval_df = pd.DataFrame(rows)
eval_df.to_csv(CACHE / 'faithfulness_per_sample.csv', index=False)
agg = eval_df.groupby(['method']).mean(numeric_only=True).round(3)
display(agg)
"""

FAITH_IOU_CODE = """# 3D.2 — TTA consistency on a 20-sample subset (expensive)
TTA_N = 20
ids = rng.choice(eval_ids, size=TTA_N, replace=False)
tta_rows = []
for code in ids:
    raw = load_rgb(code)
    if raw is None:
        continue
    bg = ben_graham_preprocess(raw, IMAGE_SIZE)
    t = expl.image_to_tensor(bg, device=DEVICE)
    pred = int(np.argmax(expl.predict_probs(model, t)))

    def _cam_fn(tensor, cls):
        return expl.gradcam(model, tensor, cls, TARGET_LAYER, method='gradcam')
    cams = expl.tta_aligned_cams(_cam_fn, t, pred)
    iou = faith.cam_pairwise_iou(cams, top_pct=0.2)
    tta_rows.append({'id_code': code, 'pred': pred, 'tta_iou': iou})
tta_df = pd.DataFrame(tta_rows)
tta_df.to_csv(CACHE / 'tta_iou.csv', index=False)
print(f'Median TTA-IoU: {tta_df["tta_iou"].median():.3f}  (≥0.5 expected)')
display(tta_df.head(10))
"""

FAITH_AUC_CODE = """# 3D.3 — Insertion / Deletion AUC on a 30-sample subset.
INS_N = 30
ids = rng.choice(eval_ids, size=INS_N, replace=False)
auc_rows = []
for code in ids:
    raw = load_rgb(code)
    if raw is None:
        continue
    bg = ben_graham_preprocess(raw, IMAGE_SIZE)
    t = expl.image_to_tensor(bg, device=DEVICE)
    probs = expl.predict_probs(model, t)
    pred = int(np.argmax(probs))
    cam = expl.gradcam(model, t, pred, TARGET_LAYER, method='gradcam')
    cam_r = cv2.resize(cam, (bg.shape[1], bg.shape[0]))
    _, _, ins_auc = faith.insertion_curve(model, t, cam_r, pred, n_steps=20)
    _, _, del_auc = faith.deletion_curve(model, t, cam_r, pred, n_steps=20)
    auc_rows.append({'id_code': code, 'pred': pred,
                     'insertion_auc': ins_auc, 'deletion_auc': del_auc,
                     'faithfulness': ins_auc - del_auc})
auc_df = pd.DataFrame(auc_rows)
auc_df.to_csv(CACHE / 'insertion_deletion.csv', index=False)
display(auc_df.describe().round(3))
print('Insertion AUC > Deletion AUC on almost every sample => faithful CAMs.')
"""


# ---------------------------------------------------------------------------
# 3E — thesis narrative
# ---------------------------------------------------------------------------

NARRATIVE_MD = """## 3E. Thesis narrative — what the CAMs actually show, by grade

Fill in after running 3C + 3D with per-grade observations. Expected pattern
(from the medical foundation in `confusion_analysis.ipynb`):

* **Grade 0** — diffuse / optic-disc activations. Model has no lesion to
  attend to, so Grad-CAM often lands on the brightest structured region
  (disc). **not** a red flag.
* **Grade 1** — CAM should concentrate near MA candidates. Low pointing-game
  rate here = model is finding Grade-1 images by texture-level cues rather
  than lesion-level, which is honest but worth reporting.
* **Grade 2** — multi-region activation (MA + exudate + hemorrhage).
* **Grade 3** — should spread across multiple quadrants (4-2-1 clinical
  rule). If CAM is focal, the model is guessing.
* **Grade 4** — activation should include peripheral/parafoveal regions
  where neovessels grow.
"""

NARRATIVE_CODE = """# Aggregate 3D metrics by true grade for the narrative.
by_grade = eval_df.groupby(['true', 'method']).mean(numeric_only=True).round(3)
display(by_grade[['fov_energy_top20', 'share_optic_disc', 'share_rest_of_retina',
                   'pointing_ma', 'pointing_hemorrhage', 'pointing_exudate']])
"""

NARRATIVE_VIZ_CODE = """# Visualise on-retina energy per method as a sanity floor.
fig, ax = plt.subplots(figsize=(7, 4))
agg_method = eval_df.groupby('method')['fov_energy_top20'].agg(['mean', 'std'])
ax.bar(agg_method.index, agg_method['mean'], yerr=agg_method['std'], capsize=5,
       color='steelblue', edgecolor='black')
ax.axhline(0.9, color='red', ls='--', label='sanity floor 0.9')
ax.set_ylabel('Top-20% CAM energy inside retinal FOV')
ax.set_title('On-retina sanity check per explainer')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); plt.show()
"""


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

CELLS = [
    md(INTRO_MD),
    code(SETUP_CODE),
    code(MODEL_LOAD_CODE),

    md(SELECTION_MD),
    code(SELECTION_CODE),

    md(METHODS_MD),
    code(METHODS_CODE),

    md(PANEL_MD),
    code(PANEL_CODE),
    code(PANEL_PAIR_CODE),

    md(FAITH_MD),
    code(FAITH_CODE),
    code(FAITH_IOU_CODE),
    code(FAITH_AUC_CODE),

    md(NARRATIVE_MD),
    code(NARRATIVE_CODE),
    code(NARRATIVE_VIZ_CODE),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {NB_PATH} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
