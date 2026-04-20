"""Generate notebooks/result_analysis.ipynb programmatically.

Run once, then edit interactively. Mirrors the pattern used by
``_build_eda_deep_notebook.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "result_analysis.ipynb"


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
# SECTION 0 — Setup / data loading
# ---------------------------------------------------------------------------

SETUP_MD = """# Deep Result Analysis — DR-NRT

Extends `result_dis.ipynb` with the four mentor-flagged gaps:

- **A. Statistical rigor** — bootstrap CIs on every reported metric, paired
  bootstraps + McNemar between {ensemble, exp701, exp300, exp00 baseline},
  ROC/PR with CIs, threshold sensitivity, calibration battery.
- **B. Clinical metrics** — referable-DR (>=2) and sight-threatening-DR (>=3)
  sensitivity/specificity/PPV/NPV, cost-weighted risk, coverage-accuracy
  abstention curve, literature anchoring (Gulshan 2016, APTOS top-5).
- **C. Stratified errors** — error rate by image-quality bin, illumination
  regime, resolution bucket; pulls bins from `results/eda_cache/`.
- **D. Ablation ladder** — marginal-contribution table baseline → +StdAug →
  +AdvAug → +Focal → +D1 dropout → +OrdSupCon → +Ensemble with paired-
  bootstrap p-values.
- **E. Failure taxonomy** — manual tagging set + inspection panel for 50
  ensemble errors, mapped to medical interpretation buckets.
- **F. Ensemble decomposition** — agreement-vs-accuracy, sample-level gain
  attribution, weight-sensitivity grid.

All heavy outputs cache to `../results/result_cache/`.
"""

SETUP_CODE = """from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

ROOT = Path('..').resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES, NUM_CLASSES
from src.evaluate import (
    OptimizedRounder,
    compute_metrics,
    compute_ece,
    quadratic_weighted_kappa,
    regression_to_class,
)
from src.analysis import confusion_stats as cs
from src.analysis import calibration as cal

RESULTS = ROOT / 'results'
CACHE = RESULTS / 'result_cache'
CACHE.mkdir(parents=True, exist_ok=True)

sns.set_context('notebook')
plt.rcParams['figure.dpi'] = 110
RNG = np.random.default_rng(42)
"""

LOADER_MD = """## 0. Load per-experiment predictions

Every experiment in `results/<exp>/` writes a `*_preds.csv` with schema
`id_code, raw_prediction, rounded_prediction, true_label`. Build a registry
keyed by short alias for the rest of the notebook.
"""

LOADER_CODE = """def _find_preds(exp_dir: Path) -> Path | None:
    matches = list(exp_dir.glob('*_preds.csv'))
    if not matches:
        return None
    # prefer the ensemble's expected_grade_opt variant when multiple
    expected = [m for m in matches if 'expected_grade_opt' in m.name]
    return expected[0] if expected else matches[0]


REGISTRY = {
    'ensemble':  RESULTS / 'ensemble_900_300_701',
    'exp701':    RESULTS / 'exp701_h1_ordsupcon_d1recipe',
    'exp300':    RESULTS / 'exp300_d1_dropout_cosine',
    'exp00':     RESULTS / 'exp00_baseline',
    'exp01_std': RESULTS / 'exp01_std_aug',
    'exp02_adv': RESULTS / 'exp02_adv_aug',
    'exp03_foc': RESULTS / 'exp03_focal_loss',
    'exp07_reg': RESULTS / 'exp07_regression',
    'exp12_thr': RESULTS / 'exp12_opt_thresh_opA',
    'exp103_a1': RESULTS / 'exp103_a1_ordsupcon_aptos',
    'exp605_a1v3': RESULTS / 'exp605_a1v3_ordsupcon_40ep',
    'exp700_lp': RESULTS / 'exp700_h0_linear_probe_a2',
    'exp702_lpft': RESULTS / 'exp702_h2_lpft_a2',
    'exp804_swad': RESULTS / 'exp804_i3_swad_on_d1',
    'exp805_l2sp': RESULTS / 'exp805_i4_l2sp_a2_d1recipe',
    'exp806_proto': RESULTS / 'exp806_i5_prototype_head_a2_emd',
}

PREDS = {}
for alias, d in REGISTRY.items():
    if not d.exists():
        continue
    p = _find_preds(d)
    if p is None:
        continue
    df = pd.read_csv(p)
    df = df.sort_values('id_code').reset_index(drop=True)
    PREDS[alias] = df

print('Loaded experiments:', list(PREDS.keys()))
print('Sample sizes:', {k: len(v) for k, v in PREDS.items()})
"""

ALIGN_CODE = """# Align all experiments to the same id_code order so paired tests are valid.
common_ids = set.intersection(*[set(v['id_code']) for v in PREDS.values()])
common_ids = sorted(common_ids)
print(f'Common ids across experiments: {len(common_ids)}')

ALIGNED = {}
for alias, df in PREDS.items():
    sub = df[df['id_code'].isin(common_ids)].sort_values('id_code').reset_index(drop=True)
    ALIGNED[alias] = sub

Y_TRUE = ALIGNED['ensemble']['true_label'].to_numpy()
PRED_INT = {a: ALIGNED[a]['rounded_prediction'].to_numpy().astype(int) for a in ALIGNED}
PRED_RAW = {a: ALIGNED[a]['raw_prediction'].to_numpy().astype(float) for a in ALIGNED}

# sanity: same ground truth across all aligned predictions
for a, df in ALIGNED.items():
    assert (df['true_label'].to_numpy() == Y_TRUE).all(), f'true_label mismatch for {a}'
print('Aligned arrays ready. N =', len(Y_TRUE))
"""


# ---------------------------------------------------------------------------
# SECTION A — Statistical rigor
# ---------------------------------------------------------------------------

A_MD = """## A. Statistical rigor

Bootstrap CIs and paired tests are the spine of the new results chapter.
Reuse `src.analysis.confusion_stats` so the methodology lives outside the
notebook.
"""

A1_CODE = """# A1. Bootstrap 95% CIs for QWK / Macro-F1 / Accuracy on each experiment
N_BOOT = 1000
metric_funcs = {
    'qwk': cs.metric_qwk,
    'macro_f1': cs.metric_macro_f1,
    'accuracy': cs.metric_accuracy,
}

rows = []
for alias, y_pred in PRED_INT.items():
    for metric_name, fn in metric_funcs.items():
        point, lo, hi = cs.bootstrap_ci(fn, Y_TRUE, y_pred, n_boot=N_BOOT, rng=RNG)
        rows.append({'experiment': alias, 'metric': metric_name,
                     'value': point, 'ci_lo': lo, 'ci_hi': hi})
ci_df = pd.DataFrame(rows)
ci_pivot = ci_df.pivot_table(index='experiment', columns='metric',
                             values=['value', 'ci_lo', 'ci_hi'])
ci_pivot.to_csv(CACHE / 'metric_cis.csv')
display(ci_pivot)
"""

A2_CODE = """# A2. Per-class F1 with bootstrap CIs (ensemble + exp701 only — keeps table compact)
focus = ['ensemble', 'exp701']
rows = []
for alias in focus:
    for c in range(NUM_CLASSES):
        fn = cs.metric_per_class_f1(c)
        point, lo, hi = cs.bootstrap_ci(fn, Y_TRUE, PRED_INT[alias], n_boot=N_BOOT, rng=RNG)
        rows.append({'experiment': alias, 'class': CLASS_NAMES[c],
                     'f1': point, 'ci_lo': lo, 'ci_hi': hi})
per_class_df = pd.DataFrame(rows)
per_class_df.to_csv(CACHE / 'per_class_f1_cis.csv', index=False)
display(per_class_df.pivot(index='class', columns='experiment',
                            values=['f1', 'ci_lo', 'ci_hi']))
"""

A3_CODE = """# A3. Paired bootstrap deltas: ensemble vs each baseline
PAIRS = [('ensemble', 'exp701'), ('ensemble', 'exp300'),
         ('ensemble', 'exp00'),  ('exp701', 'exp00')]
rows = []
for a, b in PAIRS:
    if a not in PRED_INT or b not in PRED_INT:
        continue
    for metric_name, fn in metric_funcs.items():
        r = cs.paired_bootstrap_diff(fn, Y_TRUE, PRED_INT[a], PRED_INT[b],
                                     n_boot=N_BOOT, rng=RNG)
        rows.append({'A': a, 'B': b, 'metric': metric_name, **r})
paired_df = pd.DataFrame(rows)
paired_df.to_csv(CACHE / 'paired_bootstrap.csv', index=False)
display(paired_df)
"""

A4_CODE = """# A4. McNemar between ensemble and each comparator (overall)
rows = []
for a, b in PAIRS:
    if a not in PRED_INT or b not in PRED_INT:
        continue
    r = cs.mcnemar_test(Y_TRUE, PRED_INT[a], PRED_INT[b])
    rows.append({'A': a, 'B': b, **r})
mcnemar_df = pd.DataFrame(rows)
display(mcnemar_df)
"""

A5_CODE = """# A5. Confusion-matrix decomposition with per-cell CIs (ensemble)
ens_decomp = cs.confusion_with_ci(Y_TRUE, PRED_INT['ensemble'], NUM_CLASSES,
                                  n_boot=N_BOOT, rng=RNG)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, key, title in zip(
    axes,
    ['cm_raw', 'cm_row', 'cm_col'],
    ['Raw counts', 'Row-normalised (recall)', 'Col-normalised (precision)'],
):
    data = ens_decomp[key]
    sns.heatmap(data, annot=True, fmt='.2f' if data.dtype.kind == 'f' else 'd',
                cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                cbar=False)
    ax.set_title(f'Ensemble — {title}')
    ax.set_xlabel('Pred'); ax.set_ylabel('True')
plt.tight_layout(); plt.show()
"""

A6_CODE = """# A6. Adjacent vs non-adjacent error rate
rows = []
for alias in ['ensemble', 'exp701', 'exp300', 'exp00']:
    if alias not in PRED_INT:
        continue
    info = cs.adjacent_error_rate(Y_TRUE, PRED_INT[alias], NUM_CLASSES)
    rows.append({
        'experiment': alias,
        'adjacent_frac': info['adjacent_frac'],
        'nonadjacent_frac': info['nonadjacent_frac'],
        **{f'adj_{CLASS_NAMES[i]}': float(info['per_class_adjacent'][i])
           for i in range(NUM_CLASSES)},
    })
adj_df = pd.DataFrame(rows)
adj_df.to_csv(CACHE / 'adjacent_errors.csv', index=False)
display(adj_df)
"""

A7_CODE = """# A7. Off-by-N distribution (ensemble) heatmap
off = cs.off_by_n_distribution(Y_TRUE, PRED_INT['ensemble'], NUM_CLASSES)
offsets = np.arange(-(NUM_CLASSES - 1), NUM_CLASSES)
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(off, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            xticklabels=offsets, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel('pred - true'); ax.set_ylabel('True class')
ax.set_title('Off-by-N distribution per true class (ensemble)')
plt.tight_layout(); plt.show()
"""

A8_CODE = """# A8. Ordinal vs nominal kappa decomposition
rows = []
for alias in ['ensemble', 'exp701', 'exp300', 'exp00']:
    if alias not in PRED_INT:
        continue
    rows.append({'experiment': alias, **cs.kappa_split(Y_TRUE, PRED_INT[alias])})
kappa_df = pd.DataFrame(rows)
kappa_df.to_csv(CACHE / 'kappa_split.csv', index=False)
display(kappa_df)
"""

A9_CODE = """# A9. Threshold sensitivity sweep on ensemble's regression score
# Use OptimizedRounder to first locate the optimum, then sweep ±0.4 around each.
opt = OptimizedRounder()
opt.fit(PRED_RAW['ensemble'], Y_TRUE)
opt_thresh = list(opt.thresholds)
print('Optimum thresholds (re-fit on TEST — for sensitivity sweep only):', opt_thresh)
print('NOTE: the headline 0.9105 number uses thresholds fitted on VAL; re-fitting'
      ' on test inflates QWK and is used here only to map the local sensitivity'
      ' surface around the optimum.')

deltas = np.linspace(-0.4, 0.4, 17)
sens = np.zeros((len(opt_thresh), len(deltas)))
for ti in range(len(opt_thresh)):
    for di, d in enumerate(deltas):
        t = list(opt_thresh)
        t[ti] = opt_thresh[ti] + d
        t = sorted(t)
        pred = regression_to_class(PRED_RAW['ensemble'], t)
        sens[ti, di] = quadratic_weighted_kappa(Y_TRUE, pred)

fig, ax = plt.subplots(figsize=(8, 5))
for ti in range(len(opt_thresh)):
    ax.plot(deltas, sens[ti], marker='o', label=f'T{ti+1} (opt={opt_thresh[ti]:.3f})')
ax.axvline(0, color='k', lw=0.7, ls='--')
ax.set_xlabel(r'$\\Delta$ from optimum threshold')
ax.set_ylabel('QWK')
ax.set_title('Threshold sensitivity — ensemble (one-at-a-time)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
"""

A10_CODE = """# A10. Calibration view for the regression ensemble
binned, conf = cal.regression_calibration_curve(
    PRED_RAW['ensemble'], PRED_INT['ensemble'], Y_TRUE,
    opt_thresh, n_bins=10,
)
nonempty = binned['count'] > 0
centers = 0.5 * (binned['edges'][:-1] + binned['edges'][1:])
fig, ax = plt.subplots(figsize=(5, 5))
ax.bar(centers[nonempty], binned['acc'][nonempty], width=0.1, alpha=0.7,
       edgecolor='black', label='Accuracy')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
ax.set_xlabel('Margin-confidence (regression head)')
ax.set_ylabel('Accuracy of rounded prediction')
ax.set_title('Ensemble — ordinal-margin reliability')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

print(f'Note: per-class softmax probabilities were not saved; standard ECE on '
      f'softmax is unavailable for this ensemble. Margin-confidence is a '
      f'regression-head proxy: distance from the nearest OptimizedRounder '
      f'threshold, normalised by the bin half-width.')
"""


# ---------------------------------------------------------------------------
# SECTION B — Clinical metrics
# ---------------------------------------------------------------------------

B_MD = """## B. Clinical-oriented metrics

The thesis ultimately argues for screening utility. These metrics speak the
clinician's language: referable-DR (>=Moderate) and sight-threatening-DR
(>=Severe) sensitivity / specificity / PPV / NPV, plus a cost-weighted risk
and a triage-style abstention curve.
"""

B1_CODE = """def binary_metrics(y_true_bin, y_pred_bin):
    y_true_bin = y_true_bin.astype(int); y_pred_bin = y_pred_bin.astype(int)
    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) else 0.0
    npv  = tn / (tn + fn) if (tn + fn) else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'sensitivity': sens, 'specificity': spec, 'ppv': ppv, 'npv': npv}


def boot_binary_ci(y_true_bin, y_pred_bin, key='sensitivity', n_boot=1000):
    rng = np.random.default_rng(42)
    n = len(y_true_bin)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = binary_metrics(y_true_bin[idx], y_pred_bin[idx])[key]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


rows = []
for label, thresh in [('referable_DR (>=2)', 2), ('sight_threatening_DR (>=3)', 3)]:
    yt = (Y_TRUE >= thresh).astype(int)
    for alias in ['ensemble', 'exp701', 'exp300']:
        if alias not in PRED_INT:
            continue
        yp = (PRED_INT[alias] >= thresh).astype(int)
        m = binary_metrics(yt, yp)
        slo, shi = boot_binary_ci(yt, yp, 'sensitivity')
        plo, phi = boot_binary_ci(yt, yp, 'specificity')
        rows.append({'task': label, 'experiment': alias, **m,
                     'sens_ci': f'[{slo:.3f}, {shi:.3f}]',
                     'spec_ci': f'[{plo:.3f}, {phi:.3f}]'})
clinical_df = pd.DataFrame(rows)
clinical_df.to_csv(CACHE / 'clinical_binary.csv', index=False)
display(clinical_df)
"""

B2_CODE = """# B2. Cost-weighted risk: under-grading is much worse than over-grading.
# Cost matrix: rows=true, cols=pred. C[i, j] = penalty for predicting j when truth is i.
COST = np.array([
    # pred:  0     1     2     3     4
    [   0.0,  1.0,  2.0,  3.0,  4.0],   # true 0 (over-grade penalty mild)
    [   2.0,  0.0,  1.0,  2.0,  3.0],   # true 1
    [   4.0,  2.0,  0.0,  1.0,  2.0],   # true 2 (under-grade hurts)
    [   8.0,  4.0,  2.0,  0.0,  1.0],   # true 3 (under-grade Severe = bad)
    [  16.0,  8.0,  4.0,  2.0,  0.0],   # true 4 (missed PDR = catastrophic)
])

def cost_risk(y_true, y_pred, C=COST):
    return float(C[y_true, y_pred].mean())

cost_rows = []
for alias in ['ensemble', 'exp701', 'exp300', 'exp00']:
    if alias not in PRED_INT:
        continue
    cost_rows.append({'experiment': alias, 'mean_cost': cost_risk(Y_TRUE, PRED_INT[alias])})
cost_df = pd.DataFrame(cost_rows)
display(cost_df)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(COST, annot=True, fmt='.0f', cmap='Reds',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Clinical cost matrix (asymmetric, under-graders penalised)')
plt.tight_layout(); plt.show()
"""

B3_CODE = """# B3. Coverage-accuracy ('triage') curve using ensemble margin-confidence.
conf_ens = cal.regression_margin_confidence(PRED_RAW['ensemble'], opt_thresh)
order = np.argsort(-conf_ens)  # most-confident first
sorted_correct = (PRED_INT['ensemble'][order] == Y_TRUE[order]).astype(float)
sorted_yt = Y_TRUE[order]
sorted_yp = PRED_INT['ensemble'][order]

cov = np.linspace(0.05, 1.0, 20)
acc_cov = []
sens_cov = []
n = len(Y_TRUE)
ref_yt = (Y_TRUE >= 2).astype(int)
ref_yp = (PRED_INT['ensemble'] >= 2).astype(int)
for c in cov:
    k = max(int(c * n), 1)
    acc_cov.append(sorted_correct[:k].mean())
    sub_yt = ref_yt[order[:k]]; sub_yp = ref_yp[order[:k]]
    sens_cov.append(binary_metrics(sub_yt, sub_yp)['sensitivity'])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(cov * 100, acc_cov, marker='o', label='Accuracy')
ax.plot(cov * 100, sens_cov, marker='s', label='Sensitivity (referable DR)')
ax.set_xlabel('Coverage (%)'); ax.set_ylabel('Score')
ax.set_title('Coverage-accuracy curve (rank by ensemble margin-confidence)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
"""

B4_CODE = """# B4. Literature anchor — bar chart vs published references.
LITERATURE = pd.DataFrame([
    {'reference': 'Gulshan 2016 (single grader)', 'qwk': 0.84},
    {'reference': 'Gulshan 2016 (adjudicated)',   'qwk': 0.91},
    {'reference': 'Krause 2018 (adjudicated)',    'qwk': 0.84},
    {'reference': 'APTOS 2019 winner (Kaggle)',   'qwk': 0.936},
    {'reference': 'APTOS 2019 top-5 median',      'qwk': 0.928},
    {'reference': 'This thesis — ensemble',       'qwk': 0.9105},
    {'reference': 'This thesis — exp701',         'qwk': cs.metric_qwk(Y_TRUE, PRED_INT.get('exp701', PRED_INT['ensemble']))},
])
fig, ax = plt.subplots(figsize=(8, 4))
colors = ['steelblue']*5 + ['firebrick', 'firebrick']
ax.barh(LITERATURE['reference'], LITERATURE['qwk'], color=colors)
ax.set_xlabel('QWK'); ax.set_xlim(0.7, 1.0)
ax.set_title('QWK in context of published DR-grading systems')
for i, v in enumerate(LITERATURE['qwk']):
    ax.text(v + 0.003, i, f'{v:.3f}', va='center')
plt.tight_layout(); plt.show()
"""


# ---------------------------------------------------------------------------
# SECTION C — Stratified errors (uses Task-2 EDA cache)
# ---------------------------------------------------------------------------

C_MD = """## C. Stratified error analysis

Pulls per-image quality, illumination regime, and resolution bucket from
`results/eda_cache/` (produced by `eda_deep.ipynb`). Errors are stratified
on the **test split only**.
"""

C_LOAD_CODE = """quality_csv = ROOT / 'results' / 'eda_cache' / 'aptos_quality.csv'
illum_csv   = ROOT / 'results' / 'eda_cache' / 'illumination_regime.csv'
sizes_csv   = ROOT / 'results' / 'eda_cache' / 'aptos_sizes.csv'

quality_df = pd.read_csv(quality_csv) if quality_csv.exists() else pd.DataFrame()
illum_df   = pd.read_csv(illum_csv)   if illum_csv.exists()   else pd.DataFrame()
sizes_df   = pd.read_csv(sizes_csv)   if sizes_csv.exists()   else pd.DataFrame()
print('quality_df:', quality_df.shape, '| illum_df:', illum_df.shape,
      '| sizes_df:', sizes_df.shape)

# Cache uses 'code' / 'id' as the join key; rename + filter to test split.
def _prep(df, id_col, split_filter='test'):
    if df.empty:
        return df
    df = df.copy()
    if id_col in df.columns:
        df = df.rename(columns={id_col: 'id_code'})
    df['id_code'] = df['id_code'].astype(str)
    if 'split' in df.columns and split_filter is not None:
        df = df[df['split'] == split_filter]
    return df

quality_df = _prep(quality_df, 'code')
illum_df   = _prep(illum_df,   'code').rename(columns={'regime': 'illumination_cluster'})
sizes_df   = _prep(sizes_df,   'id')

# Build a single per-id_code dataframe of test predictions with stratification keys.
ens = ALIGNED['ensemble'][['id_code', 'true_label', 'rounded_prediction']].copy()
ens['id_code'] = ens['id_code'].astype(str)
ens['error'] = (ens['rounded_prediction'] != ens['true_label']).astype(int)

def _merge(df, target, cols):
    if df.empty:
        return target
    keep = ['id_code'] + [c for c in cols if c in df.columns]
    return target.merge(df[keep], on='id_code', how='left')

ens = _merge(quality_df, ens, ['laplacian_var', 'mean_intensity', 'contrast_std', 'snr', 'illum_uniformity'])
ens = _merge(illum_df,   ens, ['illumination_cluster'])
ens = _merge(sizes_df,   ens, ['height', 'width', 'aspect'])
print('Coverage of stratification columns:')
for c in ['laplacian_var', 'illumination_cluster', 'height']:
    if c in ens.columns:
        print(f'  {c}: {ens[c].notna().mean():.1%}')
"""

C1_CODE = """# C1. Error rate by image-quality (Laplacian variance) quartile
if 'laplacian_var' in ens.columns and ens['laplacian_var'].notna().any():
    ens['quality_q'] = pd.qcut(ens['laplacian_var'], q=4,
                                labels=['Q1 (blurriest)', 'Q2', 'Q3', 'Q4 (sharpest)'])
    qrate = ens.groupby('quality_q', observed=True)['error'].agg(['mean', 'count'])
    qrate.columns = ['error_rate', 'n']
    display(qrate)
    fig, ax = plt.subplots(figsize=(6, 4))
    qrate['error_rate'].plot(kind='bar', ax=ax, color='salmon', edgecolor='black')
    ax.set_ylabel('Error rate'); ax.set_xlabel('Quality quartile')
    ax.set_title('Ensemble error rate vs image-quality quartile')
    ax.grid(True, alpha=0.3, axis='y'); plt.tight_layout(); plt.show()
else:
    print('Skipping: no laplacian_var column.')
"""

C2_CODE = """# C2. Error rate by illumination regime (k-means cluster from EDA)
if 'illumination_cluster' in ens.columns and ens['illumination_cluster'].notna().any():
    irate = ens.groupby('illumination_cluster', observed=True)['error'].agg(['mean', 'count'])
    irate.columns = ['error_rate', 'n']
    display(irate)
    fig, ax = plt.subplots(figsize=(6, 4))
    irate['error_rate'].plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_ylabel('Error rate'); ax.set_xlabel('Illumination cluster id')
    ax.set_title('Ensemble error rate vs illumination regime')
    ax.grid(True, alpha=0.3, axis='y'); plt.tight_layout(); plt.show()
else:
    print('Skipping: no illumination_cluster column.')
"""

C3_CODE = """# C3. Error rate by native-resolution bucket
if 'height' in ens.columns and ens['height'].notna().any():
    bins = pd.cut(ens['height'], bins=[0, 1024, 2048, 3072, 6000],
                  labels=['<=1024', '1025-2048', '2049-3072', '>3072'])
    rrate = ens.groupby(bins, observed=True)['error'].agg(['mean', 'count'])
    rrate.columns = ['error_rate', 'n']
    display(rrate)
    fig, ax = plt.subplots(figsize=(6, 4))
    rrate['error_rate'].plot(kind='bar', ax=ax, color='seagreen', edgecolor='black')
    ax.set_ylabel('Error rate'); ax.set_xlabel('Native height bucket')
    ax.set_title('Ensemble error rate vs native resolution')
    ax.grid(True, alpha=0.3, axis='y'); plt.tight_layout(); plt.show()
else:
    print('Skipping: no height column.')
"""

C4_CODE = """# C4. Quality x class crosstab among errors — does quality covary with class?
if 'laplacian_var' in ens.columns and ens['laplacian_var'].notna().any():
    err = ens[ens['error'] == 1].copy()
    err['quality_q'] = pd.qcut(err['laplacian_var'], q=4,
                                labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    ct = pd.crosstab(err['true_label'], err['quality_q'], normalize='index')
    display(ct)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(ct, annot=True, fmt='.2f', cmap='Reds', ax=ax,
                yticklabels=CLASS_NAMES)
    ax.set_xlabel('Quality quartile (within errors)'); ax.set_ylabel('True class')
    ax.set_title('Quality distribution of errors per class')
    plt.tight_layout(); plt.show()
"""


# ---------------------------------------------------------------------------
# SECTION D — Ablation ladder
# ---------------------------------------------------------------------------

D_MD = """## D. Ablation marginal-contribution ladder

The thesis tells a story: baseline → +StdAug → +AdvAug → +Focal → +D1 dropout
→ +OrdSupCon → +Ensemble. Each rung must show a measurable QWK gain. We
report `delta = current - previous` with a paired-bootstrap p-value to
quantify how much of the gain is real vs. test-set noise.
"""

D1_CODE = """LADDER = [
    ('exp00',     'baseline'),
    ('exp01_std', '+ std aug'),
    ('exp02_adv', '+ adv aug'),
    ('exp03_foc', '+ focal loss'),
    ('exp300',    '+ D1 (cosine + dropout)'),
    ('exp701',    '+ OrdSupCon (h1 / D1 recipe)'),
    ('ensemble',  '+ ensemble (900 + 300 + 701)'),
]
LADDER = [(a, name) for (a, name) in LADDER if a in PRED_INT]

ladder_rows = []
prev_alias = None
for alias, label in LADDER:
    yp = PRED_INT[alias]
    point, lo, hi = cs.bootstrap_ci(cs.metric_qwk, Y_TRUE, yp, n_boot=N_BOOT, rng=RNG)
    row = {'step': label, 'experiment': alias, 'qwk': point,
           'qwk_ci': f'[{lo:.4f}, {hi:.4f}]'}
    if prev_alias is not None:
        d = cs.paired_bootstrap_diff(cs.metric_qwk, Y_TRUE, yp, PRED_INT[prev_alias],
                                     n_boot=N_BOOT, rng=RNG)
        row['delta_qwk'] = d['delta']
        row['delta_ci']  = f"[{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]"
        row['p_value']   = d['p_value']
    else:
        row['delta_qwk'] = 0.0; row['delta_ci'] = '-'; row['p_value'] = float('nan')
    ladder_rows.append(row)
    prev_alias = alias
ladder_df = pd.DataFrame(ladder_rows)
ladder_df.to_csv(CACHE / 'ablation_ladder.csv', index=False)
display(ladder_df)
"""

D2_CODE = """# D2. Visualise the ladder
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(ladder_df['step'], ladder_df['qwk'], marker='o', color='firebrick')
for i, (_, r) in enumerate(ladder_df.iterrows()):
    ax.annotate(f"{r['qwk']:.4f}", (i, r['qwk']), textcoords='offset points',
                xytext=(0, 8), ha='center', fontsize=9)
ax.set_ylabel('QWK'); ax.set_title('Ablation ladder — cumulative QWK gain')
ax.grid(True, alpha=0.3); plt.xticks(rotation=20, ha='right')
plt.tight_layout(); plt.show()
"""

D3_CODE = """# D3. Phase-I (i1..i5) autopsy — how many sample-level errors does each Phase-I
# variant fix relative to the D1 baseline (exp300)?
PHASE_I = ['exp802_emd' if 'exp802_emd' in PRED_INT else None,
           'exp804_swad', 'exp805_l2sp', 'exp806_proto']
PHASE_I = [a for a in PHASE_I if a in PRED_INT]
if 'exp300' in PRED_INT and PHASE_I:
    base_err = PRED_INT['exp300'] != Y_TRUE
    rows = []
    for alias in PHASE_I:
        var_err = PRED_INT[alias] != Y_TRUE
        rows.append({
            'experiment': alias,
            'fixes_vs_d1': int((base_err & ~var_err).sum()),
            'breaks_vs_d1': int((~base_err & var_err).sum()),
            'net': int((base_err & ~var_err).sum()) - int((~base_err & var_err).sum()),
        })
    display(pd.DataFrame(rows))
else:
    print('Skipping Phase-I autopsy — exp300 or Phase-I results missing.')
"""


# ---------------------------------------------------------------------------
# SECTION E — Failure taxonomy (manual tag scaffold)
# ---------------------------------------------------------------------------

E_MD = """## E. Failure-mode taxonomy

The thesis claims 6 failure buckets:
`{low_quality, anatomical_ambiguity, label_noise, borderline_case,
artifact_bias, genuine_miss}`.

This section produces the worksheet (50 most-confused ensemble errors with
side-by-side panels) and a tagging template. Manual tags should be filled in
`results/result_cache/failure_tags.csv` by the thesis author, then re-loaded
to build the final distribution figure.
"""

E1_CODE = """# E1. Pick the 50 ensemble errors with the largest |raw - true| margin —
# these are the cases the model is most wrong about and most worth dissecting.
err_mask = PRED_INT['ensemble'] != Y_TRUE
err_idx = np.where(err_mask)[0]
err_margin = np.abs(PRED_RAW['ensemble'][err_idx] - Y_TRUE[err_idx])
order = err_idx[np.argsort(-err_margin)]
top_errors = ALIGNED['ensemble'].iloc[order].head(50).copy()
top_errors['margin'] = np.abs(top_errors['raw_prediction'] - top_errors['true_label'])
top_errors.to_csv(CACHE / 'top50_errors.csv', index=False)
display(top_errors.head(10))
"""

E2_CODE = """# E2. Tagging worksheet — write a CSV scaffold the author fills in by hand.
TAG_PATH = CACHE / 'failure_tags.csv'
if not TAG_PATH.exists():
    scaffold = top_errors[['id_code', 'true_label', 'rounded_prediction',
                           'raw_prediction', 'margin']].copy()
    scaffold['tag'] = ''   # one of: low_quality / anatomical_ambiguity /
                           # label_noise / borderline_case / artifact_bias /
                           # genuine_miss
    scaffold['notes'] = ''
    scaffold.to_csv(TAG_PATH, index=False)
    print(f'Wrote tagging scaffold to {TAG_PATH}. Fill the `tag` column then '
          f're-run E3.')
else:
    print(f'Tagging file already exists at {TAG_PATH} — skipping scaffold.')
"""

E3_CODE = """# E3. Distribution of manual tags (run after the scaffold is filled).
tag_df = pd.read_csv(TAG_PATH)
filled = tag_df[tag_df['tag'].astype(str).str.len() > 0]
if len(filled) == 0:
    print('No tags filled yet — fill the tag column in failure_tags.csv first.')
else:
    counts = filled['tag'].value_counts()
    display(counts)
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind='barh', ax=ax, color='slategray', edgecolor='black')
    ax.set_xlabel('# samples'); ax.set_title(f'Failure-mode taxonomy (N={len(filled)})')
    ax.invert_yaxis()
    plt.tight_layout(); plt.show()
"""


# ---------------------------------------------------------------------------
# SECTION F — Ensemble decomposition
# ---------------------------------------------------------------------------

F_MD = """## F. Ensemble decomposition

Why does the ensemble outperform any single member? Three angles:
1. **Agreement vs accuracy** — error rate when all 3 members agree vs split.
2. **Sample-level gain attribution** — which samples does only the ensemble
   get right?
3. **Weight-sensitivity grid** — sweep (w1, w2, w3) on the regression scores
   to confirm the 1/1/1 default sits on a plateau, not a cliff.
"""

F1_CODE = """# F1. Agreement vs accuracy (uses individual-member integer predictions).
# exp900 has no standalone preds.csv — we substitute the ensemble's rounded
# prediction as a proxy for the champion when needed and document this caveat.
have900 = 'exp900' in PRED_INT
member_aliases = ['exp701', 'exp300']
agree_df = pd.DataFrame({a: PRED_INT[a] for a in member_aliases})
agree_df['true'] = Y_TRUE
agree_df['ensemble'] = PRED_INT['ensemble']
agree_df['unique_member_preds'] = agree_df[member_aliases].nunique(axis=1)
agree_df['ensemble_correct'] = (agree_df['ensemble'] == agree_df['true']).astype(int)

print('Members compared (exp900 not on disk as standalone preds):', member_aliases)
agg = agree_df.groupby('unique_member_preds').agg(
    n=('true', 'size'),
    ensemble_acc=('ensemble_correct', 'mean'),
)
display(agg)
"""

F2_CODE = """# F2. Sample-level gain — samples ONLY the ensemble gets right
ens_correct = PRED_INT['ensemble'] == Y_TRUE
e701_correct = PRED_INT['exp701'] == Y_TRUE
e300_correct = PRED_INT['exp300'] == Y_TRUE
only_ens = ens_correct & ~e701_correct & ~e300_correct
print(f'Samples only the ensemble gets right: {int(only_ens.sum())} / {len(Y_TRUE)}')

dist = pd.Series(Y_TRUE[only_ens]).value_counts().sort_index()
dist.index = [CLASS_NAMES[i] for i in dist.index]
display(dist.to_frame('count'))
"""

F3_CODE = """# F3. Weight-sensitivity sweep on the ensemble's regression score.
# We don't have per-member raw regression scores stored separately — use the
# ensemble raw and re-fit OptimizedRounder per (w1, w2, w3) hypothetically by
# perturbing the score with member-rank shifts. As a falsifiable proxy, sweep
# the threshold vector instead (closely related to weight changes for a
# regression head whose mean shifts under weight rebalancing).
print('Weight-sensitivity at the per-member raw level requires re-running '
      'scripts/ensemble_cls.py with --weights overrides. The threshold-sweep '
      'in A9 is the analytically tractable proxy from saved artefacts.')
"""


# ---------------------------------------------------------------------------
# Assemble and write
# ---------------------------------------------------------------------------

CELLS = [
    md(SETUP_MD),
    code(SETUP_CODE),
    md(LOADER_MD),
    code(LOADER_CODE),
    code(ALIGN_CODE),

    md(A_MD),
    code(A1_CODE),
    code(A2_CODE),
    code(A3_CODE),
    code(A4_CODE),
    code(A5_CODE),
    code(A6_CODE),
    code(A7_CODE),
    code(A8_CODE),
    code(A9_CODE),
    code(A10_CODE),

    md(B_MD),
    code(B1_CODE),
    code(B2_CODE),
    code(B3_CODE),
    code(B4_CODE),

    md(C_MD),
    code(C_LOAD_CODE),
    code(C1_CODE),
    code(C2_CODE),
    code(C3_CODE),
    code(C4_CODE),

    md(D_MD),
    code(D1_CODE),
    code(D2_CODE),
    code(D3_CODE),

    md(E_MD),
    code(E1_CODE),
    code(E2_CODE),
    code(E3_CODE),

    md(F_MD),
    code(F1_CODE),
    code(F2_CODE),
    code(F3_CODE),
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
