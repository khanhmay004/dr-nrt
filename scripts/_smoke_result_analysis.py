"""Smoke-test the core logic of notebooks/result_analysis.ipynb.

Runs each section's load + first-pass call so syntax / API / data-shape bugs
surface before opening the notebook.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES, NUM_CLASSES
from src.evaluate import OptimizedRounder, regression_to_class, quadratic_weighted_kappa
from src.analysis import confusion_stats as cs
from src.analysis import calibration as cal

RESULTS = ROOT / "results"
RNG = np.random.default_rng(42)


def find_preds(d: Path) -> Path | None:
    matches = list(d.glob("*_preds.csv"))
    if not matches:
        return None
    expected = [m for m in matches if "expected_grade_opt" in m.name]
    return expected[0] if expected else matches[0]


REGISTRY = {
    "ensemble":  RESULTS / "ensemble_900_300_701",
    "exp701":    RESULTS / "exp701_h1_ordsupcon_d1recipe",
    "exp300":    RESULTS / "exp300_d1_dropout_cosine",
    "exp00":     RESULTS / "exp00_baseline",
    "exp01_std": RESULTS / "exp01_std_aug",
    "exp02_adv": RESULTS / "exp02_adv_aug",
    "exp03_foc": RESULTS / "exp03_focal_loss",
    "exp07_reg": RESULTS / "exp07_regression",
    "exp12_thr": RESULTS / "exp12_opt_thresh_opA",
    "exp103_a1": RESULTS / "exp103_a1_ordsupcon_aptos",
    "exp605_a1v3": RESULTS / "exp605_a1v3_ordsupcon_40ep",
    "exp700_lp": RESULTS / "exp700_h0_linear_probe_a2",
    "exp702_lpft": RESULTS / "exp702_h2_lpft_a2",
    "exp804_swad": RESULTS / "exp804_i3_swad_on_d1",
    "exp805_l2sp": RESULTS / "exp805_i4_l2sp_a2_d1recipe",
    "exp806_proto": RESULTS / "exp806_i5_prototype_head_a2_emd",
}

PREDS = {}
for alias, d in REGISTRY.items():
    if not d.exists():
        continue
    p = find_preds(d)
    if p is None:
        continue
    df = pd.read_csv(p).sort_values("id_code").reset_index(drop=True)
    PREDS[alias] = df

print(f"[loader] loaded experiments: {list(PREDS.keys())}")
print(f"[loader] sizes: {[(a, len(df)) for a, df in PREDS.items()]}")

common = set.intersection(*[set(v["id_code"]) for v in PREDS.values()])
common = sorted(common)
print(f"[align] common ids: {len(common)}")

ALIGNED = {a: df[df["id_code"].isin(common)].sort_values("id_code").reset_index(drop=True)
           for a, df in PREDS.items()}
Y_TRUE = ALIGNED["ensemble"]["true_label"].to_numpy()
PRED_INT = {a: ALIGNED[a]["rounded_prediction"].to_numpy().astype(int) for a in ALIGNED}
PRED_RAW = {a: ALIGNED[a]["raw_prediction"].to_numpy().astype(float) for a in ALIGNED}
for a, df in ALIGNED.items():
    assert (df["true_label"].to_numpy() == Y_TRUE).all(), f"true_label mismatch for {a}"

# --- A1 bootstrap CI on QWK
point, lo, hi = cs.bootstrap_ci(cs.metric_qwk, Y_TRUE, PRED_INT["ensemble"], n_boot=200, rng=RNG)
print(f"[A1] ensemble QWK = {point:.4f} CI[{lo:.4f}, {hi:.4f}]")

# --- A2 per-class F1 CI for class 3 (Severe)
fn = cs.metric_per_class_f1(3)
pf, plo, phi = cs.bootstrap_ci(fn, Y_TRUE, PRED_INT["ensemble"], n_boot=200, rng=RNG)
print(f"[A2] f1[Severe] = {pf:.4f} CI[{plo:.4f}, {phi:.4f}]")

# --- A3 paired bootstrap
diff = cs.paired_bootstrap_diff(cs.metric_qwk, Y_TRUE,
                                 PRED_INT["ensemble"], PRED_INT["exp701"],
                                 n_boot=200, rng=RNG)
print(f"[A3] ensemble - exp701 QWK delta: {diff}")

# --- A4 McNemar
mc = cs.mcnemar_test(Y_TRUE, PRED_INT["ensemble"], PRED_INT["exp701"])
print(f"[A4] McNemar: {mc}")

# --- A5 confusion CI
dec = cs.confusion_with_ci(Y_TRUE, PRED_INT["ensemble"], NUM_CLASSES, n_boot=200, rng=RNG)
print(f"[A5] cm shapes: raw={dec['cm_raw'].shape} ci_lo={dec['cm_row_ci_lo'].shape}")

# --- A6 adjacent
adj = cs.adjacent_error_rate(Y_TRUE, PRED_INT["ensemble"], NUM_CLASSES)
print(f"[A6] adjacent_frac={adj['adjacent_frac']:.3f} per_class={adj['per_class_adjacent']}")

# --- A7 off-by-N
off = cs.off_by_n_distribution(Y_TRUE, PRED_INT["ensemble"], NUM_CLASSES)
print(f"[A7] off-by-N matrix shape: {off.shape}")

# --- A8 kappa split
ks = cs.kappa_split(Y_TRUE, PRED_INT["ensemble"])
print(f"[A8] kappa split: {ks}")

# --- A9 threshold sensitivity
opt = OptimizedRounder()
opt.fit(PRED_RAW["ensemble"], Y_TRUE)
opt_thresh = list(opt.thresholds)
print(f"[A9] optimum thresholds: {opt_thresh}")
qwk_at_opt = quadratic_weighted_kappa(Y_TRUE, regression_to_class(PRED_RAW["ensemble"], opt_thresh))
print(f"[A9] QWK at fitted thresholds: {qwk_at_opt:.4f}")

# --- A10 ordinal-margin reliability
binned, conf = cal.regression_calibration_curve(
    PRED_RAW["ensemble"], PRED_INT["ensemble"], Y_TRUE, opt_thresh, n_bins=10,
)
print(f"[A10] reliability bins: counts={binned['count'].tolist()}")

# --- B1 binary metrics
def binary_metrics(yt, yp):
    yt, yp = yt.astype(int), yp.astype(int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    return {
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "ppv": tp / (tp + fp) if (tp + fp) else 0.0,
        "npv": tn / (tn + fn) if (tn + fn) else 0.0,
    }

for thresh, name in [(2, "referable"), (3, "sight-threatening")]:
    yt = (Y_TRUE >= thresh).astype(int)
    yp = (PRED_INT["ensemble"] >= thresh).astype(int)
    print(f"[B1] {name}: {binary_metrics(yt, yp)}")

# --- B2 cost
COST = np.array([
    [0, 1, 2, 3, 4],
    [2, 0, 1, 2, 3],
    [4, 2, 0, 1, 2],
    [8, 4, 2, 0, 1],
    [16, 8, 4, 2, 0],
])
print(f"[B2] mean cost ensemble: {COST[Y_TRUE, PRED_INT['ensemble']].mean():.3f}")

# --- B3 coverage curve
order = np.argsort(-conf)
correct = (PRED_INT["ensemble"][order] == Y_TRUE[order]).astype(float)
for c in [0.5, 0.8, 1.0]:
    k = max(int(c * len(Y_TRUE)), 1)
    print(f"[B3] coverage={c:.0%} acc={correct[:k].mean():.4f}")

# --- C load EDA cache
quality_csv = ROOT / "results" / "eda_cache" / "aptos_quality.csv"
illum_csv = ROOT / "results" / "eda_cache" / "illumination_regime.csv"
sizes_csv = ROOT / "results" / "eda_cache" / "aptos_sizes.csv"
print(f"[C] quality_csv exists: {quality_csv.exists()}")
print(f"[C] illum_csv exists: {illum_csv.exists()}")
print(f"[C] sizes_csv exists: {sizes_csv.exists()}")
if quality_csv.exists():
    qdf = pd.read_csv(quality_csv)
    print(f"[C] quality_df cols: {list(qdf.columns)}")
if illum_csv.exists():
    idf = pd.read_csv(illum_csv)
    print(f"[C] illum_df cols: {list(idf.columns)}")
if sizes_csv.exists():
    sdf = pd.read_csv(sizes_csv)
    print(f"[C] sizes_df cols: {list(sdf.columns)}")

# --- C merge sanity (replicates the notebook's stratification join)
def _prep(df, id_col, split_filter="test"):
    if df.empty:
        return df
    df = df.copy()
    if id_col in df.columns:
        df = df.rename(columns={id_col: "id_code"})
    df["id_code"] = df["id_code"].astype(str)
    if "split" in df.columns and split_filter is not None:
        df = df[df["split"] == split_filter]
    return df

ens = ALIGNED["ensemble"][["id_code", "true_label", "rounded_prediction"]].copy()
ens["id_code"] = ens["id_code"].astype(str)
ens["error"] = (ens["rounded_prediction"] != ens["true_label"]).astype(int)
qdf = _prep(pd.read_csv(quality_csv), "code")
idf = _prep(pd.read_csv(illum_csv), "code").rename(columns={"regime": "illumination_cluster"})
sdf = _prep(pd.read_csv(sizes_csv), "id")
ens = ens.merge(qdf[["id_code", "laplacian_var"]], on="id_code", how="left")
ens = ens.merge(idf[["id_code", "illumination_cluster"]], on="id_code", how="left")
ens = ens.merge(sdf[["id_code", "height"]], on="id_code", how="left")
print(f"[C-merge] coverage laplacian_var={ens['laplacian_var'].notna().mean():.1%} "
      f"illum={ens['illumination_cluster'].notna().mean():.1%} "
      f"height={ens['height'].notna().mean():.1%}")

# --- D ladder
LADDER = [("exp00", "baseline"), ("exp01_std", "+std"), ("exp02_adv", "+adv"),
          ("exp03_foc", "+focal"), ("exp300", "+D1"), ("exp701", "+OrdSupCon"),
          ("ensemble", "+ensemble")]
LADDER = [(a, n) for a, n in LADDER if a in PRED_INT]
prev = None
for alias, name in LADDER:
    point, lo, hi = cs.bootstrap_ci(cs.metric_qwk, Y_TRUE, PRED_INT[alias], n_boot=200, rng=RNG)
    msg = f"[D] {name:<14} qwk={point:.4f} [{lo:.4f}, {hi:.4f}]"
    if prev is not None:
        d = cs.paired_bootstrap_diff(cs.metric_qwk, Y_TRUE,
                                      PRED_INT[alias], PRED_INT[prev],
                                      n_boot=200, rng=RNG)
        msg += f"  delta={d['delta']:+.4f} p={d['p_value']:.4f}"
    print(msg)
    prev = alias

# --- E top errors
err_mask = PRED_INT["ensemble"] != Y_TRUE
err_idx = np.where(err_mask)[0]
margins = np.abs(PRED_RAW["ensemble"][err_idx] - Y_TRUE[err_idx])
top = err_idx[np.argsort(-margins)][:50]
print(f"[E] total ensemble errors: {err_mask.sum()}, top-50 max margin: {margins[np.argsort(-margins)][0]:.3f}")

# --- F decomposition
e701_correct = PRED_INT["exp701"] == Y_TRUE
e300_correct = PRED_INT["exp300"] == Y_TRUE
ens_correct = PRED_INT["ensemble"] == Y_TRUE
only_ens = ens_correct & ~e701_correct & ~e300_correct
print(f"[F] only-ensemble correct: {only_ens.sum()} of {len(Y_TRUE)}")

print("\nALL SMOKE CHECKS PASSED")
