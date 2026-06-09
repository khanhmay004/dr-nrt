"""Smoke-test the core logic of notebooks/confusion_analysis.ipynb.

Focuses on the bits most likely to break: feature alignment from the EDA
cache, prototype-distance deltas, JS-div matrix, and per-pair sampling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES, NUM_CLASSES, TEST_IMG_DIR, TRAIN_IMG_DIR
from src.analysis import confusion_stats as cs
from src.analysis import calibration as cal
from src.evaluate import OptimizedRounder

RESULTS = ROOT / "results"
RNG = np.random.default_rng(42)


def _resolve(p: Path, fb: Path) -> Path:
    return p if p.exists() else fb


TEST_IMG_DIR = _resolve(Path(TEST_IMG_DIR), ROOT / "data" / "test_split")


def find_preds(d: Path) -> Path | None:
    matches = list(d.glob("*_preds.csv"))
    if not matches:
        return None
    expected = [m for m in matches if "expected_grade_opt" in m.name]
    return expected[0] if expected else matches[0]


REGISTRY = {
    "ensemble": RESULTS / "ensemble_900_300_701",
    "exp701": RESULTS / "exp701_h1_ordsupcon_d1recipe",
    "exp300": RESULTS / "exp300_d1_dropout_cosine",
}
PREDS = {a: pd.read_csv(find_preds(d)).sort_values("id_code").reset_index(drop=True)
         for a, d in REGISTRY.items() if d.exists() and find_preds(d) is not None}
common = sorted(set.intersection(*[set(v["id_code"]) for v in PREDS.values()]))
ALIGNED = {a: df[df["id_code"].isin(common)].sort_values("id_code").reset_index(drop=True)
           for a, df in PREDS.items()}
Y_TRUE = ALIGNED["ensemble"]["true_label"].to_numpy()
PRED_INT = {a: ALIGNED[a]["rounded_prediction"].to_numpy().astype(int) for a in ALIGNED}
PRED_RAW = {a: ALIGNED[a]["raw_prediction"].to_numpy().astype(float) for a in ALIGNED}
CODES = ALIGNED["ensemble"]["id_code"].to_numpy()
print(f"[load] N={len(Y_TRUE)} codes aligned across experiments")

# --- 1B.1 confusion decomp
d = cs.confusion_with_ci(Y_TRUE, PRED_INT["ensemble"], NUM_CLASSES, n_boot=200, rng=RNG)
print(f"[1B.1] cm_row shape={d['cm_row'].shape} sample_cell[2,3]={d['cm_row'][2,3]:.3f} "
      f"CI=[{d['cm_row_ci_lo'][2,3]:.3f},{d['cm_row_ci_hi'][2,3]:.3f}]")

# --- 1B.2 adjacent
adj = cs.adjacent_error_rate(Y_TRUE, PRED_INT["ensemble"], NUM_CLASSES)
print(f"[1B.2] adjacent_frac={adj['adjacent_frac']:.3f}")

# --- 1B.3 off-by-N
off = cs.off_by_n_distribution(Y_TRUE, PRED_INT["ensemble"], NUM_CLASSES)
print(f"[1B.3] off-by-N shape={off.shape}")

# --- 1B.4 kappa split
k = cs.kappa_split(Y_TRUE, PRED_INT["ensemble"])
print(f"[1B.4] ordinal_gap={k['ordinal_gap']:.3f}")

# --- 1B.5 per-pair McNemar
pp = cs.mcnemar_per_pair(Y_TRUE, PRED_INT["ensemble"], PRED_INT["exp701"])
for (i, j), r in pp.items():
    print(f"[1B.5] ({i},{j}) b01={r['b01']} b10={r['b10']} n={r['n_discordant']} p={r['p_value']:.3f}")

# --- 1B.6 confidence
opt = OptimizedRounder(); opt.fit(PRED_RAW["ensemble"], Y_TRUE)
conf = cal.regression_margin_confidence(PRED_RAW["ensemble"], opt.thresholds)
correct = PRED_INT["ensemble"] == Y_TRUE
print(f"[1B.6] mean conf correct={conf[correct].mean():.3f} wrong={conf[~correct].mean():.3f}")

# --- 1B.8 features
feat_path = RESULTS / "eda_cache" / "aptos_features.npz"
assert feat_path.exists(), f"missing feature cache: {feat_path}"
data = np.load(feat_path, allow_pickle=True)
test_codes = data["test_codes"]; test_feats = data["test"]
code_to_i = {c: i for i, c in enumerate(test_codes)}
idx = np.array([code_to_i.get(c, -1) for c in CODES])
mask = idx >= 0
X_test = test_feats[idx[mask]]
y_test = Y_TRUE[mask]
yp_ens = PRED_INT["ensemble"][mask]
print(f"[1B.8] feature alignment {mask.sum()}/{len(CODES)}  X_test={X_test.shape}")

delta = cs.prototype_distance_delta(X_test, y_test, yp_ens, NUM_CLASSES)
n_err = np.isfinite(delta).sum()
if n_err:
    print(f"[1B.8] proto-delta n={n_err} mean={np.nanmean(delta):+.3f} "
          f"%positive={(delta[np.isfinite(delta)] > 0).mean():.2%}")

# --- 1B.9 JS-divergence
js = cs.class_embedding_js_matrix(X_test, y_test, NUM_CLASSES,
                                   n_bins=30, n_projections=8, rng=RNG)
print(f"[1B.9] JS matrix shape={js.shape}  min_off_diag={js[np.where(~np.eye(NUM_CLASSES, dtype=bool))].min():.3f}  "
      f"max={js.max():.3f}")

# --- 1C sampling + one thumbnail
PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4)]
for (i, j) in PAIRS:
    m = ((Y_TRUE == i) & (PRED_INT["ensemble"] == j)) | \
        ((Y_TRUE == j) & (PRED_INT["ensemble"] == i))
    n_pair = int(m.sum())
    print(f"[1C] pair {i}<->{j}: {n_pair} errors")

# verify thumbnail load works for at least one code
sample_code = CODES[np.argmax(PRED_INT["ensemble"] != Y_TRUE)]
for d in (TEST_IMG_DIR, ROOT / "data" / "train_split"):
    p = d / f"{sample_code}.png"
    if p.exists():
        img = cv2.imread(str(p))
        assert img is not None
        print(f"[1C] thumbnail load OK: {sample_code} from {d.name}")
        break
else:
    print(f"[1C] WARN: could not locate image for {sample_code}")

# --- 1D label-noise candidate
both_same = (PRED_INT["exp701"] == PRED_INT["exp300"]) & (PRED_INT["exp701"] != Y_TRUE)
near = lambda r, y: np.abs(r - y) < 0.3
hi = near(PRED_RAW["exp701"], PRED_INT["exp701"]) & near(PRED_RAW["exp300"], PRED_INT["exp300"])
cand = both_same & hi
print(f"[1D] candidate mislabels: {int(cand.sum())}")

print("\nALL SMOKE CHECKS PASSED")
