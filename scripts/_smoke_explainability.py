"""CPU-only smoke-test for Task 3 explainability plumbing.

Covers the math that does NOT need a GPU / trained checkpoint:
  * fundus_cv detectors (FOV, disc, fovea, MA, hemorrhage, exudate)
  * faithfulness metrics (fov_energy, anatomy_breakdown, pointing_game,
    cam_lesion_dice, cam_pairwise_iou, compute_lesion_proxies)
  * explainers.image_to_tensor / tensor_to_display_rgb round-trip
  * weighted_cam ensemble math

Anything that calls ``model(...)`` is deferred to the remote GPU run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis import explainers as ex
from src.analysis import faithfulness as faith
from src.analysis import fundus_cv
from src.dataset import ben_graham_preprocess

TEST_DIR = ROOT / "data" / "test_split"
assert TEST_DIR.exists(), f"missing {TEST_DIR}"
imgs = sorted(TEST_DIR.glob("*.png"))[:3]
assert imgs, "no test images found"
print(f"[load] using {len(imgs)} probe images")

for ipath in imgs:
    bgr = cv2.imread(str(ipath))
    assert bgr is not None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bg = ben_graham_preprocess(rgb.copy(), size=512)
    H, W = bg.shape[:2]
    assert bg.dtype == np.uint8 and bg.shape == (512, 512, 3)

    # --- fundus_cv ---
    fov = fundus_cv.retinal_fov_mask(bg)
    frac = float(fov.mean())
    disc_loc, disc_r = fundus_cv.detect_optic_disc(bg, fov_mask=fov)
    fovea_loc = fundus_cv.detect_fovea(bg, fov_mask=fov, optic_disc=disc_loc)
    ma = fundus_cv.ma_candidates(bg)
    hem = fundus_cv.hemorrhage_candidates(bg)
    exu = fundus_cv.hard_exudate_candidates(bg)
    print(f"[fundus] {ipath.name} fov_frac={frac:.3f} disc={disc_loc} fovea={fovea_loc} "
          f"ma_px={int(ma.sum())} hem_px={int(hem.sum())} exu_px={int(exu.sum())}")
    assert fov.shape == (H, W)
    for m in (ma, hem, exu):
        assert m.shape == (H, W) and m.dtype == bool or m.dtype == np.uint8

    # --- synthetic CAM (bright blob inside FOV) ---
    cam = np.zeros((H, W), dtype=np.float32)
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    cam = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (H * 0.15) ** 2)).astype(np.float32)

    # --- faithfulness spatial metrics ---
    fov_e = faith.fov_energy_fraction(cam, fov, top_pct=0.2)
    assert 0.0 <= fov_e <= 1.0
    assert fov_e >= 0.9, f"center blob should land in FOV ({fov_e:.3f})"

    disc_mask = np.zeros((H, W), dtype=np.uint8)
    if disc_loc is not None:
        cv2.circle(disc_mask, disc_loc, int(disc_r), 1, -1)
    fovea_mask = np.zeros((H, W), dtype=np.uint8)
    if fovea_loc is not None:
        cv2.circle(fovea_mask, fovea_loc, int(H * 0.08), 1, -1)
    ana = faith.anatomy_breakdown(cam, fov, disc_mask, fovea_mask, top_pct=0.2)
    shares_sum = sum(ana.values())
    assert abs(shares_sum - 1.0) < 1e-3, f"anatomy shares should sum to 1 (got {shares_sum:.4f})"

    # --- lesion-overlap metrics ---
    proxies = faith.compute_lesion_proxies(bg)
    for name, mask in proxies.items():
        pg = faith.pointing_game(cam, mask)
        dc = faith.cam_lesion_dice(cam, mask, top_pct=0.2)
        assert isinstance(pg, bool)
        assert 0.0 <= dc <= 1.0
        print(f"[faith] {ipath.name} {name}: pointing={pg} dice={dc:.3f}")

    # --- TTA-style pairwise IoU: same CAM -> IoU == 1 ---
    iou_same = faith.cam_pairwise_iou([cam, cam, cam], top_pct=0.2)
    assert abs(iou_same - 1.0) < 1e-6, f"identical CAMs should give IoU 1.0 (got {iou_same})"
    shifted = np.roll(cam, shift=W // 4, axis=1)
    iou_shift = faith.cam_pairwise_iou([cam, shifted], top_pct=0.2)
    assert 0.0 <= iou_shift < 1.0, "shifted CAMs should have IoU < 1"
    print(f"[faith] iou(same)={iou_same:.3f} iou(shift)={iou_shift:.3f}")

    # --- evaluate_sample all-in-one ---
    out = faith.evaluate_sample(cam, bg, fov_mask=fov, disc_mask=disc_mask,
                                fovea_mask=fovea_mask, lesion_masks=proxies, top_pct=0.2)
    expected = {"fov_energy_top20", "share_optic_disc", "share_fovea",
                "share_rest_of_retina", "share_background"}
    assert expected.issubset(out.keys()), f"missing keys: {expected - out.keys()}"
    for name in ("ma", "hemorrhage", "exudate"):
        assert f"pointing_{name}" in out and f"dice_{name}" in out

# --- explainers: tensor round-trip ---
img = cv2.cvtColor(cv2.imread(str(imgs[0])), cv2.COLOR_BGR2RGB)
bg = ben_graham_preprocess(img, size=512)
t = ex.image_to_tensor(bg, device="cpu")
assert tuple(t.shape) == (1, 3, 512, 512)
disp = ex.tensor_to_display_rgb(t)
assert disp.shape == bg.shape and disp.dtype == np.uint8
diff = float(np.abs(disp.astype(np.int32) - bg.astype(np.int32)).mean())
assert diff < 2.0, f"tensor round-trip error too large ({diff:.2f})"
print(f"[explainers] tensor round-trip mean-abs-diff={diff:.3f}")

# --- weighted_cam math ---
cam_a = np.random.default_rng(0).random((64, 64)).astype(np.float32)
cam_b = np.random.default_rng(1).random((64, 64)).astype(np.float32)
w_cam = ex.weighted_cam([cam_a, cam_b], [0.6, 0.4])
assert w_cam.shape == (64, 64)
assert 0.0 <= float(w_cam.min()) and float(w_cam.max()) <= 1.0
# single-CAM weighted should match normalized input
single = ex.weighted_cam([cam_a], [1.0])
assert single.shape == cam_a.shape

# --- resolve-to-image-shape in evaluate_sample (cam != image shape) ---
small_cam = cv2.resize(cam, (32, 32))
out_small = faith.evaluate_sample(small_cam, bg, fov_mask=fov,
                                  lesion_masks=proxies, top_pct=0.2)
assert "fov_energy_top20" in out_small
print(f"[explainers] weighted_cam OK; resized-cam OK")

print("\nALL SMOKE CHECKS PASSED")
