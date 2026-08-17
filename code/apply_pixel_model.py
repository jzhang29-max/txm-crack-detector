"""
Apply the trained per-pixel HistGradientBoostingClassifier crack detector
(models/pixel_hgb_final.joblib, see train_pixel_hgb.py) to a single raw TXM
TIFF image and produce a clean, post-processed crack mask.

This script only *applies* the already-trained model -- it does not retrain
or refit anything.

Pipeline:
  1. Load the raw TIFF, robust_normalize it (txm_features.robust_normalize).
  2. compute_feature_stack -> reshape to (H*W, 17) -> model.predict_proba
     -> reshape back to (H, W) crack-probability map.
  3. Post-process the raw >=0.5 threshold mask so it isn't "spotty":
       a. Gaussian-blur the probability map (sigma~2), then hysteresis-
          threshold (confident pixels >=0.5, plus any moderately-likely
          pixel >=0.35 that's connected to a confident one). This is what
          closes small holes inside an otherwise-obvious crack region
          without lowering the bar for isolated low-confidence specks.
       b. Binary closing (small radius) + remove_small_objects to drop tiny
          leftover specks, then fill any remaining small interior holes.
       c. Reject ring/donut artifacts and round dust blobs via regionprops
          shape descriptors:
            - euler_number >= 1 (no interior hole), but ONLY for components
              under ~800px area -- a large sprawling real crack can
              legitimately enclose one small incidental gap.
            - eccentricity >= 0.5 (not round), but skipped for very large
              components (>5000px) -- a big crack region can be broad/
              non-elongated near its core.
       d. Blank out a border margin (Gaussian filtering at large sigma
          creates edge artifacts right at the image boundary -- our largest
          feature scale is sigma=64, and empirically the resulting false-
          positive band decays to zero only by ~80px in, so the margin here
          is 100px, not a token few pixels).
  4. Save <name>_crack_mask.png (crack=black/0, background=white/255),
     <name>_overlay.png (grayscale original with crack pixels in red),
     <name>_stats.csv (one row per kept region).
  5. Print final crack region count and total crack area fraction.

Usage:
    python3 apply_pixel_model.py INPUT.tif --model MODEL.joblib --out-dir OUTDIR
"""

import argparse
import csv
import os
import sys

import joblib
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from txm_features import compute_feature_stack, robust_normalize, N_FEATURES

# Post-processing knobs.
BLUR_SIGMA = 2.0
CLOSING_RADIUS = 2
MIN_SIZE = 150
HOLE_CHECK_MAX_AREA = 800
ECCENTRICITY_MIN = 0.5
ECCENTRICITY_SKIP_AREA = 5000
BORDER_MARGIN = 100
PROB_THRESHOLD = 0.5
HYSTERESIS_LOW = 0.35
GROW_RADIUS = 3
SMALL_HOLE_MAX_AREA = 500


def parse_args():
    p = argparse.ArgumentParser(description="Apply trained pixel crack classifier to a TXM TIFF.")
    p.add_argument("input", help="Path to input raw TIFF image.")
    p.add_argument("--model", required=True, help="Path to joblib-serialized classifier.")
    p.add_argument("--out-dir", required=True, help="Directory to write outputs to.")
    return p.parse_args()


def predict_probability_map(model, img01):
    feats = compute_feature_stack(img01)
    h, w, c = feats.shape
    assert c == N_FEATURES, f"expected {N_FEATURES} features, got {c}"
    flat = feats.reshape(-1, c)
    proba = model.predict_proba(flat)[:, 1]
    return proba.reshape(h, w)


def postprocess_mask(prob_map):
    # a. Gaussian-blur the probability map, then threshold at the normal
    #    confident cutoff (PROB_THRESHOLD) -- NOT hysteresis yet. Hysteresis
    #    is applied later (step c2), restricted to grow only from regions
    #    that have already passed shape validation (step c). Thresholding
    #    the whole image with hysteresis up front was tried first and
    #    caused a real regression: a weak, spatially-connected cluster of
    #    noise pixels (none individually confident) could grow just large
    #    enough via its 0.35-0.5 neighbors to clear MIN_SIZE and pass the
    #    elongation check, becoming a brand new artifact blob that never
    #    existed under a flat threshold -- measured directly: 22 such
    #    blobs, zero overlap with the flat-threshold mask, on one test
    #    image alone. Restricting growth to already-validated regions
    #    means a patch of noise with no confident core to attach to can
    #    never spontaneously qualify, no matter how connected it is to
    #    itself.
    blurred = ndi.gaussian_filter(prob_map, sigma=BLUR_SIGMA)
    mask = blurred >= PROB_THRESHOLD

    # b. Morphological closing + remove small specks.
    mask = closing(mask, footprint=disk(CLOSING_RADIUS))
    # remove_small_objects (skimage >= 0.26) removes objects with area
    # <= max_size, so max_size=MIN_SIZE-1 keeps objects with area >= MIN_SIZE.
    mask = remove_small_objects(mask, max_size=MIN_SIZE - 1)

    # c. Reject ring/donut artifacts and round dust blobs. This MUST run
    #    before hole-filling (step below) -- filling holes first would erase
    #    the euler_number signal a ring artifact is identified by.
    labeled = label(mask)
    keep_mask = np.zeros_like(mask, dtype=bool)
    kept_regions = []
    for region in regionprops(labeled):
        area = region.area
        if area < HOLE_CHECK_MAX_AREA and region.euler_number < 1:
            continue  # has a hole -> ring/donut artifact
        if area <= ECCENTRICITY_SKIP_AREA and region.eccentricity < ECCENTRICITY_MIN:
            continue  # too round -> dust blob
        keep_mask[labeled == region.label] = True
        kept_regions.append(region)

    # c2. NOW grow validated regions into their own fuzzy fringe: admit
    #     moderately-confident pixels (>=HYSTERESIS_LOW) only where they
    #     touch a small dilation of an ALREADY-VALIDATED region. This is
    #     what fixes "small holes in the red" (a low-confidence pixel deep
    #     inside/at the edge of a real region gets pulled in by its
    #     validated neighbor) without the earlier bug -- an isolated patch
    #     of moderate-confidence noise with no validated neighbor within
    #     GROW_RADIUS pixels has an entirely False dilated-seed footprint,
    #     so it can never be admitted no matter how internally connected it is.
    moderate_mask = blurred >= HYSTERESIS_LOW
    grown_seed = ndi.binary_dilation(keep_mask, disk(GROW_RADIUS))
    keep_mask = keep_mask | (moderate_mask & grown_seed)

    # c3. Fill small interior holes left in the KEPT regions -- a real crack
    #     region shouldn't have a speckled interior. Only holes smaller than
    #     SMALL_HOLE_MAX_AREA are filled, so a genuinely large enclosed
    #     island of intact material (which the ring check above already
    #     chose to keep, since it wasn't small enough to look like a ring
    #     artifact) is left alone rather than papered over.
    keep_mask = remove_small_holes(keep_mask, max_size=SMALL_HOLE_MAX_AREA)

    # d. Blank out border margin (Gaussian-filter edge artifacts).
    if BORDER_MARGIN > 0:
        keep_mask[:BORDER_MARGIN, :] = False
        keep_mask[-BORDER_MARGIN:, :] = False
        keep_mask[:, :BORDER_MARGIN] = False
        keep_mask[:, -BORDER_MARGIN:] = False

    return keep_mask


def save_outputs(out_dir, base_name, img01, final_mask):
    os.makedirs(out_dir, exist_ok=True)

    # crack_mask.png: crack = black (0), background = white (255).
    mask_png = np.where(final_mask, 0, 255).astype(np.uint8)
    mask_path = os.path.join(out_dir, f"{base_name}_crack_mask.png")
    Image.fromarray(mask_png, mode="L").save(mask_path)

    # overlay.png: grayscale original with crack pixels in red.
    gray = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    overlay = np.stack([gray, gray, gray], axis=-1)
    overlay[final_mask] = [255, 0, 0]
    overlay_path = os.path.join(out_dir, f"{base_name}_overlay.png")
    Image.fromarray(overlay, mode="RGB").save(overlay_path)

    # stats.csv: one row per kept region, re-labeled on the FINAL mask
    # (after the border blanking step) so stats match what's actually kept.
    labeled_final = label(final_mask)
    stats_path = os.path.join(out_dir, f"{base_name}_stats.csv")
    with open(stats_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "area", "solidity", "eccentricity", "centroid_row", "centroid_col"])
        for region in regionprops(labeled_final):
            writer.writerow([
                region.label,
                region.area,
                f"{region.solidity:.4f}",
                f"{region.eccentricity:.4f}",
                f"{region.centroid[0]:.2f}",
                f"{region.centroid[1]:.2f}",
            ])

    return mask_path, overlay_path, stats_path, labeled_final


def main():
    args = parse_args()

    base_name = os.path.splitext(os.path.basename(args.input))[0]

    raw = tifffile.imread(args.input).astype(np.float64)
    img01 = robust_normalize(raw, 1.0, 99.0)

    model = joblib.load(args.model)

    prob_map = predict_probability_map(model, img01)
    raw_mask = prob_map >= PROB_THRESHOLD  # for reference/debugging only

    final_mask = postprocess_mask(prob_map)

    mask_path, overlay_path, stats_path, labeled_final = save_outputs(
        args.out_dir, base_name, img01, final_mask
    )

    n_regions = labeled_final.max()
    area_fraction = final_mask.mean()

    print(f"Input: {args.input}")
    print(f"Image shape: {img01.shape}")
    print(f"Raw (pre-postprocess) crack pixel fraction: {raw_mask.mean():.4f}")
    print(f"Final crack region count: {n_regions}")
    print(f"Final crack area fraction: {area_fraction:.4f}")
    print(f"Saved: {mask_path}")
    print(f"Saved: {overlay_path}")
    print(f"Saved: {stats_path}")


if __name__ == "__main__":
    main()
