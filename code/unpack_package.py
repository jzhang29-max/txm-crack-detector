"""
First-run unpack for a distributed checkout. Idempotent; run_app.sh calls it
every start and it does nothing once expanded.

Why the repo ships compressed instead of ready-to-use: a git repo has to stay
small, and the arrays this project needs do not.

  paint/corrections/*.npy   1.0 GB raw  ->  ~3 MB compressed  (346x; they are
                            uint8 and overwhelmingly zero)
  dataset_cache/*_gt.npy    boolean masks, similar story
  dataset_cache/*_features.npy   2.1 GB, and NOT shipped at all -- the 17-feature
                            stack is a pure function of the image, so it is
                            recomputed here rather than stored. LARGE_343_75's
                            alone is 1.5 GB.

So the package carries dataset_cache/packed.npz plus corrections.npz, and this
expands them into the plain .npy layout every other module expects. Nothing else
in the codebase needs to know packaging happened.

Usage:
    python3 code/unpack_package.py            # expand + build missing features
    python3 code/unpack_package.py --check    # report what is present, change nothing
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

GT_CACHE = os.path.join(PROJECT, "dataset_cache")
CORR_DIR = os.path.join(PROJECT, "paint", "corrections")
PACKED_MASKS = os.path.join(GT_CACHE, "masks.npz")
PACKED_GT = os.path.join(GT_CACHE, "packed.npz")   # older packages
PACKED_CORR = os.path.join(CORR_DIR, "corrections.npz")


def expand_gt():
    """Ground-truth masks from masks.npz, and images from the 16-bit PNGs.

    Also handles the older packed.npz layout so an existing checkout still works.
    """
    n = 0
    for packed in (PACKED_MASKS, PACKED_GT):
        if not os.path.exists(packed):
            continue
        z = np.load(packed)
        for key in z.files:
            out = os.path.join(GT_CACHE, f"{key}.npy")
            if os.path.exists(out):
                continue
            np.save(out, z[key])
            n += 1

    # Images ship as uint16 PNG; convert back to the float32 [0,1] arrays the
    # rest of the code expects. Exactly inverts the packaging step.
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
    except Exception:
        Image = None
    if Image is not None:
        for p in sorted(glob.glob(os.path.join(GT_CACHE, "*_img.png"))):
            stem = os.path.basename(p)[:-len("_img.png")]
            out = os.path.join(GT_CACHE, f"{stem}_img.npy")
            if os.path.exists(out):
                continue
            np.save(out, (np.asarray(Image.open(p), np.float32) / 65535.0))
            n += 1
    return n


def expand_corrections():
    if not os.path.exists(PACKED_CORR):
        return 0
    z = np.load(PACKED_CORR)
    os.makedirs(CORR_DIR, exist_ok=True)
    n = 0
    for key in z.files:
        out = os.path.join(CORR_DIR, f"{key}_correction.npy")
        if os.path.exists(out):
            continue
        np.save(out, z[key])
        n += 1
    return n


def build_features():
    """Recompute the 17-feature stacks that were deliberately not shipped."""
    from txm_features import compute_feature_stack
    built = []
    for img_path in sorted(glob.glob(os.path.join(GT_CACHE, "*_img.npy"))):
        stem = os.path.basename(img_path)[: -len("_img.npy")]
        out = os.path.join(GT_CACHE, f"{stem}_features.npy")
        if os.path.exists(out):
            continue
        img = np.load(img_path)
        print(f"    computing 17 features for {stem} ({img.size/1e6:.1f} MP, "
              f"~{img.size*17*4/1e9:.1f} GB) ...", flush=True)
        np.save(out, compute_feature_stack(img).astype(np.float32))
        built.append(stem)
        del img
    return built


def status():
    gt = len(glob.glob(os.path.join(GT_CACHE, "*_gt.npy")))
    feat = len(glob.glob(os.path.join(GT_CACHE, "*_features.npy")))
    corr = len(glob.glob(os.path.join(CORR_DIR, "*_correction.npy")))
    return dict(ground_truth_masks=gt, feature_stacks=feat, correction_masks=corr,
                packed_gt=os.path.exists(PACKED_GT), packed_corr=os.path.exists(PACKED_CORR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--skip-features", action="store_true",
                    help="expand arrays but do not build the 17-feature stacks "
                         "(they are only needed to validate a retrain)")
    args = ap.parse_args()

    if args.check:
        for k, v in status().items():
            print(f"  {k}: {v}")
        return

    a = expand_gt()
    b = expand_corrections()
    if a or b:
        print(f"==> unpacked {a} ground-truth array(s), {b} correction mask(s)")
    if not args.skip_features:
        built = build_features()
        if built:
            print(f"==> built feature stacks for {len(built)} image(s)")
    s = status()
    if s["ground_truth_masks"] and not s["feature_stacks"]:
        print("==> NOTE: no feature stacks yet. Retrain validation needs them; "
              "rerun without --skip-features to build them.")


if __name__ == "__main__":
    main()
