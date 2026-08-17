"""
Import the research-phase correction masks into the app, so retraining can use them.

    python3 code/import_research_corrections.py --dry-run
    python3 code/import_research_corrections.py

paint/corrections/*_correction.npy holds the labels made during the research phase --
8.5 M force-crack and 264 M force-not-crack pixels spanning all four specimen groups.
The app never saw them: it reads app_data/images/<id>/correction.npy, and a freshly
uploaded image starts empty. So the labels that exist for B3, Wrought and AM sat on
disk while the model's only negatives came from the four shipped ground-truth images,
all of which are B2.

Matching is by original filename stem, which is exactly how the app derives an image id,
so a mask can only ever land on the image it was drawn on. Shapes are checked before
anything is written -- a mismatch means the arrays are not the same picture and the file
is skipped rather than resized.

By default an image that ALREADY has corrections in the app is left alone: work done in
the app is newer than the research archive and must not be overwritten by it. --force
overrides that, and --only-empty is the default for the same reason.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
sys.path.insert(0, HERE)

import store as S            # noqa: E402

CORR_DIR = os.path.join(PROJECT, "paint", "corrections")
SUFFIX = "_correction.npy"


def group_of(name):
    n = name.lower()
    if "_b2_" in n or "b2_" in n and "260618" in n:
        return "B2"
    if "_b3_" in n:
        return "B3"
    if "wrought" in n:
        return "Wrought"
    if "hc_316l" in n:
        return "AM/HC"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite app corrections that already exist (default: skip them)")
    ap.add_argument("--with-positives", action="store_true",
                    help="also import force-crack labels (default: negatives only -- see below)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(CORR_DIR, f"*{SUFFIX}")))
    if not files:
        print(f"no correction masks under {CORR_DIR}")
        return 1

    # index the app's images by original filename stem
    by_stem = {}
    for m in S.list_images():
        stem = os.path.splitext(m.get("filename", ""))[0]
        if stem:
            by_stem[stem] = m
    print(f"{len(files)} research mask(s), {len(by_stem)} image(s) in the app")

    plan, skipped = [], []
    for f in files:
        stem = os.path.basename(f)[: -len(SUFFIX)]
        m = by_stem.get(stem)
        if m is None:
            skipped.append((stem, "not loaded in the app"))
            continue
        a = np.load(f, mmap_mode="r")
        h, w = int(m.get("height") or 0), int(m.get("width") or 0)
        if (h, w) != tuple(a.shape):
            skipped.append((stem, f"shape {a.shape} != app image {(h, w)}"))
            continue
        have = (m.get("corrected_crack_px") or 0) + (m.get("corrected_not_px") or 0)
        if have and not args.force:
            skipped.append((stem, f"app already has {have:,} corrected px -- keeping it"))
            continue
        arr = np.asarray(a)
        c, n = int((arr == 1).sum()), int((arr == 2).sum())
        if not args.with_positives:
            c = 0                                  # positives are dropped on import
        if c + n == 0:
            skipped.append((stem, "no usable labels" if not args.with_positives
                                  else "mask is empty"))
            continue
        plan.append((m["id"], stem, f, c, n))
        del arr

    tot_c = sum(p[3] for p in plan)
    tot_n = sum(p[4] for p in plan)
    print(f"\nwould import {len(plan)} mask(s): {tot_c:,} force-crack + {tot_n:,} force-not-crack")
    groups = {}
    for _, stem, _, c, n in plan:
        g = group_of(stem)
        d = groups.setdefault(g, [0, 0, 0])
        d[0] += 1; d[1] += c; d[2] += n
    for g in sorted(groups):
        k, c, n = groups[g]
        print(f"    {g:<8} {k:>3} image(s)  {c:>10,} crack  {n:>12,} not-crack")
    if skipped:
        print(f"\n  skipping {len(skipped)}:")
        for stem, why in skipped[:12]:
            print(f"    {stem[:52]}  --  {why}")
        if len(skipped) > 12:
            print(f"    ... and {len(skipped)-12} more")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return 0

    for i, (iid, stem, f, c, n) in enumerate(plan, 1):
        arr = np.load(f).astype(np.uint8)
        if not args.with_positives:
            # Import the NEGATIVES only, by default and on purpose.
            #
            # HANDOFF.md's label inventory rates the not-crack sources HIGH: 80.3 M px of
            # off-specimen geometric exclusion ("imaging geometry, not a morphology
            # judgment") and 91.6 M px across six specimens the owner confirmed
            # crack-free. The positive labels are the category with a documented history
            # of being wrong -- one batch of 4.9 M algorithmic crack pixels was adopted
            # and then reverted, and the file says plainly: "do not generate positive
            # crack labels algorithmically. Two attempts produced confidently-wrong
            # labels. Positive labelling needs the owner's strokes in the paint tool."
            #
            # The archive does not record which force-crack pixels came from which batch,
            # and 7.0 M of them sit on images the inventory does not list as hand-drawn.
            # Rather than guess, positives stay out: they are the owner's to draw, and a
            # wrong positive actively teaches the wrong thing. --with-positives overrides.
            arr[arr == 1] = 0
        with S.image_lock(iid):
            S.save_npy(iid, "correction.npy", arr)
        # Drop the rendered overlay so the cyan not-crack regions show up immediately.
        ov = S.path(iid, "overlays")
        if os.path.isdir(ov):
            for g in os.listdir(ov):
                try:
                    os.remove(os.path.join(ov, g))
                except OSError:
                    pass
        for g in os.listdir(S.path(iid)):
            if g.startswith("thumb_"):
                try:
                    os.remove(S.path(iid, g))
                except OSError:
                    pass
        print(f"  [{i}/{len(plan)}] {stem[:56]}  {c:,} crack / {n:,} not")
        del arr

    items = [m for m in S.list_images()
             if m.get("corrected_crack_px") or m.get("corrected_not_px")]
    print(f"\nretrain will now draw on {len(items)} labelled image(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
