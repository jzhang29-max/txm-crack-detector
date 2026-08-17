"""
On-demand, cached TXM preprocessing. Hand it any raw TXM .tif and it returns
the destitched and/or flatfielded version, computing it the first time and
reusing it afterwards.

This exists so the pipeline no longer depends on a separate repository and a
manual batch step. Previously `destitch.py` and `flatfield.py` lived in
~/Desktop/TXM_Destitch_Pipeline, had to be run by hand over a folder, and their
output was read from ~/Desktop/'TXM DATA processed'/{destitched,flatfielded}/.
A new image meant remembering to re-run two scripts before anything worked.
Both modules are now vendored into code/ and driven from here.

The chain is the same one the batch scripts implemented, in the same order --
flatfield.py's own default input was `destitched/`, i.e. the existing
"flatfielded" set is destitched THEN flatfielded, which is why it suppresses the
tile grid far better than destitched alone (measured 2.0-4.8x versus 1.1-1.3x):

    raw  ->  destitch  ->  flatfield
             (border artifacts +      (pseudo-flat-field:
              periodic tile banding    divide by an anisotropic
              via 2D FFT notch)        blur, sigma_y=16 sigma_x=22)

Cached as float32 .npy under paint/preproc_cache/. Existing pre-batched TIFFs
are still preferred when present, so nothing recomputes needlessly and results
stay bit-identical to what has already been measured and labelled against.

Usage:
    import txm_preprocess as tp
    img = tp.get(raw_path, "flatfielded")     # or "destitched" / "raw"
    tp.verify(raw_path)                       # check we reproduce the batch output
"""

import hashlib
import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(PROJECT_DIR, "paint", "preproc_cache")

# Pre-batched output from when these steps were run by hand. Preferred when it
# exists: it is what every existing measurement and label was made against, so
# recomputing could shift results underneath them for no benefit.
PREBATCH_ROOT = os.path.expanduser("~/Desktop/TXM DATA processed")

STAGES = ("raw", "destitched", "flatfielded")


def _key(path, stage):
    h = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{stage}_{h}_{os.path.basename(path)}.npy")


def prebatched_path(raw_path, stage):
    """The hand-batched TIFF for this stage, if it is still on disk."""
    if stage == "raw":
        return raw_path
    root = os.path.join(PREBATCH_ROOT, stage)
    if not os.path.isdir(root):
        return None
    if "/TXM DATA/" in raw_path:
        cand = raw_path.replace("/TXM DATA/", f"/TXM DATA processed/{stage}/")
        if os.path.exists(cand):
            return cand
    import glob
    hits = glob.glob(os.path.join(root, "**", os.path.basename(raw_path)), recursive=True)
    return hits[0] if hits else None


def compute(raw_path, stage):
    """Run the chain from raw. Returns float32; no normalisation applied here.

    Normalisation is deliberately left to the caller (robust_normalize) so this
    returns the same quantity the batch scripts wrote to TIFF, which is what
    makes verify() a meaningful check.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    img = tifffile.imread(raw_path).astype(np.float32)
    if stage == "raw":
        return img
    import destitch
    # destitch_image returns (corrected, report), not just the array
    out, _report = destitch.destitch_image(img)
    if stage == "destitched":
        return np.asarray(out, np.float32)
    import flatfield
    ff = flatfield.flatfield(np.asarray(out, np.float32))
    if isinstance(ff, tuple):
        ff = ff[0]
    return np.asarray(ff, np.float32)


def get(raw_path, stage="flatfielded", allow_compute=True, use_prebatch=True):
    """Preprocessed image for `raw_path`, from pre-batched TIFF -> cache -> compute."""
    if stage == "raw":
        return tifffile.imread(raw_path).astype(np.float32)

    if use_prebatch:
        p = prebatched_path(raw_path, stage)
        if p and os.path.exists(p):
            return tifffile.imread(p).astype(np.float32)

    cp = _key(raw_path, stage)
    if os.path.exists(cp):
        return np.load(cp)
    if not allow_compute:
        return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    out = compute(raw_path, stage)
    np.save(cp, out)
    return out


def verify(raw_path, stage="flatfielded", tol=1e-3):
    """Does computing from raw reproduce the hand-batched TIFF?

    Worth checking rather than assuming: if the vendored modules drifted from
    whatever version produced the batch output, every label and metric in this
    project was made against images this code no longer generates.
    """
    p = prebatched_path(raw_path, stage)
    if not p or not os.path.exists(p):
        return dict(stage=stage, status="no prebatched file to compare against")
    ref = tifffile.imread(p).astype(np.float64)
    mine = compute(raw_path, stage).astype(np.float64)
    if ref.shape != mine.shape:
        return dict(stage=stage, status="SHAPE MISMATCH", ref=ref.shape, mine=mine.shape)
    # Compare after matching scale: the batch script may have written 8/16-bit
    # while compute() stays in float, so an affine rescale is expected and fine.
    rs = (ref - ref.mean()) / (ref.std() or 1)
    ms = (mine - mine.mean()) / (mine.std() or 1)
    d = np.abs(rs - ms)
    return dict(stage=stage, status="ok", corr=float(np.corrcoef(rs.ravel(), ms.ravel())[0, 1]),
                max_abs_z_diff=float(d.max()), mean_abs_z_diff=float(d.mean()),
                matches=bool(d.mean() < tol * 10))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", type=int, default=3, help="verify N images against the batch output")
    ap.add_argument("--stage", default="flatfielded", choices=STAGES)
    args = ap.parse_args()

    import paint_common as pc
    infos = pc.list_images()[:max(args.verify, 0)]
    print(f"verifying {len(infos)} images, stage={args.stage}\n")
    for i in infos:
        r = verify(pc._find_path(i["name"]), args.stage)
        extra = ""
        if r.get("status") == "ok":
            extra = (f"corr {r['corr']:.6f}  mean|dz| {r['mean_abs_z_diff']:.2e}  "
                     f"{'MATCH' if r['matches'] else '*** DIFFERS ***'}")
        print(f"  {i['name'][:44]:46s} {r['status']:12s} {extra}")
