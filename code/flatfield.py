#!/usr/bin/env python3
"""
flatfield.py — pseudo-flat-field correction: divide each image by a blurred
copy of itself.

    corrected = original / blur(original)

A high-pass filter expressed as a ratio. Structure much LARGER than the blur
sigma appears in both the image and the blur, so it divides out; structure
much SMALLER survives. This removes the mosaic tile banding (pitch ~420-1260px,
far larger than sigma) along with the macro brightness gradient. That also
means it discards absolute brightness and flattens soft structure above the
sigma scale — good for morphology/crack work, wrong for quantitative
absorbance.

Three things here are not obvious and were each found by measurement:

1. ANISOTROPIC SIGMA. A Gaussian attenuates a ripple of period P by
   exp(-2*pi^2*sigma^2/P^2). The row (horizontal-band) tile pitch in this
   dataset is shorter than the column pitch, so one isotropic sigma leaves
   far more residual on the row axis — measured 2.6x worse (0.042 vs 0.019).
   sigma_y=16 / sigma_x=22. Both were initially set far too high (35/50)
   because the crack-preservation constraint was being measured against a
   PER-IMAGE display range; once previews moved to a common window (point 3)
   that constraint turned out to be mostly an artifact, and both sigmas could
   drop a lot. Worst-case residual fell ~3x on both axes (col 3.8%->1.3%,
   row 3.9%->1.3%) with the crack still plainly visible — it is deep enough
   to clip to black regardless, which the numeric "depth / display range"
   proxy had understated.

2. NORMALIZED CONVOLUTION for the background estimate. These mosaics have
   genuine no-data regions outside the ragged tile outline. A plain Gaussian
   near that boundary averages in the zero-valued blanks, dragging the
   background estimate down, so the ratio blows up — measured up to 15.5x,
   in a ring around the outline. Blurring image and mask separately and
   dividing (num/den below) estimates the background from valid pixels only
   and removes the blow-up at its source rather than masking the symptom.

3. A COMMON DISPLAY WINDOW across all images. This is what makes different
   images look like they have different contrast when they don't: after
   flat-fielding every image centers on ~1.001 with an IQR of 0.019-0.036 —
   i.e. genuinely consistent — but a PER-IMAGE percentile stretch maps each
   of those slightly different ranges onto the same 0-255, manufacturing an
   apparent contrast difference. The driver below pools all images, derives
   one window, and renders every preview with it, so they are directly
   comparable. Pass --per-image to get the old (inconsistent) behaviour.
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter


def flatfield(img, sigma_y=16.0, sigma_x=22.0, blank_frac=0.05):
    """Divide by a blurred background estimated only from valid (non-blank) pixels."""
    img = img.astype(np.float64)
    ref = np.median(img[img > 0]) if np.any(img > 0) else 1.0
    valid = (img >= max(blank_frac * ref, 1e-6)).astype(np.float64)

    num = gaussian_filter(img * valid, sigma=(sigma_y, sigma_x), mode="nearest")
    den = gaussian_filter(valid, sigma=(sigma_y, sigma_x), mode="nearest")
    blur = num / np.clip(den, 1e-3, None)

    out = img / np.clip(blur, max(1e-3 * ref, 1e-9), None)
    out[valid == 0] = 0.0
    return out


def to8(a, lo, hi):
    return (np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype("uint8")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    p.add_argument("--input", default=str(here / "destitched"),
                   help="stage-1 output to correct (default: <repo>/destitched)")
    p.add_argument("--output", default=str(here / "flatfielded"),
                   help="where to write corrected .tif files (default: <repo>/flatfielded)")
    p.add_argument("--preview", default=str(here / "flatfielded" / "preview"),
                   help="where to write before/after PNGs (default: <repo>/flatfielded/preview)")
    p.add_argument("--sigma-y", type=float, default=16.0,
                   help="blur sigma down rows — controls horizontal banding (default: 16)")
    p.add_argument("--sigma-x", type=float, default=22.0,
                   help="blur sigma across columns — controls vertical banding (default: 22)")
    p.add_argument("--pattern", default="*.tif*",
                   help="glob for input files (default: *.tif* — matches .tif and .tiff)")
    p.add_argument("--lo-pct", type=float, default=1.0)
    p.add_argument("--hi-pct", type=float, default=99.0)
    p.add_argument("--per-image", action="store_true",
                   help="stretch each preview independently (inconsistent between images)")
    args = p.parse_args()

    out_dir, prev_dir = Path(args.output), Path(args.preview)
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    files = sorted(Path(args.input).glob(args.pattern))
    rng = np.random.default_rng(0)

    corrected, sample = {}, []
    for f in files:
        img = tifffile.imread(str(f)).astype(np.float64)
        out = flatfield(img, args.sigma_y, args.sigma_x)
        corrected[f] = (img, out)
        tifffile.imwrite(str(out_dir / f.name), out.astype(np.float32))
        v = out[out != 0]
        sample.append(rng.choice(v, size=min(400_000, v.size), replace=False))

    glo, ghi = np.percentile(np.concatenate(sample), [args.lo_pct, args.hi_pct])
    print(f"common display window = [{glo:.3f}, {ghi:.3f}]"
          f"{'  (ignored: --per-image)' if args.per_image else ''}\n")

    for f, (img, out) in corrected.items():
        live = out != 0
        if args.per_image:
            lo, hi = np.percentile(out[live], [args.lo_pct, args.hi_pct])
        else:
            lo, hi = glo, ghi
        olo, ohi = np.percentile(img[img > 0], [args.lo_pct, args.hi_pct])
        before8, after8 = to8(img, olo, ohi), to8(out, lo, hi)
        gap = np.full((before8.shape[0], 8), 255, dtype="uint8")
        comb = np.concatenate([before8, gap, after8], axis=1)
        tw = 1400
        s = tw / comb.shape[1]
        Image.fromarray(comb).resize((tw, max(1, int(comb.shape[0] * s))),
                                      Image.LANCZOS).save(prev_dir / (f.stem + ".png"))
        v = out[live]
        clipped = float(((v < lo) | (v > hi)).mean())
        print(f"OK  {f.name[:46]:46s} median={np.median(v):.3f} "
              f"IQR={np.subtract(*np.percentile(v,[75,25])):.4f} clipped={clipped*100:.1f}%")


if __name__ == "__main__":
    main()
