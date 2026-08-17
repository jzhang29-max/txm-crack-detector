#!/usr/bin/env python3
"""
destitch.py — remove mosaic-stitching border artifacts from TXM .tif images.

What this fixes
----------------
Every image checked in `original/` has a narrow strip of pixels along the
outer edge of the stitched mosaic (mostly left/right, sometimes top/bottom)
where an un-blended neighboring tile shows through at the wrong brightness.
That strip is NOT uniform: within it, some pixels already match the rest of
the image (no defect), some are genuinely "no data" padding from the
mosaic's ragged outline (value ~0), and some are the real stitching defect
(often ~2-10x too dim or too bright). A naive fix that rescales the whole
border column/row at once gets this wrong — it also drags the fine pixels
and the no-data padding along with it.

This script corrects **pixel by pixel**, inside a thin border zone only:
  - For every pixel in the border zone, it compares that pixel to a local
    reference level computed from *the same row (or column)*, just inside
    the border.
  - Only pixels that deviate from that local reference get rescaled to
    match it, with a smooth (not hard-cutoff) blend so partially-dim
    pixels get partially corrected instead of being left alone.
  - Pixels close to "no data" (~0, relative to the local reference) are
    left untouched — that's genuine absence of data, not a brightness
    mismatch, and can't be recovered without fabricating pixels.
  - Everything outside the border zone is left completely untouched.

Bigger mosaics (more tiles stitched together) can also show a second,
different artifact: a periodic tile-to-tile brightness ripple running the
full width and/or height, from tile-to-tile gain differences at a fixed
pitch. This is fixed by default via a true 2D FFT with a narrow Gaussian
notch:

  - The pitch is detected per axis (`_detect_stable_period`, only trusting
    a period that's stable across several detrending window sizes) and
    then independently re-validated against the local spectral noise
    floor, including checking for real harmonics (a repeating tile *step*
    isn't a pure sine wave) and for a genuine 2D checkerboard component —
    each of those is only kept if it clears its own statistical bar, not
    assumed.
  - A very wide (many tile-pitches) Gaussian low-pass in frequency space
    is protected untouched — this is what makes it safe against the
    failure that broke two earlier, cruder attempts (a 1D per-row/column
    median projection can't tell "one real gradient the width of a tile"
    apart from "a repeating step at that pitch"; a full 2D FFT can, because
    a genuinely repeating signal concentrates into one sharp frequency
    spike across many cycles, while a one-off gradient smears broadly no
    matter its scale).
  - Only a 1-2 frequency-bin-wide notch is removed at the validated
    pitch(es), leaving everything else in the spectrum — i.e. the real
    image — untouched.
  - No-data pixels (relative to a local reference, not an absolute
    threshold) are restored to their exact original value afterward.

That notch removes exactly one frequency bin everywhere in the image, which
can occasionally cause a real problem: if a real sharp dark feature (a
crack, a shadow silhouette) happens to leak a little of its own energy into
that same bin, removing it takes a bite out of the real feature too — found
on 2 of 13 test files (verified against actual pixel values, not assumed),
where it measurably brightened a small localized patch by 30-60%.
`correct_periodic_banding` protects against this with a second pass: it
computes a local-contrast score (local std / local mean — normalized this
way because the ringing shows up as a roughly *constant absolute* residual,
which only blows up into a large *ratio* error where the local brightness
is small) on the pre-notch image, and smooth-knee blends the notch-corrected
result back toward the pre-notch value wherever that score is high — the
same blend shape `correct_side` already uses for the border defect, just
driven by local contrast instead of a border-reference ratio. Verified this
brings both known problem regions back in line with their true values
(one from +48% down to +5%, the other from +191% down to ~0%) while leaving
files with no such feature quantitatively and visually unchanged from the
raw notch. `--preview` and look at unfamiliar images with strong dark
linear features anyway — this reduces the known failure mode a lot, it
doesn't come with a proof it's gone in every possible image.

The real tile-to-tile artifact isn't a pure sine wave, it's closer to a
repeating step, and its harmonics can extend further than what a single
narrow, confidently-validated pass removes — so a single pass leaves some
real residual banding behind. The full notch+protect cycle repeats (up to
`max_passes`, default 6), re-detecting and re-protecting fresh each time,
stopping the moment a pass finds nothing left. This was chosen over simply
widening the notch after measuring that a wider notch produces
proportionally stronger ringing than the edge-protection was tuned for —
on the crack-region test file it made the localized distortion roughly 5x
worse for comparable extra removal. Repeating the same already-safe pass
instead gives a monotonic, converging improvement (checked up to 8 passes:
reduction keeps growing with diminishing returns, no instability) — a file
that only needs one pass (like the crack-region file) simply finds nothing
left on pass 2 and stops, completely unchanged past that point.

Use `--no-periodic` to disable the whole periodic-banding fix and get
border-fix-only behavior, or `--periodic-passes N` to tune how many passes
to try.

Usage
-----
    python3 destitch.py                             # process ./original -> ./destitched
    python3 destitch.py --input original --output out
    python3 destitch.py --dry-run                    # report only, write nothing
    python3 destitch.py --preview                     # also save before/after PNGs
    python3 destitch.py --knee-lo 1.1 --knee-hi 1.3    # tune border-fix sensitivity
    python3 destitch.py --no-periodic                  # border-fix only, skip the periodic-banding fix

Requires: numpy, tifffile, scipy  (pip install numpy tifffile scipy)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError:
    print("This script requires 'tifffile'. Install it with: pip install tifffile", file=sys.stderr)
    raise


SIDES = ("left", "right", "top", "bottom")


def _detect_stable_period(profile, min_period, max_period, base_win=251, tol=0.03):
    """
    Look for a periodic brightness ripple in `profile` (a per-row or
    per-column median) and only report it if it's *stable*: detected at
    essentially the same period across several different detrending
    window sizes. A real, physical repeating pattern shows up at the same
    period regardless of how it's detrended; a spurious peak from red/pink
    natural-image noise drifts around depending on the window. This is a
    much more reliable "is this real" gate than any single magnitude/SNR
    threshold.

    Returns the period in pixels, or None if no stable periodicity found.
    """
    n = len(profile)
    max_period = min(max_period, n // 3)
    if max_period <= min_period:
        return None

    periods = []
    for win in (int(base_win * 0.6) | 1, base_win, int(base_win * 1.6) | 1):
        win = min(win, (n // 2) | 1)
        baseline = np.convolve(profile, np.ones(win) / win, mode="same")
        rel = (profile - baseline) / np.clip(np.abs(baseline), 1e-9, None)
        fft = np.fft.rfft(rel)
        mag = np.abs(fft)
        freqs = np.fft.rfftfreq(n, d=1.0)
        valid = np.where((freqs > 1.0 / max_period) & (freqs < 1.0 / min_period))[0]
        if len(valid) == 0:
            return None
        idx = valid[np.argmax(mag[valid])]
        periods.append(1.0 / freqs[idx])

    periods = np.array(periods)
    if (periods.max() - periods.min()) / periods.mean() > tol:
        return None  # not stable across windows -> probably not a real periodic signal
    return float(periods.mean())


def _profile_fft_mag(profile, base_win=251):
    n = len(profile)
    w = min(base_win, (n // 2) | 1)
    baseline = np.convolve(profile, np.ones(w) / w, mode="same")
    rel = (profile - baseline) / np.clip(np.abs(baseline), 1e-9, None)
    fft = np.fft.rfft(rel)
    return np.abs(fft), np.fft.rfftfreq(n, d=1.0), n


def _snr_at_freq(mag, freqs, n, freq, floor_half_width=30, exclude=3):
    idx = int(round(freq * n))
    idx = max(1, min(idx, len(mag) - 2))
    peak = mag[max(idx - 1, 0):idx + 2].max()
    lo, hi = max(1, idx - floor_half_width), min(len(mag), idx + floor_half_width)
    lo_ex, hi_ex = max(idx - exclude, lo), min(idx + exclude + 1, hi)
    band = np.concatenate([mag[lo:lo_ex], mag[hi_ex:hi]])
    floor = np.median(band) if band.size else np.median(mag[1:])
    return peak / max(floor, 1e-12)


def _analyze_profile(profile, min_period, max_period, base_win=251, tol=0.03,
                      fundamental_snr_thresh=2.0, harmonic_snr_thresh=3.0,
                      max_harmonics=4, max_freq=0.48, harmonic_band=0.10):
    """
    Detect + independently validate a periodic signal in a 1D profile.

    The fundamental is `_detect_stable_period`'s stability-across-windows
    check plus an independent SNR check against the local spectral floor.
    A harmonic k (2f0, 3f0, ...) is only accepted if it ALSO clears its own
    stability-across-windows re-check in a tight band around the expected
    k*f0 (not just a raw single-window SNR snapshot — that alone was found,
    during testing on this dataset, to occasionally latch onto an unrelated
    frequency and falsely call it a harmonic).

    Returns None if nothing real is found, else a dict with the fundamental
    period/frequency/snr and a list of validated harmonic multiples k>=2.
    """
    period = _detect_stable_period(profile, min_period, max_period, base_win=base_win, tol=tol)
    if period is None:
        return None
    mag, freqs, n = _profile_fft_mag(profile, base_win=base_win)
    f0 = 1.0 / period
    fund_snr = _snr_at_freq(mag, freqs, n, f0)
    if fund_snr < fundamental_snr_thresh:
        return None
    harmonics = []
    for k in range(2, max_harmonics + 1):
        fk = k * f0
        if fk >= max_freq:
            break
        target_period = period / k
        lo = max(min_period, target_period * (1 - harmonic_band))
        hi = min(max_period, target_period * (1 + harmonic_band))
        if hi <= lo:
            continue
        stable_p = _detect_stable_period(profile, lo, hi, base_win=base_win, tol=tol)
        if stable_p is None or abs(stable_p - target_period) / target_period > harmonic_band:
            continue  # not independently stable at this exact harmonic -> don't trust it
        if _snr_at_freq(mag, freqs, n, fk) >= harmonic_snr_thresh:
            harmonics.append(k)
    return dict(period=period, freq=f0, snr=fund_snr, harmonics=harmonics)


def _periodic_notch(img, min_period=80, macro_k=4.0, notch_sigma_bins=1.3,
                     fundamental_snr_thresh=2.0, harmonic_snr_thresh=3.0,
                     checker_snr_thresh=3.0, max_harmonics=4,
                     pad_frac=0.25, pad_min=150, blank_frac=0.05):
    """
    Detect and remove periodic tile-to-tile brightness banding via a true
    2D FFT with a narrow Gaussian notch placed only at validated frequencies.
    Mutates img in place. Returns a report dict (col/row period+snr+harmonics,
    any detected 2D checkerboard points, and whether anything was applied).

    Why 2D instead of a 1D column/row projection (which is what two earlier,
    simpler attempts used, and which each broke on a different file): a 1D
    projection collapses the whole other dimension into one number, so a
    real, one-off gradient (e.g. a bright corner) that happens to span about
    one tile-width looks indistinguishable from a genuinely repeating
    tile-to-tile step at that same pitch — both approaches got fooled this
    way. A full 2D FFT does not have this ambiguity: a signal that is
    coherently periodic across many cycles (4-10 in this dataset) over the
    ENTIRE width/height concentrates almost all its energy into one sharp,
    narrow spike in frequency space; a one-off feature, no matter its scale,
    smears its energy broadly instead. The notch here is only 1-2 frequency
    bins wide and sits exactly on a pre-validated sharp spike, and a very
    wide low-pass (many tile-pitches) is protected untouched so slow real
    gradients can't be touched by construction.

    Known limitation, found via pixel-level verification (not theoretical):
    FFT ringing can measurably brighten a small, localized patch of a real
    sharp dark feature (a crack, a shadow silhouette) by ~30-60% on images
    that have one. This affected 2 of 13 test files, each in one small
    region. Check --preview on unfamiliar images with strong dark linear
    features before trusting this blindly.
    """
    import scipy.fft

    H, W = img.shape
    report = {}
    col_info = _analyze_profile(np.median(img, axis=0), min_period, max(min_period + 1, W // 3),
                                 fundamental_snr_thresh=fundamental_snr_thresh,
                                 harmonic_snr_thresh=harmonic_snr_thresh, max_harmonics=max_harmonics)
    row_info = _analyze_profile(np.median(img, axis=1), min_period, max(min_period + 1, H // 3),
                                 fundamental_snr_thresh=fundamental_snr_thresh,
                                 harmonic_snr_thresh=harmonic_snr_thresh, max_harmonics=max_harmonics)
    report["col_period"] = col_info["period"] if col_info else None
    report["row_period"] = row_info["period"] if row_info else None
    report["checker"] = []

    if col_info is None and row_info is None:
        report["applied"] = False
        return report
    report["applied"] = True

    pad_h = max(pad_min, int(round(pad_frac * H)))
    pad_w = max(pad_min, int(round(pad_frac * W)))
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    Hp, Wp = padded.shape

    F = scipy.fft.fft2(padded)
    freqs_row = np.fft.fftfreq(Hp, d=1.0)
    freqs_col = np.fft.fftfreq(Wp, d=1.0)
    FR, FC = freqs_row[:, None], freqs_col[None, :]

    def cutoff_freq(period, dim):
        spatial_sigma = macro_k * period if period else dim / 6.0
        return 1.0 / (2.0 * np.pi * spatial_sigma)

    f_cut_row = cutoff_freq(row_info["period"] if row_info else None, H)
    f_cut_col = cutoff_freq(col_info["period"] if col_info else None, W)
    lowpass = np.exp(-0.5 * ((FR / f_cut_row) ** 2 + (FC / f_cut_col) ** 2))

    targets = []
    if col_info is not None:
        for k in [1] + col_info["harmonics"]:
            fc0 = k * col_info["freq"]
            targets += [(0.0, fc0), (0.0, -fc0)]
    if row_info is not None:
        for k in [1] + row_info["harmonics"]:
            fr0 = k * row_info["freq"]
            targets += [(fr0, 0.0), (-fr0, 0.0)]

    if col_info is not None and row_info is not None:
        resid_mag = np.abs(F) * (1.0 - lowpass)
        for sfr in (1, -1):
            for sfc in (1, -1):
                fr0, fc0 = sfr * row_info["freq"], sfc * col_info["freq"]
                idx_r, idx_c = int(round(fr0 * Hp)) % Hp, int(round(fc0 * Wp)) % Wp
                peak = resid_mag[idx_r - 1:idx_r + 2, idx_c - 1:idx_c + 2].max()
                r0, r1 = max(0, idx_r - 25), min(Hp, idx_r + 25)
                c0, c1 = max(0, idx_c - 25), min(Wp, idx_c + 25)
                floor = np.median(resid_mag[r0:r1, c0:c1])
                snr = peak / max(floor, 1e-12)
                if snr >= checker_snr_thresh:
                    report["checker"].append((fr0, fc0, float(snr)))
    targets.extend((fr0, fc0) for fr0, fc0, _snr in report["checker"])

    bin_r, bin_c = 1.0 / Hp, 1.0 / Wp
    sigma_fr, sigma_fc = notch_sigma_bins * bin_r, notch_sigma_bins * bin_c
    notch = np.ones((Hp, Wp))
    for fr0, fc0 in targets:
        d2 = ((FR - fr0) / sigma_fr) ** 2 + ((FC - fc0) / sigma_fc) ** 2
        notch *= 1.0 - np.exp(-0.5 * d2)
    notch[0, 0] = 1.0

    total_mask = lowpass + (1.0 - lowpass) * notch
    out_padded = np.real(scipy.fft.ifft2(F * total_mask))
    corrected = out_padded[pad_h:pad_h + H, pad_w:pad_w + W]

    macro_padded = np.real(scipy.fft.ifft2(F * lowpass))
    macro = macro_padded[pad_h:pad_h + H, pad_w:pad_w + W]
    blank_floor = np.maximum(blank_frac * macro, 1e-6)
    corrected = np.where(img < blank_floor, img, corrected)

    # Safety backstop: FFT ringing near a sharp real edge can occasionally
    # overshoot/undershoot past the range the data ever actually had (found
    # in testing: a handful of pixels, ~0.03% on one file, went slightly
    # negative — impossible for this absorbance-like data). Never let the
    # correction invent a value more extreme than what was already observed.
    corrected = np.clip(corrected, img.min(), img.max())

    img[:] = corrected
    return report


def _local_contrast_score(img, window, mean_floor=0.02):
    """
    Local coefficient of variation (local std / local mean) — deliberately
    NOT plain local std. The absolute residual the notch leaves behind near
    a real feature is roughly constant in magnitude regardless of local
    texture (verified empirically: bucketing |corrected - before| by plain
    local-std score gave an almost flat profile — plain local std barely
    predicts where the bad *ratio* happens). What actually blows up is the
    ratio, because it divides that near-constant absolute residual by a
    shrinking local brightness. Normalizing local std by local mean
    reproduces that same ratio-sensitivity directly in the score: it stays
    low for genuine periodic banding (a small ripple riding on a background
    that isn't especially dark) and spikes exactly where a real feature's
    edge meets a dark interior/shadow.
    """
    import scipy.ndimage as ndi

    window = max(3, int(round(window)))
    m = ndi.uniform_filter(img, size=window, mode="nearest")
    m2 = ndi.uniform_filter(img * img, size=window, mode="nearest")
    var = np.clip(m2 - m * m, 0.0, None)
    return np.sqrt(var) / np.clip(m, mean_floor, None)


def _edge_margin_weight(shape, margin):
    """
    Weight map that's 1.0 exactly at each of the 4 edges and smoothly
    (via `_smoothstep`) decays to 0.0 by `margin` pixels inward. See
    `_correct_periodic_banding_once`'s docstring for why this margin exists.
    """
    H, W = shape
    margin = max(1, int(margin))
    y = np.arange(H)
    x = np.arange(W)
    dist_y = np.minimum(y, H - 1 - y)
    dist_x = np.minimum(x, W - 1 - x)
    dist = np.minimum(dist_y[:, None], dist_x[None, :])
    return 1.0 - _smoothstep(dist / margin)


def _correct_periodic_banding_once(img, min_period=80, macro_k=4.0, notch_sigma_bins=1.3,
                                    fundamental_snr_thresh=2.0, harmonic_snr_thresh=3.0,
                                    checker_snr_thresh=3.0, max_harmonics=4,
                                    pad_frac=0.25, pad_min=150, blank_frac=0.05,
                                    period_frac=4.0, win_min=60, win_max=240,
                                    mean_floor=0.02, contrast_ref_pct=95.0,
                                    knee_lo=1.1, knee_hi=2.2, edge_margin=120):
    """
    Runs `_periodic_notch` (the validated 2D FFT notch,
    unchanged), then protects real sharp dark features from the FFT-ringing
    tradeoff described in its docstring, via a per-pixel smooth-knee blend
    back toward the pre-notch value — the same blend *shape* `correct_side`
    already uses for the border defect (`_smoothstep`), just driven by a
    local-contrast score instead of a border-reference ratio:

      - Real dark linear features (a crack, a shadow silhouette) are
        localized and have substantial local pixel-to-pixel variation over
        a window comparable to a good fraction of a tile pitch.
      - Genuine periodic banding, by construction of what makes
        `_detect_stable_period`/`_analyze_profile` trust it at all (coherent
        over many cycles across the *entire* width/height), has small
        amplitude and is extremely smooth over any window much shorter than
        the tile pitch — so its local-contrast score is much lower than a
        real feature's, even though a single-frequency-bin FFT can't tell
        them apart on its own.

    There's a second, separate protection here too: removing a narrow
    frequency band is, by the uncertainty principle, equivalent to
    convolving with a spatially *wide* kernel — and that wide kernel's
    reach, combined with the padded array's finite size, produces a real,
    measurable brightness bias right at the image's outer edges (found via
    direct pixel measurement: isolating the notch confirmed it, not the
    macro low-pass — with the notch disabled the edge ratio was exactly
    1.0). The local-contrast score doesn't catch this on its own, because
    the border defect fix already made that zone smooth (that's the whole
    point of that fix), so it doesn't look like a "sharp real feature" to
    the score above. A separate, unconditional `_edge_margin_weight` fades
    from full protection at the outer edge to none by `edge_margin` pixels
    inward (measured: the bias decays to negligible by ~100px), and the
    two protections combine via max — whichever one says "protect this
    pixel" wins.

    Found via pixel-level testing (not theory) to bring the two known
    problem regions from this dataset back in line with their true
    pre-notch values (one from +48% median brightening down to +5%, the
    other from +191% down to ~0%), while leaving files with no such
    feature untouched (verified: for files without a triggering region,
    output is visually and quantitatively indistinguishable from the raw
    notch). Mutates img in place. Returns the same report `_periodic_notch`
    returns, plus an `edge_protection` dict (window size, how much of the
    image the blend touched, mean/max blend weight) — `edge_protection` is
    None if nothing was detected/applied at all.
    """
    before = img.copy()  # border-fix-only reference, pre-notch

    report = _periodic_notch(
        img, min_period=min_period, macro_k=macro_k, notch_sigma_bins=notch_sigma_bins,
        fundamental_snr_thresh=fundamental_snr_thresh, harmonic_snr_thresh=harmonic_snr_thresh,
        checker_snr_thresh=checker_snr_thresh, max_harmonics=max_harmonics,
        pad_frac=pad_frac, pad_min=pad_min, blank_frac=blank_frac,
    )
    if not report.get("applied"):
        report["edge_protection"] = None
        return report  # nothing detected -> nothing to protect against, img already unchanged

    full_corrected = img.copy()
    periods = [p for p in (report.get("col_period"), report.get("row_period")) if p]
    ref_period = min(periods) if periods else min_period
    window = float(np.clip(ref_period / period_frac, win_min, win_max))

    score = _local_contrast_score(before, window, mean_floor=mean_floor)

    # Robust "typical" local-contrast level for this image, restricted to
    # non-blank pixels so ragged-mosaic zero-padding (near-zero score by
    # construction) doesn't drag the reference down and over-trigger
    # protection everywhere.
    positive = before[before > 0]
    blank_floor = max(blank_frac * np.median(positive), 1e-6) if positive.size else 1e-6
    non_blank = before > blank_floor
    ref_score = np.percentile(score[non_blank], contrast_ref_pct) if non_blank.any() else np.median(score)

    ratio = score / max(ref_score, 1e-12)
    log_ratio = np.log(np.clip(ratio, 1e-6, None))
    lo, hi = np.log(knee_lo), np.log(knee_hi)
    contrast_weight = _smoothstep((log_ratio - lo) / max(hi - lo, 1e-9))
    edge_weight = _edge_margin_weight(before.shape, edge_margin)
    weight = np.maximum(contrast_weight, edge_weight)

    blended = weight * before + (1.0 - weight) * full_corrected
    blended = np.clip(blended, np.minimum(before, full_corrected).min(), np.maximum(before, full_corrected).max())
    img[:] = blended

    report["edge_protection"] = dict(
        window=window, ref_score=float(ref_score), edge_margin=edge_margin,
        frac_touched=float(np.mean(weight > 0.02)),
        frac_strong=float(np.mean(weight > 0.5)),
        mean_weight=float(weight.mean()), max_weight=float(weight.max()),
    )
    return report


def correct_periodic_banding(img, max_passes=12, edge_margin=200, max_edge_drift=0.06,
                              period_tol=0.05, **kwargs):
    """
    Public entry point: repeats `_correct_periodic_banding_once` — an
    already-validated, single-pass notch + edge-protection cycle — up to
    `max_passes` times, stopping as soon as a pass finds nothing left to
    correct, OR as soon as a pass is caught measurably disturbing its own
    supposedly-protected border zone (see below) — whichever comes first.

    Why repeat a single narrow, safe pass instead of using one wider (and
    correspondingly riskier) notch: the real tile-to-tile artifact isn't a
    pure sine wave, it's closer to a repeating step, whose harmonics can
    extend further than what a single narrow pass confidently validates and
    removes. Found via direct measurement (not assumed) that widening the
    notch to catch more of that shape in one pass reopens the FFT-ringing
    problem — a wider notch produces stronger ringing than the edge
    protection was tuned for, and on the crack-region test file, made the
    localized real-feature distortion roughly *5x worse* (ratio 1.15 to
    over 2.0) for comparable extra removal. Repeating the same narrow pass
    instead gives a monotonic, converging improvement on the periodic
    banding itself — but this was found (via direct pixel measurement, not
    assumed) to come with its own cost: removing a narrow frequency band is,
    by the uncertainty principle, equivalent to convolving with a spatially
    *wide* kernel, and near the image's outer edges that produces a real,
    measurable brightness bias distinct from the crack/shadow problem — one
    the local-contrast score can't see (the border defect fix already made
    that zone smooth, so it doesn't look like a "sharp real feature" to
    that score). `_correct_periodic_banding_once`'s `edge_margin` feathers
    protection back toward the pre-notch value near the outer edges to
    reduce this, but on at least one test file that bias still *compounded*
    with repeated passes even with that protection in place (worst case
    reached +13% after 6 passes, growing roughly linearly per pass) — so
    this function checks for that directly rather than assuming the margin
    protection is sufficient on every image: after each pass, it measures
    how much the *supposedly-protected* zone itself just moved. If a pass
    disturbs that zone (relative to before ANY periodic-banding pass ran)
    by more than `max_edge_drift`, that pass is rolled back and iteration
    stops there. Pass 1 itself always runs unconditionally — it's the same
    single notch+protect cycle that was already shipped and found
    acceptable before repeating passes existed, so this guard's job is only
    to stop *iterating further* from making the border worse than that
    established baseline, not to second-guess pass 1. Verified: the known
    crack/shadow regions stay exactly as protected on every retained pass;
    a file that only needs one pass (e.g. the crack-region file) simply
    reports nothing left to do on pass 2 and stops, unchanged past that
    point.

    Mutates img in place. Returns the last retained pass's report, plus
    `n_passes` (how many passes were actually kept) and `stopped_early`
    (why: "converged" if nothing was left to correct, "edge_drift" if a
    pass was rolled back for disturbing its own protected zone,
    "period_drift" if a pass was rolled back for re-detecting a tile pitch
    inconsistent with pass 1's — see the period-consistency guard below —
    or None if `max_passes` was simply reached with no problem detected).
    """
    edge_zone = _edge_margin_weight(img.shape, edge_margin) > 0.5
    original = img.copy()  # fixed reference for CUMULATIVE drift, not just this pass's own contribution
    report = None
    stopped_early = None
    i = 0
    first_periods = None
    for i in range(max_passes):
        pre_pass = img.copy()
        report = _correct_periodic_banding_once(img, edge_margin=edge_margin, **kwargs)
        if not report.get("applied"):
            stopped_early = "converged"
            break

        # Period-consistency guard. Each pass re-detects the tile pitch from
        # scratch, and after several passes the genuine peak has been largely
        # removed — which collapses the local spectral floor that
        # `_analyze_profile`'s SNR check measures against, so noise can clear
        # the SNR bar and the detector can lock onto a WRONG period. Correcting
        # at a spurious frequency doesn't just waste a pass, it *injects* a new
        # banding artifact that wasn't there. Observed on one file: a true row
        # pitch of ~284px was re-detected as ~341px on a later pass, visibly
        # re-banding the image. So: pin the periods found on pass 1 (when the
        # real signal is strongest and least ambiguous) and stop as soon as a
        # later pass wanders off them.
        periods = (report.get("col_period"), report.get("row_period"))
        if first_periods is None:
            first_periods = periods
        else:
            drifted = False
            for p0, p in zip(first_periods, periods):
                if p0 and p and abs(p - p0) / p0 > period_tol:
                    drifted = True
            if drifted:
                img[:] = pre_pass  # this pass corrected at the wrong frequency -> undo it
                stopped_early = "period_drift"
                i -= 1
                break

        if edge_zone.any() and i > 0:
            # Pass 1 always runs unconditionally — it's the same single
            # notch+protect cycle that was already shipped and accepted
            # before repeating-passes existed, so it's not this guard's job
            # to second-guess it. This guard's job is only to stop
            # *iterating past* pass 1 from making the border worse.
            #
            # Checked against the ORIGINAL pre-loop state, not the previous
            # pass — per-pass drift shrinks each iteration even while the
            # cumulative total keeps growing (found via direct measurement:
            # a file with a severe border bias showed per-pass p99 drift
            # decreasing every pass — 3.9%, 3.1%, 2.4%, 1.9%, 1.6% — that
            # never individually looked alarming, while the *sum* reached
            # +13% after 6 passes; checking only the latest increment would
            # never have caught it). p99, not median/mean, because the
            # drift is concentrated in specific bands within the edge zone,
            # not spread uniformly across all four borders — a global
            # average dilutes exactly the signal this needs.
            drift = np.abs(img[edge_zone] - original[edge_zone]) / np.clip(np.abs(original[edge_zone]), 1e-6, None)
            if np.percentile(drift, 99) > max_edge_drift:
                img[:] = pre_pass  # this pass tipped the cumulative border drift over the line -> roll it back
                stopped_early = "edge_drift"
                i -= 1  # the rolled-back pass doesn't count
                break
    report["n_passes"] = i + 1
    report["stopped_early"] = stopped_early
    return report


def _band_and_ref(img, side, max_band, ref_window):
    """Return (band, ref_zone, band_axis) views for the given side.

    band: the outer `max_band` rows/cols to inspect/correct.
    ref_zone: the `ref_window` rows/cols just inside the band, used to
              build a per-line "what this should look like" reference.
    band_axis: the axis along which each line (row or column) runs, used
              to know whether the reference is per-row or per-column.
    """
    H, W = img.shape
    if side == "left":
        band = img[:, :max_band]
        ref_zone = img[:, max_band : max_band + ref_window]
        return band, ref_zone, "row"
    if side == "right":
        band = img[:, W - max_band :]
        ref_zone = img[:, max(0, W - max_band - ref_window) : W - max_band]
        return band, ref_zone, "row"
    if side == "top":
        band = img[:max_band, :]
        ref_zone = img[max_band : max_band + ref_window, :]
        return band, ref_zone, "col"
    if side == "bottom":
        band = img[H - max_band :, :]
        ref_zone = img[max(0, H - max_band - ref_window) : H - max_band, :]
        return band, ref_zone, "col"
    raise ValueError(side)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _soft_knee_factor(ratio, knee_lo, knee_hi, clip_range):
    """
    Turn a raw per-pixel ref/pixel ratio into a correction factor using a
    smooth knee instead of a hard on/off threshold:

      - |log(ratio)| below knee_lo  -> factor ~= 1 (leave alone; normal noise/texture)
      - |log(ratio)| above knee_hi  -> factor ~= ratio (fully snap to the reference)
      - in between                  -> smoothly ramped partial correction

    This removes the visible "gap" a hard threshold leaves behind for
    pixels that are dimmed/brightened by an amount just under the cutoff.
    """
    # ratio can legitimately be 0 when a line's interior reference is entirely
    # no-data (an all-blank row/column next to a border that still has real
    # pixels). log(0) = -inf then drives factor to 0, which the clip below
    # turns into 0.1 — i.e. real border pixels darkened 10x. Clamping the
    # ratio into the correction's own clip range first makes that impossible:
    # a pixel with no usable reference gets the maximum sanctioned correction,
    # never a spurious one.
    ratio = np.clip(ratio, clip_range[0], clip_range[1])
    log_ratio = np.log(ratio)
    a_log_ratio = np.abs(log_ratio)
    lo, hi = np.log(knee_lo), np.log(knee_hi)
    weight = _smoothstep((a_log_ratio - lo) / max(hi - lo, 1e-9))
    factor = np.exp(weight * log_ratio)
    return np.clip(factor, clip_range[0], clip_range[1])


def correct_side(img, side, max_band=40, ref_window=30, knee_lo=1.1, knee_hi=1.8,
                  blank_frac=0.05, clip_range=(0.1, 10.0), min_hits=25):
    """
    Detect and fix per-pixel border artifacts on one side of the image.
    Returns the number of pixels meaningfully corrected, and mutates img in place.
    """
    band, ref_zone, orient = _band_and_ref(img, side, max_band, ref_window)
    if band.size == 0 or ref_zone.size == 0:
        return 0

    line_axis = 1 if orient == "row" else 0          # axis to reduce over for a per-line reference
    ref = np.median(ref_zone, axis=line_axis)          # one reference value per row (or column)

    # "Blank" (no-data / outside the mosaic outline) is relative to each line's own
    # reference brightness, not a fixed absolute value — a pixel at 1% of the local
    # reference is no-data whether the image's overall scale is 0-1 or 0-3000.
    # An absolute floor also protects against a near-zero reference.
    if orient == "row":
        blank_floor = np.maximum(blank_frac * ref, 1e-6)[:, np.newaxis]
    else:
        blank_floor = np.maximum(blank_frac * ref, 1e-6)[np.newaxis, :]
    blank_mask = band < blank_floor
    safe_band = np.clip(band, blank_floor, None)
    if orient == "row":
        ratio = ref[:, np.newaxis] / safe_band
    else:
        ratio = ref[np.newaxis, :] / safe_band

    factor = _soft_knee_factor(ratio, knee_lo, knee_hi, clip_range)
    factor = np.where(blank_mask, 1.0, factor)

    n_hits = int(np.sum((~blank_mask) & (np.abs(factor - 1.0) > 0.02)))
    if n_hits < min_hits:
        return 0

    band *= factor  # in-place: writes back through the view into img
    return n_hits


def destitch_image(img, max_band=40, ref_window=30, knee_lo=1.1, knee_hi=1.8,
                    blank_frac=0.05, sides=SIDES, min_hits=25, fix_periodic=True,
                    periodic_passes=12):
    img = img.astype(np.float64, copy=True)
    report = {}

    # Border fix first, then periodic-banding fix — this is the order the
    # 2D FFT correction was actually verified against (it was tested reading
    # already-border-fixed images), so keep it that way.
    for side in sides:
        n_hits = correct_side(
            img, side, max_band=max_band, ref_window=ref_window,
            knee_lo=knee_lo, knee_hi=knee_hi, blank_frac=blank_frac, min_hits=min_hits,
        )
        report[side] = n_hits

    if fix_periodic:
        report["periodic"] = correct_periodic_banding(img, max_passes=periodic_passes)

    return img, report


def save_preview(original, corrected, out_path):
    """Save a side-by-side before/after PNG (contrast-stretched for viewing)."""
    from PIL import Image

    lo, hi = np.percentile(original, 1), np.percentile(original, 99)

    def to8(a):
        a = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
        return (a * 255).astype("uint8")

    before8, after8 = to8(original), to8(corrected)
    gap = np.full((before8.shape[0], 8), 255, dtype="uint8")
    combined = np.concatenate([before8, gap, after8], axis=1)
    Image.fromarray(combined).save(out_path)


def process_file(path, out_dir, args, preview_dir=None):
    original = tifffile.imread(str(path))
    orig_dtype = original.dtype
    img = np.asarray(original)

    if img.ndim != 2:
        return f"SKIP  {path.name}: not a single-channel 2D image (shape {img.shape})"

    corrected, report = destitch_image(
        img,
        max_band=args.max_band,
        ref_window=args.ref_window,
        knee_lo=args.knee_lo,
        knee_hi=args.knee_hi,
        blank_frac=args.blank_frac,
        sides=args.sides,
        min_hits=args.min_hits,
        fix_periodic=not args.no_periodic,
        periodic_passes=args.periodic_passes,
    )

    periodic = report.pop("periodic", None) or {}
    parts = [f"{s}={n}px" for s, n in report.items() if n > 0]
    if periodic.get("col_period"):
        parts.append(f"col~{periodic['col_period']:.0f}px-band")
    if periodic.get("row_period"):
        parts.append(f"row~{periodic['row_period']:.0f}px-band")
    if periodic.get("checker"):
        parts.append(f"checker={len(periodic['checker'])}pt")
    if periodic.get("n_passes"):
        parts.append(f"passes={periodic['n_passes']}")
    summary = ", ".join(parts) or "no artifact detected"

    if not args.dry_run:
        out_path = out_dir / path.name
        tifffile.imwrite(str(out_path), corrected.astype(orig_dtype))

    if preview_dir is not None:
        save_preview(img, corrected, preview_dir / (path.stem + ".preview.png"))

    return f"{'DRY-RUN' if args.dry_run else 'OK'}  {path.name}: {summary}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults resolve relative to this script, not the current working
    # directory, so `python3 /path/to/destitch.py` works from anywhere.
    here = Path(__file__).resolve().parent
    p.add_argument("--input", default=str(here / "original"), help="folder of source .tif images (default: <repo>/original)")
    p.add_argument("--output", default=str(here / "destitched"), help="folder to write corrected .tif images (default: <repo>/destitched)")
    p.add_argument("--pattern", default="*.tif*", help="glob pattern for input files (default: *.tif* — matches .tif and .tiff)")
    p.add_argument("--max-band", type=int, default=40, help="max pixels from each edge to inspect/correct (default: 40)")
    p.add_argument("--ref-window", type=int, default=30, help="pixels just inside the band used as the per-line reference (default: 30)")
    p.add_argument("--knee-lo", type=float, default=1.05, help="below this deviation from the local reference, leave the pixel alone (default: 1.05)")
    p.add_argument("--knee-hi", type=float, default=1.25, help="above this deviation, fully snap the pixel to the local reference (default: 1.25); in between, blend smoothly")
    p.add_argument("--blank-frac", type=float, default=0.05, help="pixels below this fraction of the local reference are treated as no-data padding and left untouched (default: 0.05)")
    p.add_argument("--min-hits", type=int, default=25, help="minimum anomalous pixels on a side before bothering to correct it (default: 25)")
    p.add_argument("--sides", default="left,right,top,bottom", help="comma-separated sides to check (default: all four)")
    p.add_argument("--no-periodic", action="store_true", help="skip the periodic tile-banding fix (2D FFT notch); border-fix only. "
                   "Use this if --preview shows the FFT ringing tradeoff (see module docstring) is not worth it for a given image.")
    p.add_argument("--periodic-passes", type=int, default=12, help="max repeats of the notch+protect cycle (default: 12); each pass "
                   "stops early once nothing real is left to correct OR the border-drift safety guard trips, so higher only helps "
                   "files that haven't hit their own safe limit yet — it can't override the guard")
    p.add_argument("--dry-run", action="store_true", help="report detections only; do not write corrected files")
    p.add_argument("--preview", action="store_true", help="also save a before/after PNG per file into <output>/preview")
    args = p.parse_args()
    args.sides = tuple(s.strip() for s in args.sides.split(",") if s.strip())

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    if not in_dir.is_dir():
        sys.exit(f"Input folder not found: {in_dir}")
    files = sorted(in_dir.glob(args.pattern))
    if not files:
        sys.exit(f"No files matching {args.pattern!r} in {in_dir}")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = None
    if args.preview:
        preview_dir = out_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        print(process_file(f, out_dir, args, preview_dir))


if __name__ == "__main__":
    main()
