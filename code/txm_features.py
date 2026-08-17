"""
Shared multi-scale pixel-feature extraction for TXM crack pixel classification.

Why a pixel classifier at all, instead of the region-candidate classifier
the SEM tool (detect_cracks.py) uses: on the TXM dataset, the region
classifier's ceiling is set by its *initial* Otsu-based darkness mask, which
in practice only covers ~1.1-1.3% of the image while the real crack extent
(per existing Ilastik ground truth) is 18-31%. Whether or not a candidate
region gets accepted or rejected downstream can't fix an initial mask that
never covered most of the true crack area in the first place -- the fix has
to happen at the pixel level. This mirrors what the SEM project's own
"interior active learning" system found (see
../../CBS_Crack_Detection_Pipeline/interior_active_learning/README.md): a
background-flattened brightness threshold structurally can't see the wider
graded "damaged zone" around a crack's darkest core.

Feature design (17 features/pixel), informed by what worked/failed on this
exact dataset earlier in this project:
- raw normalized intensity
- Gaussian-smoothed intensity at sigma = 2,4,8,16,32,64 px: this is the
  single most important feature group. A large-radius smoothed intensity
  directly encodes "is this pixel embedded in a broad dark region," which
  is what let a simple 1D intensity-population GMM correctly fill the
  dominant crack region earlier -- a learned classifier with this as a
  feature should recover that behavior AND learn better cutoffs than one
  hand-picked global threshold, for both the big dominant crack and
  smaller/fainter secondary ones (which need a smaller-radius version of
  the same idea).
- Gaussian gradient magnitude at sigma = 1,2,4,8: boundary/edge strength.
- Gaussian Laplacian (signed) at sigma = 1,2,4,8: ridge/line response,
  cheaper than a full Frangi filter, useful for thin secondary cracks.
- Local standard deviation (texture roughness) at sigma = 2,8: helps
  separate smooth void/crack interior from textured bulk material.
"""

import numpy as np
from scipy import ndimage as ndi

SMOOTH_SIGMAS = [2, 4, 8, 16, 32, 64]
GRADIENT_SIGMAS = [1, 2, 4, 8]
LAPLACIAN_SIGMAS = [1, 2, 4, 8]
TEXTURE_SIGMAS = [2, 8]

FEATURE_NAMES = (
    ["intensity"]
    + [f"smooth_s{s}" for s in SMOOTH_SIGMAS]
    + [f"gradmag_s{s}" for s in GRADIENT_SIGMAS]
    + [f"laplacian_s{s}" for s in LAPLACIAN_SIGMAS]
    + [f"texture_s{s}" for s in TEXTURE_SIGMAS]
)
N_FEATURES = len(FEATURE_NAMES)


def robust_normalize(img, plo=1.0, phi=99.0):
    lo, hi = np.percentile(img, [plo, phi])
    if hi <= lo:
        hi, lo = float(img.max()), float(img.min())
    return np.clip((img.astype(np.float64) - lo) / max(hi - lo, 1e-8), 0.0, 1.0).astype(np.float32)


def local_std(img, sigma):
    """Local standard deviation via the sum-of-squares trick -- one pair of
    separable Gaussian blurs instead of a generic (slow) footprint filter."""
    mean = ndi.gaussian_filter(img, sigma=sigma)
    mean_sq = ndi.gaussian_filter(img.astype(np.float64) ** 2, sigma=sigma)
    var = np.clip(mean_sq - mean.astype(np.float64) ** 2, 0, None)
    return np.sqrt(var).astype(np.float32)


def compute_feature_stack(img01):
    """img01: float32 array in [0,1]. Returns (H, W, N_FEATURES) float32."""
    h, w = img01.shape
    feats = np.empty((h, w, N_FEATURES), dtype=np.float32)
    idx = 0

    feats[..., idx] = img01
    idx += 1

    for s in SMOOTH_SIGMAS:
        feats[..., idx] = ndi.gaussian_filter(img01, sigma=s)
        idx += 1

    for s in GRADIENT_SIGMAS:
        feats[..., idx] = ndi.gaussian_gradient_magnitude(img01, sigma=s)
        idx += 1

    for s in LAPLACIAN_SIGMAS:
        feats[..., idx] = ndi.gaussian_laplace(img01, sigma=s)
        idx += 1

    for s in TEXTURE_SIGMAS:
        feats[..., idx] = local_std(img01, sigma=s)
        idx += 1

    assert idx == N_FEATURES
    return feats
