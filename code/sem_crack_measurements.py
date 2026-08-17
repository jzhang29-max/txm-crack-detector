"""
Crack shape measurements, vendored from the SEM pipeline
(interior_active_learning/code/extended_features.py) so both projects report the
SAME physical quantities for a crack region.

Kept rather than rewritten because the SEM version is better than what this
project had: it reports skeleton length, mean and max width via the medial-axis
radius, tortuosity, branch-point count and boundary roughness -- the numbers a
materials reader actually wants -- where the TXM export previously gave generic
region properties (bounding box, eccentricity). Only crack_shape_measurements()
and its helpers are used here; the ML feature functions in this file are unused
by TXM but left intact so the file stays diffable against the SEM original.

DO NOT edit this to diverge. If a measurement needs changing, change it in both.
"""

# ---- original SEM module docstring follows ----
"""
Shape/topology measurements that depend on NOTHING dataset-specific -- no
absolute brightness scale, no contrast-stretch convention, no material
assumption. Two uses:

1. As candidate-level ML features (BoundaryRoughness, BranchPointDensity) --
   see experiments/benchmark_extended_features.py for the empirical test of
   whether adding these to the unified model actually helps.
2. As descriptive per-crack MEASUREMENTS for the output CSV (crack_measurements.py)
   -- skeleton length, mean/max width, tortuosity, branch-point count -- the
   kind of numbers a materials-science paper reports about detected cracks,
   independent of whether they're used for classification at all.

Kept in one module so both call sites can never define "branch point" or
"roughness" two different ways.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology

# 8-connected neighbor-count kernel for skeleton branch-point detection.
_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def _local_crop(mask_bool, margin=2):
    """Crop a boolean mask to its bounding box (+ a small margin so
    skeletonize/perimeter operations near the crop edge aren't clipped) --
    same performance rationale as every other per-candidate feature function
    in this project: never run a full-frame op per candidate."""
    ys, xs = np.where(mask_bool)
    H, W = mask_bool.shape
    y0, y1 = max(0, ys.min() - margin), min(H, ys.max() + 1 + margin)
    x0, x1 = max(0, xs.min() - margin), min(W, xs.max() + 1 + margin)
    return mask_bool[y0:y1, x0:x1]


def boundary_roughness(mask_bool):
    """Perimeter / convex-hull perimeter. A smooth blob (or a smooth,
    gently-curved crack) has a perimeter close to its convex hull's, so this
    sits near 1.0. A jagged, notched, or branching boundary -- the kind
    fracture surfaces actually have, physically, in most materials -- pushes
    this well above 1.0. Deliberately NOT the same information as Solidity
    (area(region)/area(hull)): a long thin region can be highly solid by
    area while still having a rough, wiggly edge that Solidity barely
    penalizes, because a thin sliver's area difference from its hull is
    small in absolute terms even when the boundary itself is very jagged."""
    local = _local_crop(mask_bool)
    props_list = measure.regionprops(local.astype(np.uint8))
    if not props_list:
        return 1.0
    props = props_list[0]
    perim = props.perimeter if props.perimeter > 0 else 1.0
    convex_img = props.convex_image
    convex_perim = measure.perimeter(convex_img) if convex_img.sum() > 0 else perim
    convex_perim = max(convex_perim, 1.0)
    return float(perim / convex_perim)


def skeleton_stats(mask_bool):
    """Skeletonize the region and return (skeleton_length_px, branch_point_count,
    endpoint_count, skeleton_mask_local, local_crop_mask). Branch points are
    skeleton pixels with >=3 skeleton neighbors (8-connectivity) -- a fork in
    the medial axis, which happens where a crack branches or where two
    candidates' shapes merge into one blob. Endpoints have exactly 1
    neighbor. A simple, unbranched crack skeletonizes to a single path with
    exactly 2 endpoints and 0 branch points; real branching crack networks
    (or two blobs fused into one candidate) show >=1 branch point."""
    local = _local_crop(mask_bool)
    skel = morphology.skeletonize(local)
    skel_len = int(skel.sum())
    if skel_len == 0:
        return 0, 0, 0, skel, local
    neighbor_count = ndi.convolve(skel.astype(int), _NEIGHBOR_KERNEL, mode="constant", cval=0)
    branch_points = int(((neighbor_count >= 3) & skel).sum())
    endpoints = int(((neighbor_count == 1) & skel).sum())
    return skel_len, branch_points, endpoints, skel, local


def branch_point_density(mask_bool):
    """Branch points per 100px of skeleton length -- normalized so a big
    crack and a small crack with the same TOPOLOGY (e.g. both simple,
    unbranched paths) score the same, rather than the raw count scaling
    with size. A purely blob-shaped artifact typically skeletonizes to a
    short stub with 0 branch points (density 0); an interconnected crack
    network scores higher. General topological property -- doesn't depend
    on brightness, contrast, or which material/instrument produced the
    image, so it should transfer to other SEM crack-detection datasets
    better than the brightness-based features do."""
    skel_len, branch_points, _, _, _ = skeleton_stats(mask_bool)
    if skel_len < 3:
        return 0.0
    return float(branch_points / skel_len * 100)


def crack_shape_measurements(mask_bool):
    """Full descriptive measurement set for ONE final crack region (used by
    crack_measurements.py's per-image report, not the ML feature set) --
    physical numbers a materials-science reader actually wants: length,
    width, tortuosity, branching, orientation. Returns a dict; NaN for any
    quantity that isn't well-defined for this region's topology (e.g.
    tortuosity needs exactly 2 skeleton endpoints)."""
    local = _local_crop(mask_bool, margin=0)
    props_list = measure.regionprops(local.astype(np.uint8))
    props = props_list[0] if props_list else None
    area = int(mask_bool.sum())

    skel_len, branch_points, endpoints, skel, skel_local = skeleton_stats(mask_bool)
    mean_width = (area / skel_len) if skel_len > 0 else float(np.sqrt(area))

    # Max width: 2x the largest distance-to-background value found ON the
    # skeleton (medial-axis radius) -- more robust to a jagged boundary than
    # measuring width at one arbitrary cross-section.
    max_width = float(np.sqrt(area))  # fallback for a to-degenerate-to-skeletonize region
    if skel_len > 0:
        # The distance transform MUST be computed on the same crop the skeleton
        # was built from. skeleton_stats() uses _local_crop() with its DEFAULT
        # margin, so passing margin=0 here produced a differently-shaped array,
        # the shape guard below always failed, and max_width silently fell back
        # to sqrt(area) for every region ever measured -- meaningless for an
        # elongated crack (a 10x140 px crack reported 37.4 instead of 10).
        dist = ndi.distance_transform_edt(_local_crop(mask_bool))
        if dist.shape == skel.shape:
            on_skel = dist[skel]
            if len(on_skel):
                max_width = float(on_skel.max() * 2)

    tortuosity = float("nan")
    if endpoints == 2 and branch_points == 0 and skel_len > 1:
        ys, xs = np.where(skel)
        # endpoints are the 2 skeleton pixels with exactly 1 neighbor
        neighbor_count = ndi.convolve(skel.astype(int), _NEIGHBOR_KERNEL, mode="constant", cval=0)
        ey, ex = np.where((neighbor_count == 1) & skel)
        if len(ey) == 2:
            straight_dist = float(np.hypot(ey[0] - ey[1], ex[0] - ex[1]))
            if straight_dist > 0:
                tortuosity = float(skel_len / straight_dist)

    return {
        "Area_px": area,
        "SkeletonLength_px": skel_len,
        "MeanWidth_px": round(mean_width, 2),
        "MaxWidth_px": round(max_width, 2),
        "Tortuosity": round(tortuosity, 3) if tortuosity == tortuosity else "",  # "" not nan, for clean CSV
        "BranchPointCount": branch_points,
        "MajorAxisLength_px": round(float(props.axis_major_length), 2) if props else "",
        "MinorAxisLength_px": round(float(props.axis_minor_length), 2) if props else "",
        "Orientation_deg": round(float(np.degrees(props.orientation)), 2) if props else "",
        "BoundaryRoughness": round(boundary_roughness(mask_bool), 3),
    }
