"""
Shared harness for evaluating Meta's Segment Anything (SAM / SAM 2) on the
TXM crack images, on EXACTLY the same footing as the deployed pixel model.

Why this file exists: the professor's question ("why not SAM?") is only
answerable with numbers, and numbers are only comparable if the metric, the
ground truth and the held-out images are identical to the existing benchmark.
So `metrics_from_pred` is IMPORTED from generate_benchmark_report rather than
re-implemented -- it cannot silently drift out of parity.

Design bias, stated up front: every ambiguous choice here is resolved in
SAM's FAVOUR. Multiple input renderings, prompts derived from the ground
truth SAM is being scored against (physically impossible in deployment),
tiling at native resolution so thin structures are not destroyed by
downsampling, and oracle mask-selection. If SAM loses even with those
advantages, the conclusion is safe. If it wins, the advantage it needed is
recorded so nobody mistakes an oracle result for a deployable one.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_benchmark_report import metrics_from_pred  # noqa: E402  (parity: single definition)

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_CACHE = os.path.join(PROJECT_DIR, "dataset_cache")
FLAT_CACHE = os.path.join(PROJECT_DIR, "dataset_cache_flatfield")

# The only four images with pixel-level truth. All are B2 -- which is itself a
# finding, not an oversight: no AM/Wrought/B3 ground truth exists to score on.
STEMS = ["333_75_um_zoom", "336_25", "338_13", "LARGE_343_75"]

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_pair(stem, cache="raw"):
    """Return (img float32 in [0,1], gt bool) for one ground-truth image."""
    d = RAW_CACHE if cache == "raw" else FLAT_CACHE
    img = np.load(os.path.join(d, f"{stem}_img.npy")).astype(np.float32)
    gt = np.load(os.path.join(d, f"{stem}_gt.npy")).astype(bool)
    return img, gt


def to_rgb(img, mode="gray"):
    """Grayscale -> 3-channel uint8, the format SAM's processor expects.

    Several renderings because the right one is not obvious and picking badly
    would understate SAM unfairly:
      gray   -- straight replicate; the honest default
      clahe  -- local contrast equalisation; makes faint cracks far more visible
      invert -- cracks are DARK, and SAM's training data is dominated by
                bright foreground objects on darker backgrounds
    """
    g = np.clip(img, 0, 1)
    if mode == "invert":
        g = 1.0 - g
    u8 = (g * 255).astype(np.uint8)
    if mode == "clahe":
        import cv2
        u8 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16)).apply(u8)
    return np.stack([u8] * 3, axis=-1)


def tiles(shape, size=1024, overlap=128):
    """Tile coordinates covering `shape`, so SAM sees native resolution.

    SAM resizes its input so the long edge is 1024. On a 6367px-wide mosaic
    that is a 6.2x downsample, which is why whole-image inference must not be
    the only condition tested -- it would destroy any structure a few pixels
    wide before the model ever sees it.
    """
    H, W = shape[:2]
    step = max(size - overlap, 1)
    ys = list(range(0, max(H - overlap, 1), step))
    xs = list(range(0, max(W - overlap, 1), step))
    out = []
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + size, H), min(x0 + size, W)
            out.append((max(y1 - size, 0), y1, max(x1 - size, 0), x1))
    return sorted(set(out))


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
_CACHE = {}


def get_sam(model_id="facebook/sam-vit-huge", device=None):
    """Load and cache a SAM checkpoint. Returns (processor, model, device)."""
    device = device or DEVICE
    key = (model_id, device)
    if key not in _CACHE:
        if "sam2" in model_id:
            from transformers import Sam2Model as M, Sam2Processor as P
        else:
            from transformers import SamModel as M, SamProcessor as P
        proc = P.from_pretrained(model_id)
        model = M.from_pretrained(model_id).to(device).eval()
        _CACHE[key] = (proc, model, device)
    return _CACHE[key]


def _to_dev(inp, device):
    """MPS rejects float64, which the processor emits for sizes/points."""
    out = {}
    for k, v in inp.items():
        if torch.is_tensor(v):
            out[k] = (v.float() if v.dtype == torch.float64 else v).to(device)
        else:
            out[k] = v
    return out


def predict_prompted(rgb, points=None, labels=None, boxes=None,
                     model_id="facebook/sam-vit-huge", multimask=True, device=None):
    """Run SAM on one image with point and/or box prompts.

    points: (N, P, 2) xy per prompt-group, or None
    boxes:  (N, 4) xyxy, or None
    Returns (masks bool [N, M, H, W], confidences float [N, M]).
    """
    proc, model, device = get_sam(model_id, device)
    kw = {}
    if points is not None:
        kw["input_points"] = [np.asarray(points, dtype=np.float32).tolist()]
        if labels is not None:
            kw["input_labels"] = [np.asarray(labels, dtype=np.int64).tolist()]
    if boxes is not None:
        kw["input_boxes"] = [np.asarray(boxes, dtype=np.float32).tolist()]
    inp = proc(rgb, return_tensors="pt", **kw)
    orig = inp["original_sizes"]
    resh = inp.get("reshaped_input_sizes")
    with torch.no_grad():
        out = model(**_to_dev(inp, device), multimask_output=multimask)
    masks = _post_process(proc, out.pred_masks.float().cpu(), orig, resh)[0]
    return masks.numpy().astype(bool), out.iou_scores.float().cpu().numpy()[0]


def _post_process(proc, pred_masks, orig, resh):
    """SAM 1 exposes post_process_masks on .image_processor; SAM 2 puts it on
    the processor itself and drops reshaped_input_sizes. Try both so one code
    path serves both generations."""
    if hasattr(proc, "post_process_masks"):
        try:
            return proc.post_process_masks(pred_masks, orig)
        except TypeError:
            return proc.post_process_masks(pred_masks, orig, resh)
    return proc.image_processor.post_process_masks(pred_masks, orig, resh)


def embed(rgb, model_id="facebook/sam-vit-huge", device=None):
    """SAM's ViT image embedding: (C, h, w), h=w=64 for ViT-H at 1024 input.

    This is the route that actually gives SAM a fair shot at BEATING the 17
    hand-crafted features -- use the foundation model as a feature extractor
    and train a small head on top, instead of asking it to guess what a crack
    is with no supervision at all.
    """
    proc, model, device = get_sam(model_id, device)
    inp = proc(rgb, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_embeddings(_to_dev(inp, device)["pixel_values"])
    return emb.float().cpu().numpy()[0]


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------
def grid_points(shape, n=32, margin=16):
    """Uniform n x n point grid -- the deployable prompt (no GT needed)."""
    H, W = shape[:2]
    ys = np.linspace(margin, H - 1 - margin, n)
    xs = np.linspace(margin, W - 1 - margin, n)
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float32)


def oracle_points_from_gt(gt, n=32, rng=None, on_skeleton=True):
    """Points sampled ON the true crack. NOT deployable -- an upper bound.

    Answers the sharper question: given that you already know where the crack
    is, can SAM trace its extent? That separates "SAM cannot find cracks"
    from "SAM cannot delineate cracks even when handed their location".
    """
    rng = rng or np.random.RandomState(0)
    src = gt
    if on_skeleton:
        from skimage.morphology import skeletonize
        sk = skeletonize(gt)
        if sk.sum() >= n:
            src = sk
    ys, xs = np.nonzero(src)
    if len(ys) == 0:
        return np.zeros((0, 2), np.float32)
    idx = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)


def oracle_boxes_from_gt(gt, max_boxes=32, min_area=200):
    """Per-connected-component bounding boxes of the true crack. NOT deployable."""
    from skimage.measure import label, regionprops
    props = [r for r in regionprops(label(gt, connectivity=2)) if r.area >= min_area]
    props.sort(key=lambda r: -r.area)
    out = []
    for r in props[:max_boxes]:
        y0, x0, y1, x1 = r.bbox
        out.append([x0, y0, x1, y1])
    return np.array(out, dtype=np.float32) if out else np.zeros((0, 4), np.float32)


# --------------------------------------------------------------------------
# mask selection
# --------------------------------------------------------------------------
def select_by_darkness(masks, img, q=0.5, max_area_frac=0.5):
    """Deployable heuristic: keep masks whose mean intensity is below the
    image's q-quantile AND that are not larger than max_area_frac of the frame.

    The area cap is not cosmetic. On three of the four ground-truth images the
    frame MEAN is below the frame MEDIAN (0.689 vs 0.715, 0.677 vs 0.711,
    0.663 vs 0.701), so a proposal covering the entire specimen PASSES a pure
    darkness test. That single mask then caps IoU at the ground-truth area
    fraction, ~0.19-0.30, no matter what else is selected -- which is exactly
    the 98.5%-of-frame prediction seen on 336_25. No single proposal can
    plausibly be the crack if it is half the image, so reject it.

    SAM is class-agnostic -- it returns unlabelled regions with no labels at
    all. Something has to decide which of them are cracks, and 'cracks are
    dark' is the cheapest honest rule available without training anything.

    q is a real quantile so it can be swept, which removes "you picked a bad
    threshold" as an objection to the result. q=0.5 (the median) reproduces
    the earlier formulation exactly, so numbers already collected at the
    default remain valid.
    """
    if len(masks) == 0:
        return np.zeros(img.shape, bool)
    thr = np.quantile(img, q)
    keep = np.zeros(img.shape, bool)
    for m in masks:
        if not m.any():
            continue
        if max_area_frac is not None and m.mean() > max_area_frac:
            continue
        if img[m].mean() < thr:
            keep |= m
    return keep


def select_oracle_union(masks, gt):
    """Best subset of SAM's masks that a PERFECT picker could choose. NOT deployable.

    The ceiling on 'SAM's proposals + any selection rule I could invent'. If
    this is still low, the limitation lives in SAM's proposals, not in the
    selection rule -- a distinction worth establishing rather than assuming.

    Two strategies, best-of taken, because a single greedy pass is not a
    ceiling and an "upper bound" that a dumb heuristic can beat is a bug:

      purity-greedy: consider masks in descending PURITY (fraction of the mask
        that is truly crack) and keep any that improve global IoU. Ordering by
        descending AREA instead -- the obvious first guess -- is actively
        harmful: it offers the near-full-frame mask first, that mask improves
        IoU over an empty prediction, so it gets locked in and every later
        mask is judged against an already-ruined state. That ordering made
        this "oracle" score BELOW the darkness heuristic it is supposed to
        bound.
      majority-rule: keep every mask that is more than half truly crack. Cheap,
        order-independent, and often better than any greedy pass.
    """
    if len(masks) == 0:
        return np.zeros(gt.shape, bool)

    purity, inter = [], []
    for m in masks:
        s = int(m.sum())
        i = int(np.logical_and(m, gt).sum())
        inter.append(i)
        purity.append(i / s if s else 0.0)

    cur = np.zeros(gt.shape, bool)
    best = 0.0
    for i in sorted(range(len(masks)), key=lambda k: -purity[k]):
        cand = cur | masks[i]
        iou = metrics_from_pred(cand, gt)["iou"]
        if iou > best:
            best, cur = iou, cand

    maj = np.zeros(gt.shape, bool)
    for k, m in enumerate(masks):
        if purity[k] > 0.5:
            maj |= m
    if metrics_from_pred(maj, gt)["iou"] > best:
        return maj
    return cur


def score(pred, gt, **extra):
    """Metrics plus predicted area, which is what earlier mistakes hid behind."""
    m = metrics_from_pred(np.asarray(pred, bool), np.asarray(gt, bool))
    m["pred_area_frac"] = float(np.mean(pred))
    m["gt_area_frac"] = float(np.mean(gt))
    m.update(extra)
    return m
