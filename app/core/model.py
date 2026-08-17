"""
The deployed crack model, as ONE object with ONE method.

Everything the app does goes through `CrackModel.predict(img01)`. The app never
needs to know that this is really two models averaged, that one of them needs a
SAM ViT forward pass, or how the 273-dim feature vector is assembled -- which is
the point: the frontend can offer "predict", "correct", "retrain" as buttons
because all of that complexity is behind this class.

WHY AN ENSEMBLE RATHER THAN JUST THE SAM HYBRID. Measured under
leave-one-image-out on the 4 Ilastik ground-truth images, with the crack-free
axis measured on the 6 owner-confirmed undamaged specimens:

  approach                        mean IoU   pixel-weighted   recall   crack-free FP
  17 hand-crafted features          0.744        0.721         0.891       7.43%
  SAM 256 + 17 (the hybrid)         0.795        0.719         0.894       0.14%
  mean probability of the two       0.821        0.777         0.914       0.11%

The hybrid ALONE ties the old model once you weight by pixel count (0.719 vs
0.721), because it loses badly on the one 23.5 MP mosaic -- 73% of all labelled
pixels. Averaging fixes exactly that: it wins on all four images, on both
weightings, with recall UP rather than traded away, and with the lowest false
positives on specimens known to be crack-free. That combination is why this is
the default and the hybrid alone is not.

A third member (SAM-only) was tested and adds +0.009 IoU, which is below the
measured 0.0070 retrain-noise floor, so it is not included -- it would cost a
third of the inference time for an effect indistinguishable from reseeding.

Set ensemble=False to run the hybrid alone (about 2x faster, measurably worse).
"""

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_CODE = os.path.join(_PROJECT, "code")
for p in (_CODE, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

TILE = 1024
EMB_STRIDE = 16
DEFAULT_17 = os.path.join(_PROJECT, "models", "pixel_hgb_final.joblib")
DEFAULT_HYBRID = os.path.join(_PROJECT, "models", "pixel_sam_hybrid.joblib")
SAM_MODEL_ID = "facebook/sam-vit-huge"


# ---------------------------------------------------------------- SAM embedding
_sam = None


def _get_sam():
    """Load SAM once per process. Downloads ~2.4 GB on first ever use."""
    global _sam
    if _sam is None:
        import torch
        from transformers import SamModel, SamProcessor
        dev = ("mps" if torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available() else "cpu")
        proc = SamProcessor.from_pretrained(SAM_MODEL_ID)
        model = SamModel.from_pretrained(SAM_MODEL_ID).to(dev).eval()
        _sam = (proc, model, dev, torch)
    return _sam


def tiles(shape, size=TILE):
    """Non-overlapping-by-construction tiles, clamped inward at the edges so
    every tile is exactly `size` -- which keeps the pixel->embedding mapping a
    clean divide-by-16 with no padding to reason about."""
    H, W = shape[:2]
    out = []
    for y0 in range(0, max(H - 1, 1), size):
        for x0 in range(0, max(W - 1, 1), size):
            y1, x1 = min(y0 + size, H), min(x0 + size, W)
            out.append((max(y1 - size, 0), y1, max(x1 - size, 0), x1))
    return sorted(set(out))


def embed_image(img01, progress=None):
    """Tiled SAM ViT embeddings -> (coords int32 [n,2], emb float16 [n,C,64,64])."""
    proc, model, dev, torch = _get_sam()
    tl = tiles(img01.shape)
    coords, embs = [], []
    for k, (y0, y1, x0, x1) in enumerate(tl):
        crop = img01[y0:y1, x0:x1]
        if crop.shape != (TILE, TILE):
            crop = np.pad(crop, ((0, TILE - crop.shape[0]), (0, TILE - crop.shape[1])),
                          mode="reflect")
        u8 = (np.clip(crop, 0, 1) * 255).astype(np.uint8)
        rgb = np.stack([u8] * 3, -1)
        inp = proc(rgb, return_tensors="pt")
        px = inp["pixel_values"]
        px = (px.float() if px.dtype == torch.float64 else px).to(dev)
        with torch.no_grad():
            e = model.get_image_embeddings(px)
        embs.append(e.float().cpu().numpy()[0].astype(np.float16))
        coords.append((y0, x0))
        if dev == "mps":
            torch.mps.empty_cache()
        if progress:
            progress(k + 1, len(tl))
    return np.asarray(coords, np.int32), np.stack(embs)


def interp_tile(emb_tile, rr, cc):
    """Bilinear lookup of tile-local coords in one C x 64 x 64 grid, vectorised
    across all channels in a single gather."""
    e = np.ascontiguousarray(emb_tile, dtype=np.float32)
    C, H, W = e.shape
    r = np.clip(rr / EMB_STRIDE - 0.5, 0, H - 1)
    c = np.clip(cc / EMB_STRIDE - 0.5, 0, W - 1)
    r0 = np.floor(r).astype(np.intp); c0 = np.floor(c).astype(np.intp)
    r1 = np.minimum(r0 + 1, H - 1);   c1 = np.minimum(c0 + 1, W - 1)
    dr = (r - r0).astype(np.float32)[:, None]
    dc = (c - c0).astype(np.float32)[:, None]
    f = e.reshape(C, H * W)
    return (f[:, r0 * W + c0].T * (1 - dr) * (1 - dc)
            + f[:, r0 * W + c1].T * (1 - dr) * dc
            + f[:, r1 * W + c0].T * dr * (1 - dc)
            + f[:, r1 * W + c1].T * dr * dc)


# ---------------------------------------------------------------- the model
class CrackModel:
    def __init__(self, path_17=DEFAULT_17, path_hybrid=DEFAULT_HYBRID, ensemble=True):
        import joblib
        self.ensemble = bool(ensemble)
        self.m17 = joblib.load(path_17) if os.path.exists(path_17) else None
        self.hybrid, self.n_hybrid = None, None
        if os.path.exists(path_hybrid):
            b = joblib.load(path_hybrid)
            self.hybrid = b["model"] if isinstance(b, dict) else b
            self.n_hybrid = (b.get("n_features", 273) if isinstance(b, dict) else 273)
        if self.m17 is None and self.hybrid is None:
            raise FileNotFoundError("no model found -- expected models/pixel_hgb_final.joblib "
                                    "and/or models/pixel_sam_hybrid.joblib")
        if self.ensemble and (self.m17 is None or self.hybrid is None):
            self.ensemble = False   # fall back rather than silently averaging one thing

    def describe(self):
        parts = []
        if self.m17 is not None:
            parts.append("17-feature MLP")
        if self.hybrid is not None:
            parts.append(f"SAM+17 hybrid ({self.n_hybrid}d)")
        mode = "mean-probability ensemble" if self.ensemble else "single model"
        return f"{mode}: {' + '.join(parts)}"

    def needs_sam(self):
        return self.hybrid is not None

    def predict(self, img01, emb=None, band=128, progress=None):
        """Crack probability map for a normalised image.

        `emb` is (coords, embeddings) from embed_image(); computed here if not
        supplied. Chunked by tile and row band -- a 23.5 MP image at 273 float32
        features would be 26 GB if materialised at once.
        """
        from txm_features import compute_feature_stack
        f17 = compute_feature_stack(img01)
        H, W = img01.shape

        p17 = None
        if self.m17 is not None:
            p17 = np.zeros((H, W), np.float32)
            for r0 in range(0, H, 256):
                r1 = min(r0 + 256, H)
                blk = np.asarray(f17[r0:r1], np.float32).reshape(-1, f17.shape[2])
                p17[r0:r1] = self.m17.predict_proba(blk)[:, 1].reshape(r1 - r0, W)
            if progress:
                progress("17-feature model", 1, 1)

        ph = None
        if self.hybrid is not None:
            if emb is None:
                emb = embed_image(img01, progress=(lambda k, n: progress("SAM embedding", k, n))
                                  if progress else None)
            coords, embs = emb
            ph = np.zeros((H, W), np.float32)
            done = np.zeros((H, W), bool)
            for k in range(len(coords) - 1, -1, -1):
                y0, x0 = int(coords[k][0]), int(coords[k][1])
                y1, x1 = min(y0 + TILE, H), min(x0 + TILE, W)
                for b0 in range(y0, y1, band):
                    b1 = min(b0 + band, y1)
                    sub = ~done[b0:b1, x0:x1]
                    if not sub.any():
                        continue
                    rr, cc = np.nonzero(sub)
                    rg, cg = rr + b0, cc + x0
                    X = np.concatenate([np.asarray(f17[rg, cg, :], np.float32),
                                        interp_tile(embs[k], rg - y0, cg - x0)], axis=1)
                    pr = self.hybrid.predict_proba(X)[:, 1].astype(np.float32)
                    blk = ph[b0:b1, x0:x1]; blk[rr, cc] = pr; ph[b0:b1, x0:x1] = blk
                    d = done[b0:b1, x0:x1]; d[rr, cc] = True; done[b0:b1, x0:x1] = d
                if progress:
                    progress("hybrid model", len(coords) - k, len(coords))
        del f17

        if self.ensemble and p17 is not None and ph is not None:
            return (p17 + ph) / 2.0
        return ph if ph is not None else p17


def postprocess(prob):
    """The project's standard mask cleanup. Kept behind a function so the app can
    offer it as a toggle: it is under suspicion of deleting thin hand-painted
    crack (raw-threshold stroke recall 0.869 vs 0.14-0.40 post-processed), which
    is unresolved."""
    from apply_pixel_model import postprocess_mask
    return postprocess_mask(prob)
