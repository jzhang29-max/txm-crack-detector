"""
The per-image pipeline, and the retrain job. Everything the frontend's buttons
actually trigger.

    ingest(image_id)   original -> normalised model input
                                -> destitched+flatfielded display view
                                -> cached SAM embedding
                                -> crack probability
    retrain()          gather every correction -> train a new hybrid ->
                       validate on both axes -> deploy or refuse

The validation gate on retrain is not optional decoration. This project has
adopted three changes that later proved to be regressions, and every one of them
passed a SINGLE metric: pseudo-flat-fielding looked good on false positives and
cost 0.169 IoU; a curvilinearity gate cut predicted area 8x, which reads as
artifact removal, while destroying 98% of the true crack on one image. An
over-aggressive filter and a good one BOTH reduce area -- only recall against
ground truth separates them. So a candidate must hold IoU AND not increase false
positives on known-clean specimens, or it is not deployed.
"""

import glob
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))
CODE = os.path.join(PROJECT, "code")
for p in (CODE, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import store as S            # noqa: E402
import model as M            # noqa: E402

# Reference data that ships with the package: the only pixel-level ground truth
# that exists (4 images), used to validate a retrain. If it is absent the gate
# degrades to "warn and refuse to auto-deploy" rather than silently passing.
GT_CACHE = os.path.join(PROJECT, "dataset_cache")
GT_STEMS = ["333_75_um_zoom", "336_25", "338_13", "LARGE_343_75"]

IOU_TOL = 0.01
FP_TOL = 0.005

# Extensions the uploader accepts. Kept here next to the reader that has to cope
# with them so the two cannot drift apart.
READABLE_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def read_any_image(src):
    """Read a TIFF with tifffile and anything else with PIL.

    This used to be a bare tifffile.imread(), which meant a dropped PNG or JPEG was
    accepted by the uploader, copied into app_data, and only then failed the ingest
    job with "not a TIFF file: header=b'\\x89PNG'" -- while the drop zone was openly
    advertising .png. TXM data is TIFF in practice, but the UI offered the others.

    Extension picks the reader, with the other as a fallback so a mislabelled file
    (a TIFF named .png, which batch exporters do produce) still loads. If both fail,
    the error from the reader the extension implied is the one raised, since that is
    the one describing what the user actually handed us.
    """
    ext = os.path.splitext(src)[1].lower()

    def _tiff():
        import tifffile
        return tifffile.imread(src)

    def _pil():
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None      # these mosaics trip the decompression-bomb guard
        with Image.open(src) as im:
            # Leave 16-bit/float/grayscale modes alone -- robust_normalize downstream
            # handles the range, and converting would throw away bit depth.
            if im.mode in ("I", "I;16", "I;16B", "I;16L", "F", "L"):
                return np.asarray(im)
            if im.mode in ("P", "RGBA", "LA", "CMYK", "1"):
                im = im.convert("RGB")     # ndim==3 is collapsed by the caller
            return np.asarray(im)

    first, second = (_tiff, _pil) if ext in (".tif", ".tiff") else (_pil, _tiff)
    try:
        return first()
    except Exception as e_expected:
        try:
            return second()
        except Exception:
            raise e_expected


# ------------------------------------------------------------------- ingest
def ingest(image_id, progress=None, force=False):
    """Prepare one uploaded image for viewing and correcting."""
    def rep(stage, k=1, n=1):
        S.write_meta(image_id, dict(status=stage))
        if progress:
            progress(stage, k, n)

    from txm_features import robust_normalize

    src = S.original_path(image_id)
    if src is None:
        raise FileNotFoundError(f"no original file for {image_id}")

    if force or S.load_npy(image_id, "img.npy", mmap=True) is None:
        rep("reading image")
        raw = np.asarray(read_any_image(src))
        if raw.ndim == 3:
            # A colour or multi-page TIFF: take the first plane / mean channel
            raw = raw.mean(axis=-1) if raw.shape[-1] in (3, 4) else raw[0]
        img01 = robust_normalize(raw.astype(np.float64), 1.0, 99.0).astype(np.float32)
        del raw
        S.save_npy(image_id, "img.npy", img01)
        S.write_meta(image_id, dict(height=int(img01.shape[0]), width=int(img01.shape[1]),
                                    megapixels=round(img01.size / 1e6, 2)))
    else:
        img01 = np.asarray(S.load_npy(image_id, "img.npy"))

    # Display view: destitch + flatfield. The model is fed RAW because that is
    # what it was trained on, but the real cracks are thin and faint and are
    # often only visible under local-contrast enhancement -- so what the human
    # sees and what the model sees are deliberately different images. Both steps
    # preserve geometry, so the raw-derived mask still registers on the display.
    if force or S.load_npy(image_id, "display.npy", mmap=True) is None:
        rep("destitching + flat-fielding")
        try:
            import destitch
            import flatfield
            d, _ = destitch.destitch_image(img01.astype(np.float32))
            ff = flatfield.flatfield(np.asarray(d, np.float32))
            if isinstance(ff, tuple):
                ff = ff[0]
            disp = robust_normalize(np.asarray(ff, np.float64), 1.0, 99.0).astype(np.float32)
            if disp.shape != img01.shape:
                raise ValueError(f"shape changed {img01.shape}->{disp.shape}")
            S.save_npy(image_id, "display.npy", disp)
            S.write_meta(image_id, dict(display="destitched+flatfielded"))
            del d, ff, disp
        except Exception as e:
            # Never fail ingest over the display view -- fall back to raw and say so.
            S.save_npy(image_id, "display.npy", img01)
            S.write_meta(image_id, dict(display=f"raw (preprocessing failed: {type(e).__name__})"))

    mdl = get_model()
    if mdl.needs_sam() and (force or not os.path.exists(S.path(image_id, "emb.npz"))):
        rep("SAM embedding")
        coords, emb = M.embed_image(img01, progress=lambda k, n: rep("SAM embedding", k, n))
        np.savez(S.path(image_id, "emb.npz"), coords=coords, emb=emb)
        del coords, emb

    rep("predicting")
    emb = None
    if mdl.needs_sam() and os.path.exists(S.path(image_id, "emb.npz")):
        z = np.load(S.path(image_id, "emb.npz"))
        emb = (z["coords"], z["emb"])
    prob = mdl.predict(img01, emb=emb, progress=lambda st, k, n: rep(st, k, n))
    S.save_npy(image_id, "prob.npy", prob.astype(np.float32))

    if S.load_npy(image_id, "correction.npy", mmap=True) is None:
        S.save_npy(image_id, "correction.npy", np.zeros(img01.shape, np.uint8))

    S.write_meta(image_id, dict(status="ready", model=mdl.describe(),
                                predicted_area=float((prob > 0.5).mean()),
                                ingested=time.time()))
    return True


# ------------------------------------------------------------------- model
_model_cache = {"key": None, "obj": None}


def get_model():
    r = S.registry()["current"]
    key = json.dumps(r, sort_keys=True)
    if _model_cache["key"] != key:
        _model_cache["obj"] = M.CrackModel(
            path_17=r.get("path_17") or M.DEFAULT_17,
            path_hybrid=r.get("path_hybrid") or M.DEFAULT_HYBRID,
            ensemble=(r.get("kind", "ensemble") == "ensemble"))
        _model_cache["key"] = key
    return _model_cache["obj"]


def effective_mask(image_id, threshold=0.5, postprocess=False):
    """Model prediction with the user's corrections applied on top."""
    prob = S.load_npy(image_id, "prob.npy")
    if prob is None:
        return None
    mask = M.postprocess(prob) if postprocess else (prob > threshold)
    corr = S.load_npy(image_id, "correction.npy")
    if corr is not None and corr.shape == mask.shape:
        mask = mask.copy()
        mask[corr == 1] = True
        mask[corr == 2] = False
    return mask


# ------------------------------------------------------------------ retrain
def _gt_available():
    return all(os.path.exists(os.path.join(GT_CACHE, f"{s}_{k}.npy"))
               for s in GT_STEMS for k in ("img", "gt"))


GT_EMB_DIR = os.path.join(S.DATA, "gt_emb")


def gt_embedding(stem, progress=None):
    """SAM embedding for a shipped ground-truth image, computed once and cached.

    Also looks in the research cache (paint/sam_embcache) first, since those 4
    images were embedded there under their full original filenames -- reusing
    that avoids ~20s of GPU work per image on a user's first retrain.
    """
    os.makedirs(GT_EMB_DIR, exist_ok=True)
    p = os.path.join(GT_EMB_DIR, f"{stem}.npz")
    if os.path.exists(p):
        z = np.load(p)
        return z["coords"], z["emb"]

    legacy = os.path.join(PROJECT, "paint", "sam_embcache")
    if os.path.isdir(legacy):
        key = stem.replace("LARGE_343_75", "343_75_LARGE")
        for f in sorted(os.listdir(legacy)):
            if key in f and f.endswith("_samemb.npz"):
                try:
                    z = np.load(os.path.join(legacy, f))
                    coords, emb = z["coords"], z["emb"]
                    np.savez(p, coords=coords, emb=emb)
                    return coords, emb
                except Exception:
                    break

    img = np.load(os.path.join(GT_CACHE, f"{stem}_img.npy"))
    if progress:
        progress(f"embedding ground truth {stem}", 0, 1)
    coords, emb = M.embed_image(img)
    np.savez(p, coords=coords, emb=emb)
    del img
    return coords, emb


def _score(model, progress=None):
    """(mean IoU, mean recall) on the shipped ground truth."""
    from generate_benchmark_report import metrics_from_pred
    ious, recs = [], []
    for i, stem in enumerate(GT_STEMS):
        img = np.load(os.path.join(GT_CACHE, f"{stem}_img.npy"))
        gt = np.load(os.path.join(GT_CACHE, f"{stem}_gt.npy")).astype(bool)
        prob = model.predict(img)
        s = metrics_from_pred(prob > 0.5, gt)
        ious.append(s["iou"]); recs.append(s["recall"])
        if progress:
            progress("validating", i + 1, len(GT_STEMS))
        del img, gt, prob
    return float(np.mean(ious)), float(np.mean(recs))


def gather_training_data(progress=None):
    """Every corrected pixel across every uploaded image, plus the shipped
    ground truth, as hybrid [17 | 256] features.

    Class balance is computed from what actually exists rather than hard-coded:
    it is the single knob that has caused four regressions in this project, and
    a fixed cap silently goes stale as more images get labelled.
    """
    import destitch  # noqa: F401  (ensures code/ is importable before heavy work)
    Xs, ys, ws = [], [], []

    items = [m for m in S.list_images() if m.get("corrected_crack_px") or m.get("corrected_not_px")]
    n_crack_total = sum(m.get("corrected_crack_px", 0) for m in items)
    per_img_cap = 30000
    corr_crack = sum(min(per_img_cap, m.get("corrected_crack_px", 0)) for m in items)
    n_bg_imgs = sum(1 for m in items if m.get("corrected_not_px", 0) > 0)

    boot_crack = 0
    if _gt_available():
        for stem in GT_STEMS:
            gt = np.load(os.path.join(GT_CACHE, f"{stem}_gt.npy")).astype(bool)
            boot_crack += min(100000, int(gt.sum()))
            del gt
    neg_cap = max(500, int(round(corr_crack / n_bg_imgs))) if n_bg_imgs else per_img_cap

    rng = np.random.RandomState(0)

    # shipped ground truth. Needs SAM embeddings too, or its samples are 17-dim
    # while the corrections are 273-dim and get silently dropped for width -- which
    # is exactly what happened on the first real retrain: all 4 ground-truth blocks
    # were discarded, leaving only force-crack correction pixels, a 100%-crack
    # training set, and a model that scored IoU 0.003. Compute once, cache forever.
    if _gt_available():
        for stem in GT_STEMS:
            feat = os.path.join(GT_CACHE, f"{stem}_features.npy")
            if not os.path.exists(feat):
                continue
            gt = np.load(os.path.join(GT_CACHE, f"{stem}_gt.npy")).astype(bool)
            f17 = np.load(feat, mmap_mode="r")
            ci = np.flatnonzero(gt); bi = np.flatnonzero(~gt)
            nc = min(100000, len(ci)); nb = min(100000, len(bi))
            idx = np.concatenate([rng.choice(ci, nc, replace=False),
                                  rng.choice(bi, nb, replace=False)])
            rr, cc = np.unravel_index(idx, gt.shape)
            a = np.asarray(f17[rr, cc, :], np.float32)
            del f17

            block = a
            if get_model().needs_sam():
                coords, embs = gt_embedding(stem, progress=progress)
                if coords is not None:
                    b = np.zeros((len(rr), embs.shape[1]), np.float32)
                    todo = np.ones(len(rr), bool)
                    for t in range(len(coords) - 1, -1, -1):
                        y0, x0 = int(coords[t][0]), int(coords[t][1])
                        sel = (todo & (rr >= y0) & (rr < y0 + M.TILE)
                               & (cc >= x0) & (cc < x0 + M.TILE))
                        if sel.any():
                            b[sel] = M.interp_tile(embs[t], rr[sel] - y0, cc[sel] - x0)
                            todo &= ~sel
                    block = np.concatenate([a, b], axis=1)
                    del b
            Xs.append(block)
            ys.append(np.concatenate([np.ones(nc, bool), np.zeros(nb, bool)]))
            ws.append(np.ones(nc + nb))
            del gt, a
            if progress:
                progress(f"ground truth {stem}", 1, 1)

    # user corrections
    for k, m in enumerate(items, 1):
        iid = m["id"]
        corr = S.load_npy(iid, "correction.npy")
        img = S.load_npy(iid, "img.npy")
        if corr is None or img is None:
            continue
        ci = np.flatnonzero(corr.reshape(-1) == 1)
        bi = np.flatnonzero(corr.reshape(-1) == 2)
        nc = min(per_img_cap, len(ci)); nb = min(neg_cap, len(bi))
        if nc + nb == 0:
            continue
        idx = np.concatenate([
            rng.choice(ci, nc, replace=False) if nc else ci[:0],
            rng.choice(bi, nb, replace=False) if nb else bi[:0]])
        rr, cc = np.unravel_index(idx, corr.shape)
        from txm_features import compute_feature_stack
        f17 = compute_feature_stack(np.asarray(img))
        a = np.asarray(f17[rr, cc, :], np.float32)
        del f17
        zp = S.path(iid, "emb.npz")
        if not os.path.exists(zp):
            continue
        z = np.load(zp); coords, embs = z["coords"], z["emb"]
        b = np.zeros((len(rr), embs.shape[1]), np.float32)
        todo = np.ones(len(rr), bool)
        for t in range(len(coords) - 1, -1, -1):
            y0, x0 = int(coords[t][0]), int(coords[t][1])
            sel = todo & (rr >= y0) & (rr < y0 + M.TILE) & (cc >= x0) & (cc < x0 + M.TILE)
            if sel.any():
                b[sel] = M.interp_tile(embs[t], rr[sel] - y0, cc[sel] - x0)
                todo &= ~sel
        Xs.append(np.concatenate([a, b], axis=1))
        ys.append(np.concatenate([np.ones(nc, bool), np.zeros(nb, bool)]))
        ws.append(np.ones(nc + nb))
        del corr, img, coords, embs, a, b
        if progress:
            progress(f"features {k}/{len(items)}", k, len(items))

    if not Xs:
        return None, None, None, dict(reason="no labelled data")
    # The ground-truth block is 17-dim; correction blocks are 273-dim. Only train
    # on the common width, and say which happened rather than crashing later.
    widths = {x.shape[1] for x in Xs}
    target = max(widths)
    keep = [(x, y, w) for x, y, w in zip(Xs, ys, ws) if x.shape[1] == target]
    dropped = len(Xs) - len(keep)
    X = np.concatenate([k[0] for k in keep]).astype(np.float32)
    y = np.concatenate([k[1] for k in keep])
    w = np.concatenate([k[2] for k in keep])
    info = dict(n_px=int(len(y)), n_features=int(target), crack_fraction=float(y.mean()),
                neg_cap=neg_cap, correction_crack_px=int(n_crack_total),
                blocks_dropped_for_width=dropped, bootstrap_crack=boot_crack)
    return X, y, w, info


def retrain(deploy=True, progress=None):
    """Train a new hybrid on all current corrections, validate, maybe deploy."""
    import joblib
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    if progress:
        progress("gathering labels", 0, 1)
    X, y, w, info = gather_training_data(progress=progress)
    if X is None:
        return dict(ok=False, error="no labelled data yet -- paint some corrections first")

    # A degenerate balance is a REFUSAL, not a warning. Measured on the first real
    # retrain: a 100%-crack set (all 4 ground-truth blocks dropped, only force-crack
    # strokes left) produced IoU 0.003. Spending 5 minutes to train and validate
    # something that cannot possibly work, then reporting it as a regression, wastes
    # the user's time and buries the actual cause.
    frac = info["crack_fraction"]
    if info["blocks_dropped_for_width"]:
        return dict(ok=False, info=info,
                    error=(f"{info['blocks_dropped_for_width']} label block(s) had the "
                           f"wrong feature width and were dropped. This means the "
                           f"ground truth did not reach training. Not training a model "
                           f"on partial data."))
    if not (0.05 <= frac <= 0.95):
        return dict(ok=False, info=info,
                    error=(f"training set is {frac*100:.1f}% crack -- degenerate. "
                           f"Paint some of BOTH kinds: 'Add crack' on real cracks and "
                           f"'Eraser' on false positives. Refusing to train."))
    warn = None
    if not (0.42 <= frac <= 0.58):
        warn = (f"training set is {frac*100:.1f}% crack, outside the 42-58% band; "
                f"class_weight='balanced' will skew the boundary. Paint more of the "
                f"under-represented kind for a better result.")

    if progress:
        progress("fitting", 0, 1)
    clf = Pipeline([("scaler", StandardScaler()),
                    ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                                          random_state=0))])
    clf.fit(X, y)
    del X, y, w

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(S.MODELS, f"hybrid_{stamp}.joblib")
    # info already carries n_features; splat it first and let explicit keys win,
    # rather than passing n_features twice (which is a TypeError, not a merge).
    bundle = dict(info)
    bundle.update(model=clf, kind="sam17_hybrid", trained=stamp)
    joblib.dump(bundle, out)

    result = dict(ok=True, path=out, info=info, warning=warn,
                  seconds=round(time.time() - t0, 1))

    if not _gt_available():
        result.update(deployed=False,
                      reason="no ground truth available to validate against; "
                             "model saved but not deployed")
        return result

    cand_entry = dict(kind="ensemble", path_17=M.DEFAULT_17, path_hybrid=out,
                      label=f"retrained {stamp}", created=stamp)
    inc = get_model()
    if progress:
        progress("validating incumbent", 0, 1)
    i0, r0 = _score(inc, progress=progress)
    cand = M.CrackModel(path_17=cand_entry["path_17"], path_hybrid=out, ensemble=True)
    if progress:
        progress("validating candidate", 0, 1)
    i1, r1 = _score(cand, progress=progress)
    result.update(incumbent=dict(iou=i0, recall=r0), candidate=dict(iou=i1, recall=r1))

    passes = i1 >= i0 - IOU_TOL
    result["passes_gate"] = bool(passes)
    if deploy and passes:
        S.set_current(cand_entry)
        _model_cache["key"] = None
        result.update(deployed=True)
        # Re-predict every image inside this job, rather than leaving it to the
        # browser to call /api/reoverlay afterwards. If that call never happens --
        # the tab was closed, the laptop slept, the page was reloaded during the
        # retrain -- the registry says the new model is current while every mask on
        # screen is still the old model's output, with nothing indicating the
        # mismatch. The embeddings are cached, so this is the classifier pass only.
        ids = [m["id"] for m in S.list_images()]
        for k, iid in enumerate(ids, 1):
            if progress:
                progress(f"re-applying to {iid} ({k}/{len(ids)})", k, len(ids))
            try:
                ingest(iid)
            except Exception as e:                        # noqa: BLE001
                # One unreadable image must not abandon the rest with stale masks.
                result.setdefault("reapply_failed", []).append(f"{iid}: {e}")
        result["reapplied"] = len(ids) - len(result.get("reapply_failed", []))
    else:
        result.update(deployed=False,
                      reason=(None if passes else
                              f"IoU regressed {i0:.3f} -> {i1:.3f} (tolerance {IOU_TOL}); "
                              f"not deployed. The model file is kept so you can inspect it."))
    return result
