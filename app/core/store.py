"""
Storage for the app. One directory per uploaded image, everything derived from
it alongside it, and a model registry with history.

Design constraint that drove this: the previous pipeline read images from a
hard-coded absolute path (~/Desktop/TXM DATA). When that folder went missing the
whole tool stopped working -- list_images() returned 0 and nothing could load.
So here the app OWNS its data: an uploaded image is copied inside app_data/ and
never referenced outside it again. Nothing breaks if you move or delete the
original.

Layout:
    app_data/
      images/<id>/original.<ext>   the file as uploaded, untouched
                  img.npy         normalised, MODEL input (raw path)
                  display.npy     destitched + flatfielded, HUMAN view
                  emb.npz         cached SAM embedding (the expensive part)
                  prob.npy        crack probability from the current model
                  correction.npy  uint8: 0 untouched, 1 force-crack, 2 force-not
                  meta.json
      models/registry.json        current model + retrain history
"""

import hashlib
import json
import os
import shutil
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))
DATA = os.path.join(PROJECT, "app_data")
IMAGES = os.path.join(DATA, "images")
MODELS = os.path.join(DATA, "models")
REGISTRY = os.path.join(MODELS, "registry.json")

for d in (DATA, IMAGES, MODELS):
    os.makedirs(d, exist_ok=True)


def _slug(name):
    base = os.path.splitext(os.path.basename(name))[0]
    keep = "".join(c if (c.isalnum() or c in "-_") else "_" for c in base)[:80]
    return keep or "image"


def new_id(filename, content=None):
    """Stable id: slug + short content hash, so re-uploading the same file
    reuses its entry (and its expensive SAM embedding) instead of duplicating."""
    h = hashlib.sha1(content if content is not None else filename.encode()).hexdigest()[:8]
    return f"{_slug(filename)}__{h}"


def path(image_id, *parts):
    return os.path.join(IMAGES, image_id, *parts)


# --------------------------------------------------------- per-image locking
_locks = {}
_locks_guard = threading.Lock()


def image_lock(image_id):
    """Serialise read-modify-write of one image's correction array.

    server.py runs Flask with threaded=True, and every brush stroke is a
    load correction.npy -> modify -> save cycle. Two of those in flight at once --
    a fast painter whose strokes overlap, an undo pressed while a stroke is still
    saving, or two browser tabs on the same image -- both read the same array and the
    second save overwrites the first, which had already told the user "saved". The
    stroke is not just delayed, it is gone, and nothing anywhere reports a problem.

    Per image rather than one global lock, so painting one image never blocks
    predicting or exporting another.
    """
    with _locks_guard:
        lk = _locks.get(image_id)
        if lk is None:
            lk = _locks[image_id] = threading.Lock()
        return lk


def exists(image_id):
    return os.path.isdir(path(image_id))


def save_upload(filename, content):
    """Write an uploaded file into its own directory. Returns (id, is_new)."""
    iid = new_id(filename, content)
    d = path(iid)
    if os.path.isdir(d) and os.path.exists(path(iid, "meta.json")):
        return iid, False
    os.makedirs(d, exist_ok=True)
    ext = os.path.splitext(filename)[1] or ".tif"
    with open(path(iid, "original" + ext), "wb") as f:
        f.write(content)
    write_meta(iid, dict(id=iid, filename=filename, ext=ext,
                         uploaded=time.time(), status="uploaded"))
    return iid, True


def read_meta(image_id):
    p = path(image_id, "meta.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def write_json(p, obj):
    """Atomic JSON write, for the same reason as save_npy.

    read_meta() swallows a parse error and returns {}, which is the right call for a
    missing file but means a half-written meta.json degrades silently into an image
    with no dimensions and no status. registry.json matters more still: it holds which
    model is current and the whole retrain history, and write_meta runs on every
    progress update during ingest, so these files are rewritten far more often than
    their importance suggests.
    """
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_meta(image_id, meta):
    cur = read_meta(image_id)
    cur.update(meta)
    write_json(path(image_id, "meta.json"), cur)
    return cur


def original_path(image_id):
    d = path(image_id)
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.startswith("original"):
            return os.path.join(d, f)
    return None


def load_npy(image_id, name, mmap=False):
    p = path(image_id, name)
    if not os.path.exists(p):
        return None
    return np.load(p, mmap_mode="r" if mmap else None)


def save_npy(image_id, name, arr):
    """Write atomically: full file to a temp name, fsync, then rename over the target.

    The app has no save button -- every brush stroke is a write to correction.npy --
    so this path runs constantly while someone works, and it rewrites the WHOLE array
    each time (2.9 MB for a typical image, 23 MB for the big mosaic). Writing in place
    meant a crash or power cut mid-write left a truncated file, and the loss is not the
    one stroke in flight, it is every correction ever painted on that image. rename(2)
    is atomic on the same filesystem, so a reader sees either the old file or the new
    one. Verified: SIGKILL immediately after a stroke, restart, corrections intact.
    """
    d = path(image_id)
    os.makedirs(d, exist_ok=True)
    final = path(image_id, name)
    tmp = final + ".tmp"
    try:
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
            fh.flush()
            os.fsync(fh.fileno())          # without this the rename can beat the data to disk
        os.replace(tmp, final)
    except BaseException:
        # Never leave a stray .tmp behind to be mistaken for real data later.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ------------------------------------------------------------------ undo
# Undo stores a DELTA per stroke, not a snapshot. A correction array is uint8 at
# full image resolution -- 2.9 MB for a 2.9 MP image, 23 MB for the big mosaic --
# so twenty snapshots of the large one would be half a gigabyte. A delta is the
# bounding box of the stroke plus the pixel values that were there before, which
# for a brush stroke is a few hundred kilobytes at most.
UNDO_DEPTH = 30


def _undo_dir(image_id):
    d = path(image_id, "undo")
    os.makedirs(d, exist_ok=True)
    return d


def push_undo(image_id, y0, y1, x0, x1, prev_block):
    """Record what a region looked like BEFORE a stroke overwrote it."""
    d = _undo_dir(image_id)
    n = len(os.listdir(d))
    np.savez_compressed(os.path.join(d, f"{time.time():.6f}_{n}.npz"),
                        box=np.asarray([y0, y1, x0, x1], np.int64), prev=prev_block)
    files = sorted(os.listdir(d))
    for f in files[:-UNDO_DEPTH]:            # keep the stack bounded
        try:
            os.remove(os.path.join(d, f))
        except OSError:
            pass


def pop_undo(image_id):
    """Undo the most recent stroke. Returns True if something was undone."""
    d = _undo_dir(image_id)
    files = sorted(os.listdir(d))
    if not files:
        return False
    p = os.path.join(d, files[-1])
    try:
        z = np.load(p)
        y0, y1, x0, x1 = [int(v) for v in z["box"]]
        prev = z["prev"]
    except Exception:
        os.remove(p)
        return False
    corr = load_npy(image_id, "correction.npy")
    if corr is None:
        os.remove(p)
        return False
    corr = np.asarray(corr).copy()
    if corr[y0:y1, x0:x1].shape == prev.shape:
        corr[y0:y1, x0:x1] = prev
        save_npy(image_id, "correction.npy", corr)
    os.remove(p)
    return True


def undo_depth(image_id):
    d = path(image_id, "undo")
    return len(os.listdir(d)) if os.path.isdir(d) else 0


def clear_undo(image_id):
    d = path(image_id, "undo")
    if os.path.isdir(d):
        shutil.rmtree(d)


def list_images():
    out = []
    for iid in sorted(os.listdir(IMAGES)) if os.path.isdir(IMAGES) else []:
        if not os.path.isdir(path(iid)):
            continue
        m = read_meta(iid)
        if not m:
            continue
        corr = load_npy(iid, "correction.npy", mmap=True)
        n_crack = n_not = 0
        if corr is not None:
            a = np.asarray(corr)
            n_crack, n_not = int((a == 1).sum()), int((a == 2).sum())
            del a
        m.update(has_prob=os.path.exists(path(iid, "prob.npy")),
                 has_emb=os.path.exists(path(iid, "emb.npz")),
                 corrected_crack_px=n_crack, corrected_not_px=n_not)
        out.append(m)
    return out


def delete_image(image_id):
    d = path(image_id)
    if os.path.isdir(d):
        shutil.rmtree(d)
        return True
    return False


# ------------------------------------------------------------- model registry
def _default_registry():
    return dict(current=dict(kind="ensemble",
                             path_17=os.path.join(PROJECT, "models", "pixel_hgb_final.joblib"),
                             path_hybrid=os.path.join(PROJECT, "models", "pixel_sam_hybrid.joblib"),
                             label="shipped baseline",
                             created=None),
                history=[])


def registry():
    if not os.path.exists(REGISTRY):
        r = _default_registry()
        write_json(REGISTRY, r)
        return r
    try:
        with open(REGISTRY) as f:
            return json.load(f)
    except Exception:
        # A registry we cannot parse must not take the app down: fall back to the
        # shipped baseline, which is a real working model, rather than raising on
        # every request. Retrain history is lost, the ability to predict is not.
        return _default_registry()


def set_current(entry):
    r = registry()
    if r.get("current"):
        r.setdefault("history", []).append(r["current"])
    r["current"] = entry
    write_json(REGISTRY, r)
    return r


def rollback():
    """Restore the previous model. Returns True if there was one."""
    r = registry()
    hist = r.get("history") or []
    if not hist:
        return False
    prev = hist.pop()
    r.setdefault("history", []).append(r["current"])
    r["current"] = prev
    r["history"] = hist + [h for h in r["history"] if h is not prev][-20:]
    write_json(REGISTRY, r)
    return True
