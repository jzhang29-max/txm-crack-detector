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
    # Which model the app claims to be using, so each image can be compared against it.
    cur_key = model_key(registry().get("current"))
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
        has_prob = os.path.exists(path(iid, "prob.npy"))
        # A model switch flips the registry first and predicts afterwards, so if that
        # job dies or the app is quit mid-pass, some images still hold the previous
        # model's prediction. Reporting them as plain "ready" is what made that
        # invisible: the picker said model B while the mask was model A's. This says
        # so per image, and is derived from what is on disk rather than from whether
        # a job reported success.
        img_key = m.get("model_key")
        m.update(has_prob=has_prob,
                 has_emb=os.path.exists(path(iid, "emb.npz")),
                 model_key=img_key, current_model_key=cur_key,
                 stale=bool(has_prob and img_key and img_key != cur_key),
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


def model_key(entry):
    """Short stable id for one model, used to name its cached predictions.

    A retrain stamp is unique per model, so it is the natural key. Entries without one
    (the shipped baseline, or a hand-configured entry) are keyed by their files'
    paths AND size+mtime -- not paths alone. Paths alone were wrong: the training
    scripts overwrite models/pixel_sam_hybrid.joblib in place, so a genuinely new
    model kept the old key, every image looked "ready" for it, and the app served the
    previous model's predictions while reporting the new one as current.
    """
    entry = entry or {}
    base = entry.get("created") or ""
    if not base:
        ident = []
        for p in (entry.get("path_17"), entry.get("path_hybrid")):
            try:
                st = os.stat(p)
                ident.append([p, st.st_size, int(st.st_mtime)])
            except (OSError, TypeError):
                ident.append([p, None, None])
        ident.append(entry.get("kind"))
        base = "m" + hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:10]
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(base))[:40]


def set_current(entry, remember=True):
    """Make `entry` the current model.

    remember=False is for switching between models the user already has: the old
    current still goes into history, but the entry being selected is pulled OUT of
    history so it is not listed twice, and re-selecting the same model is a no-op.
    Without this, flipping between two models a few times grew the history list with
    duplicate copies of both.
    """
    r = registry()
    cur = r.get("current")
    if cur and model_key(cur) == model_key(entry):
        return r
    hist = r.setdefault("history", [])
    if cur and not any(model_key(h) == model_key(cur) for h in hist):
        hist.append(cur)
    if not remember:
        r["history"] = [h for h in hist if model_key(h) != model_key(entry)]
    r["current"] = entry
    write_json(REGISTRY, r)
    return r


def available_models():
    """Every model the user can pick: the current one, everything in history, and
    the shipped baseline even if it has never been current.

    Ordered current-first, then newest retrain to oldest, with the unstamped shipped
    baseline last. History order depends on how the user happened to switch around,
    which is not a sensible order to present.
    """
    r = registry()
    others = sorted((r.get("history") or []),
                    key=lambda e: (e.get("created") or ""), reverse=True)
    out, seen = [], set()
    for e in [r.get("current")] + others + [_default_registry()["current"]]:
        if not e:
            continue
        k = model_key(e)
        if k in seen:
            continue
        # Do not offer a model whose files are not all present. CrackModel silently
        # degrades when a path is missing -- drop path_hybrid and it predicts with the
        # 17-feature model alone -- so offering a half-present entry would mean the
        # user picks "retrained X" and gets something else, cached under X's key.
        # Requiring EVERY declared path to exist, not just one of them, is the point.
        declared = [p for p in (e.get("path_17"), e.get("path_hybrid")) if p]
        if not declared or not all(os.path.exists(p) for p in declared):
            continue
        seen.add(k)
        out.append(dict(e, id=k, current=(k == model_key(r.get("current")))))
    return out


# ------------------------------------------------- per-model prediction cache
# Switching models has to feel instant, and a prediction is a pure function of
# (image, model), so it is cacheable forever. prob.npy stays "the current model's
# prediction" -- every other module already reads it -- and is HARD LINKED to the
# cache entry rather than copied, so keeping N models per image costs N predictions
# on disk, not 2N. save_npy writes via temp+rename, which replaces the link target
# instead of writing through it, so a later write can never corrupt a cache entry.
PROB_CACHE_KEEP = 6


def _prob_cache_dir(image_id):
    d = path(image_id, "probs")
    os.makedirs(d, exist_ok=True)
    return d


def prob_cache_path(image_id, key):
    return os.path.join(_prob_cache_dir(image_id), f"{key}.npy")


def has_prob_for(image_id, key):
    return os.path.exists(prob_cache_path(image_id, key))


def _link_or_copy(src, dst):
    """Publish src at dst atomically, sharing the inode where the filesystem allows.

    The staging name is unique per process AND thread. A single fixed "<dst>.tmp" was a
    real hazard: two threads publishing prob.npy at once (a predict job finishing while
    the user picks a model) could have one link its source and the other replace it, so
    prob.npy ended up holding the wrong model's array. Worse, the copy fallback would
    open that shared temp name "wb" while it was still a hard link to a cache entry --
    truncating that entry and poisoning it for every future switch.
    """
    tmp = f"{dst}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        os.link(src, tmp)                       # same inode: no extra disk
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        shutil.copyfile(src, tmp)               # different device, or links unsupported
    os.replace(tmp, dst)


def store_prob(image_id, key, arr):
    """Write a prediction into the cache and make it the live prob.npy."""
    p = prob_cache_path(image_id, key)
    tmp = p + ".tmp"
    try:
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _link_or_copy(p, path(image_id, "prob.npy"))
    _prune_prob_cache(image_id, keep_key=key)
    write_meta(image_id, dict(model_key=key))


def adopt_prob(image_id, key):
    """Point prob.npy at an already-computed prediction. True if the cache had it."""
    p = prob_cache_path(image_id, key)
    if not os.path.exists(p):
        return False
    _link_or_copy(p, path(image_id, "prob.npy"))
    os.utime(p, None)                           # mark as recently used for pruning
    write_meta(image_id, dict(model_key=key))
    return True


def migrate_prob_cache(image_id):
    """Seed the cache from a pre-cache prob.npy, once, and only when we KNOW its author.

    Only migrates when meta.json records which model produced the file. It used to fall
    back to "assume the current model", which is a guess that silently becomes a lie:
    if the file was actually an older model's output, it got filed under the current
    model's key, and every later request adopted it -- so the app would report the new
    model as ready for that image while showing the old model's mask, and no retrain or
    re-apply would ever correct it. An unlabelled file is left alone instead; the cost
    is one honest re-prediction, which is strictly better than a wrong cache entry.
    """
    live = path(image_id, "prob.npy")
    if not os.path.exists(live) or os.listdir(_prob_cache_dir(image_id)):
        return False
    key = read_meta(image_id).get("model_key")
    if not key:
        return False
    _link_or_copy(live, prob_cache_path(image_id, key))
    return True


def _prune_prob_cache(image_id, keep_key=None):
    """Bound the cache: a 23 MP prediction is 94 MB, so this is not free."""
    d = _prob_cache_dir(image_id)
    files = [f for f in os.listdir(d) if f.endswith(".npy")]
    if len(files) <= PROB_CACHE_KEEP:
        return
    files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    for f in files[PROB_CACHE_KEEP:]:
        if keep_key and f == f"{keep_key}.npy":
            continue
        try:
            os.remove(os.path.join(d, f))
        except OSError:
            pass


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
