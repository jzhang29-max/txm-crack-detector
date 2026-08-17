"""
TXM Crack Detection — app server.

    python3 app/server.py            then open http://127.0.0.1:8800

Drag TXM images in, the current model predicts them, paint corrections, press
Retrain. No paths to edit, no scripts to run in order, no backend knowledge.

Long operations (ingest, retrain) run on a background thread and report progress
through /api/job/<id>, so the browser never blocks on a 20-second SAM pass.
"""

import io
import os
import sys
import threading
import time
import traceback
import uuid

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(HERE, "core"), os.path.join(PROJECT, "code")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pipeline as P            # noqa: E402
import store as S               # noqa: E402

app = Flask(__name__, static_folder=os.path.join(HERE, "static"))
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024   # 2 GB per upload batch

JOBS = {}
_LOCK = threading.Lock()


def _job(fn, label):
    """Run fn(report) on a background thread; return a job id to poll."""
    jid = uuid.uuid4().hex[:12]
    JOBS[jid] = dict(id=jid, label=label, state="running", stage="starting",
                     k=0, n=0, started=time.time(), result=None, error=None)

    def report(stage, k=0, n=0):
        JOBS[jid].update(stage=str(stage), k=int(k), n=int(n))

    def run():
        try:
            with _LOCK:                     # one heavy job at a time; they share the GPU
                out = fn(report)
            JOBS[jid].update(state="done", result=out, stage="done",
                             seconds=round(time.time() - JOBS[jid]["started"], 1))
        except Exception as e:
            JOBS[jid].update(state="error", error=f"{type(e).__name__}: {e}",
                             trace=traceback.format_exc()[-1500:])
    threading.Thread(target=run, daemon=True).start()
    return jid


# ------------------------------------------------------------------- static
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ------------------------------------------------------------------- images
@app.route("/api/model")
def api_model():
    r = S.registry()
    try:
        desc = P.get_model().describe()
    except Exception as e:
        desc = f"NOT LOADED: {type(e).__name__}: {e}"
    return jsonify(current=r["current"], description=desc,
                   history=len(r.get("history") or []),
                   ground_truth_available=P._gt_available())


@app.route("/api/images")
def api_images():
    return jsonify(images=S.list_images())


@app.route("/api/upload", methods=["POST"])
def api_upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify(ok=False, error="no files in request"), 400
    added, reused = [], []
    for f in files:
        content = f.read()
        if not content:
            continue
        iid, is_new = S.save_upload(f.filename, content)
        (added if is_new else reused).append(iid)
    todo = added + [i for i in reused if not os.path.exists(S.path(i, "prob.npy"))]

    def work(report):
        out = []
        for k, iid in enumerate(todo, 1):
            report(f"{iid} ({k}/{len(todo)})", k, len(todo))
            P.ingest(iid, progress=lambda st, a, b: report(f"{iid}: {st}", a, b))
            out.append(iid)
        return dict(processed=out)

    jid = _job(work, f"ingest {len(todo)} image(s)") if todo else None
    return jsonify(ok=True, added=added, reused=reused, job=jid)


@app.route("/api/image/<iid>/display.png")
def api_display(iid):
    which = request.args.get("view", "display")
    arr = S.load_npy(iid, "display.npy" if which == "display" else "img.npy")
    if arr is None:
        return jsonify(ok=False, error="not ingested"), 404
    from PIL import Image
    a = (np.clip(np.asarray(arr), 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG", optimize=False, compress_level=1)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/image/<iid>/thumb.png")
def api_thumb(iid):
    """Small preview with the crack burned in, for the sidebar list.

    Exists because the list was pointing its <img> at display.png: a 2.5 MB
    full-resolution PNG decoded down to a 38 px box, once per row. At 71 images
    that is ~180 MB of transfer to draw a sidebar. This is ~6 KB and shows the
    result rather than just the image, so a row is informative at a glance.
    Cached on disk since it only changes when the prediction or corrections do.
    """
    from PIL import Image
    W = int(request.args.get("w", 128))
    cache = S.path(iid, f"thumb_{W}.png")
    corr_p = S.path(iid, "correction.npy")
    prob_p = S.path(iid, "prob.npy")
    fresh = (os.path.exists(cache)
             and all(not os.path.exists(p) or os.path.getmtime(p) <= os.path.getmtime(cache)
                     for p in (corr_p, prob_p)))
    if fresh:
        return send_file(cache, mimetype="image/png")

    disp = S.load_npy(iid, "display.npy")
    if disp is None:
        disp = S.load_npy(iid, "img.npy")
    if disp is None:
        return jsonify(ok=False, error="not ingested"), 404
    g = (np.clip(np.asarray(disp), 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(g).convert("RGB")
    mask = P.effective_mask(iid)
    if mask is not None and mask.shape == g.shape:
        red = Image.new("RGB", im.size, (230, 60, 55))
        im = Image.composite(Image.blend(im, red, 0.55), im,
                             Image.fromarray((mask * 255).astype(np.uint8)))
    h = max(1, round(W * im.size[1] / im.size[0]))
    im = im.resize((W, h), Image.LANCZOS)
    im.save(cache, format="PNG", optimize=True)
    return send_file(cache, mimetype="image/png")


@app.route("/api/image/<iid>/mask.png")
def api_mask(iid):
    thr = float(request.args.get("threshold", 0.5))
    pp = request.args.get("postprocess", "0") in ("1", "true", "True")
    mask = P.effective_mask(iid, threshold=thr, postprocess=pp)
    if mask is None:
        return jsonify(ok=False, error="no prediction"), 404
    from PIL import Image
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[mask] = (230, 40, 40, 140)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", compress_level=1)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/image/<iid>/stats")
def api_stats(iid):
    m = S.read_meta(iid)
    mask = P.effective_mask(iid)
    if mask is not None:
        from skimage.measure import label
        m = dict(m, area_fraction=float(mask.mean()),
                 n_regions=int(label(mask, connectivity=2).max()))
    return jsonify(m)


@app.route("/api/image/<iid>/correction", methods=["POST"])
def api_correction(iid):
    """Apply brush strokes. Body: {mode:'crack'|'erase'|'clear', points:[[x,y],...], radius:int}"""
    d = request.get_json(force=True, silent=True) or {}
    corr = S.load_npy(iid, "correction.npy")
    if corr is None:
        return jsonify(ok=False, error="not ingested"), 404
    corr = np.asarray(corr).copy()
    H, W = corr.shape
    if d.get("mode") == "clear":
        S.push_undo(iid, 0, H, 0, W, corr.copy())
        corr[:] = 0
    else:
        val = 1 if d.get("mode") == "crack" else 2
        r = max(1, int(d.get("radius", 20)))
        pts = d.get("points", [])
        if not pts:
            return jsonify(ok=True, crack_px=int((corr == 1).sum()),
                           not_px=int((corr == 2).sum()), undo_depth=S.undo_depth(iid))
        # One undo entry per STROKE (mouse-down to mouse-up), not per dot, so
        # Cmd+Z reverses what feels like one action.
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        by0 = max(0, int(min(ys)) - r - 1); by1 = min(H, int(max(ys)) + r + 2)
        bx0 = max(0, int(min(xs)) - r - 1); bx1 = min(W, int(max(xs)) + r + 2)
        if by0 < by1 and bx0 < bx1:
            S.push_undo(iid, by0, by1, bx0, bx1, corr[by0:by1, bx0:bx1].copy())
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk = (xx * xx + yy * yy) <= r * r
        for pt in pts:
            x, y = int(round(pt[0])), int(round(pt[1]))
            y0, y1 = max(0, y - r), min(H, y + r + 1)
            x0, x1 = max(0, x - r), min(W, x + r + 1)
            if y0 >= y1 or x0 >= x1:
                continue
            sub = disk[(y0 - (y - r)):(y1 - (y - r)), (x0 - (x - r)):(x1 - (x - r))]
            corr[y0:y1, x0:x1][sub] = val
    S.save_npy(iid, "correction.npy", corr)
    return jsonify(ok=True, crack_px=int((corr == 1).sum()), not_px=int((corr == 2).sum()),
                   undo_depth=S.undo_depth(iid))


@app.route("/api/image/<iid>/flip_region", methods=["POST"])
def api_flip_region(iid):
    """Click-to-remove an ENTIRE predicted region in one action.

    Borrowed from the SEM pipeline, where it is documented as the thing that made
    correction practical at all: brushing over every pixel of one connected
    component that spans most of the frame is not feasible, and those large
    components are exactly the false positives worth removing. Body:
    {x, y, mode:'remove'|'confirm'}.
    """
    from skimage.measure import label
    d = request.get_json(force=True, silent=True) or {}
    x, y = int(round(d.get("x", -1))), int(round(d.get("y", -1)))
    mode = d.get("mode", "remove")
    thr = float(d.get("threshold", 0.5))
    pp = bool(d.get("postprocess", False))

    mask = P.effective_mask(iid, threshold=thr, postprocess=pp)
    corr = S.load_npy(iid, "correction.npy")
    if mask is None or corr is None:
        return jsonify(ok=False, error="not ingested"), 404
    H, W = mask.shape
    if not (0 <= y < H and 0 <= x < W):
        return jsonify(ok=False, error="click outside image"), 400
    if not mask[y, x]:
        return jsonify(ok=False, error="no predicted region under that point"), 200

    lab = label(mask, connectivity=2)
    target = lab == lab[y, x]
    n = int(target.sum())
    ys, xs = np.nonzero(target)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    corr = np.asarray(corr).copy()
    S.push_undo(iid, y0, y1, x0, x1, corr[y0:y1, x0:x1].copy())
    corr[target] = 2 if mode == "remove" else 1
    S.save_npy(iid, "correction.npy", corr)
    return jsonify(ok=True, region_px=n, mode=mode,
                   crack_px=int((corr == 1).sum()), not_px=int((corr == 2).sum()),
                   undo_depth=S.undo_depth(iid))


@app.route("/api/image/<iid>/undo", methods=["POST"])
def api_undo(iid):
    ok = S.pop_undo(iid)
    corr = S.load_npy(iid, "correction.npy")
    a = np.asarray(corr) if corr is not None else np.zeros((1, 1), np.uint8)
    return jsonify(ok=ok, crack_px=int((a == 1).sum()), not_px=int((a == 2).sum()),
                   undo_depth=S.undo_depth(iid))


@app.route("/api/image/<iid>", methods=["DELETE"])
def api_delete(iid):
    return jsonify(ok=S.delete_image(iid))


# ------------------------------------------------------------------ actions
@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    d = request.get_json(force=True, silent=True) or {}
    deploy = bool(d.get("deploy", True))
    return jsonify(ok=True, job=_job(lambda rep: P.retrain(deploy=deploy, progress=rep),
                                    "retrain"))


@app.route("/api/reoverlay", methods=["POST"])
def api_reoverlay():
    """Re-predict every image with the current model, reusing cached embeddings."""
    ids = [m["id"] for m in S.list_images()]

    def work(report):
        done = []
        for k, iid in enumerate(ids, 1):
            report(f"{iid} ({k}/{len(ids)})", k, len(ids))
            P.ingest(iid, progress=lambda st, a, b: report(f"{iid}: {st}", a, b))
            done.append(iid)
        return dict(reoverlayed=done)
    return jsonify(ok=True, job=_job(work, f"re-overlay {len(ids)} image(s)"))


@app.route("/api/rollback", methods=["POST"])
def api_rollback():
    ok = S.rollback()
    P._model_cache["key"] = None
    return jsonify(ok=ok, current=S.registry()["current"])


@app.route("/api/job/<jid>")
def api_job(jid):
    j = JOBS.get(jid)
    if not j:
        return jsonify(ok=False, error="unknown job"), 404
    return jsonify(j)


# ------------------------------------------------------------------ exports
def _opts():
    return (float(request.args.get("threshold", 0.5)),
            request.args.get("postprocess", "0") in ("1", "true", "True"))


def _mask_png_bytes(mask):
    """Crack = BLACK on white, matching this project's existing B&W outputs."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _overlay_png_bytes(iid, mask):
    """The display image with the mask burned in as translucent red -- what you
    see on screen, as a flat RGB image you can drop in a slide."""
    from PIL import Image
    disp = S.load_npy(iid, "display.npy")
    if disp is None:
        disp = S.load_npy(iid, "img.npy")
    g = (np.clip(np.asarray(disp), 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([g] * 3, -1).astype(np.float32)
    red = np.array([230.0, 40.0, 40.0], np.float32)
    a = 0.45
    rgb[mask] = (1 - a) * rgb[mask] + a * red
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _stats_csv_bytes(iid, mask):
    """Per-image summary plus one row per connected crack region.

    Columns are deliberately IDENTICAL to the SEM pipeline's
    crack_measurements.py output, using the same vendored
    crack_shape_measurements() -- so a reader can put a TXM and an SEM CSV side
    by side, and so a measurement only has to be defined once. Those quantities
    (skeleton length, mean/max width from the medial-axis radius, tortuosity,
    branch points, boundary roughness) are what a materials reader wants;
    generic bounding boxes and eccentricity, which this used to emit, are not.
    """
    import csv
    from skimage import measure as skmeasure
    m = S.read_meta(iid)
    groups = skmeasure.label(mask, connectivity=2)
    n_groups = int(groups.max())
    img_area = mask.size

    try:
        sys.path.insert(0, os.path.join(PROJECT, "code"))
        from sem_crack_measurements import crack_shape_measurements
    except Exception:
        crack_shape_measurements = None

    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["# TXM Crack Detection export"])
    w.writerow(["# image", m.get("filename", iid)])
    w.writerow(["# id", iid])
    w.writerow(["# model", m.get("model", "?")])
    w.writerow(["# display", m.get("display", "?")])
    w.writerow(["# corrections_applied", "yes (force-crack and force-not-crack "
                                        "strokes are baked into this mask)"])
    w.writerow(["# measurement_columns", "same definitions as the SEM pipeline's "
                                        "crack_measurements.py"])
    w.writerow([])
    w.writerow(["height_px", "width_px", "megapixels", "crack_px",
                "crack_area_fraction", "n_regions"])
    w.writerow([mask.shape[0], mask.shape[1], round(mask.size / 1e6, 3),
                int(mask.sum()), round(float(mask.mean()), 6), n_groups])
    w.writerow([])

    cols = ["SourceImage", "CrackID", "Area_px", "AreaPct_of_image",
            "SkeletonLength_px", "MeanWidth_px", "MaxWidth_px", "Tortuosity",
            "BranchPointCount", "MajorAxisLength_px", "MinorAxisLength_px",
            "Orientation_deg", "BoundaryRoughness", "CentroidX_px", "CentroidY_px"]
    w.writerow(cols)
    rows = []
    for gid in range(1, n_groups + 1):
        sub = groups == gid
        n = int(sub.sum())
        if n < 5:
            continue          # skeletonize/regionprops need a few px to mean anything
        if crack_shape_measurements is not None:
            d = crack_shape_measurements(sub)
        else:
            d = dict(Area_px=n)
        ys, xs = np.nonzero(sub)
        d.update(SourceImage=m.get("filename", iid), CrackID=gid,
                 AreaPct_of_image=round(100.0 * n / img_area, 4),
                 CentroidX_px=round(float(xs.mean()), 1),
                 CentroidY_px=round(float(ys.mean()), 1))
        rows.append(d)
    for d in sorted(rows, key=lambda r: -r.get("Area_px", 0)):
        w.writerow([d.get(c, "") for c in cols])
    return out.getvalue().encode()


@app.route("/api/export/<iid>/mask.png")
def api_export_mask(iid):
    t, pp = _opts()
    mask = P.effective_mask(iid, threshold=t, postprocess=pp)
    if mask is None:
        return jsonify(ok=False, error="no prediction"), 404
    return send_file(io.BytesIO(_mask_png_bytes(mask)), mimetype="image/png",
                     as_attachment=True, download_name=f"{iid}_crack_mask.png")


@app.route("/api/export/<iid>/overlay.png")
def api_export_overlay(iid):
    t, pp = _opts()
    mask = P.effective_mask(iid, threshold=t, postprocess=pp)
    if mask is None:
        return jsonify(ok=False, error="no prediction"), 404
    return send_file(io.BytesIO(_overlay_png_bytes(iid, mask)), mimetype="image/png",
                     as_attachment=True, download_name=f"{iid}_overlay.png")


@app.route("/api/export/<iid>/stats.csv")
def api_export_stats(iid):
    t, pp = _opts()
    mask = P.effective_mask(iid, threshold=t, postprocess=pp)
    if mask is None:
        return jsonify(ok=False, error="no prediction"), 404
    return send_file(io.BytesIO(_stats_csv_bytes(iid, mask)), mimetype="text/csv",
                     as_attachment=True, download_name=f"{iid}_stats.csv")


@app.route("/api/export/all.zip")
def api_export_all():
    """Every ready image: mask PNG + overlay PNG + stats CSV, plus one summary CSV."""
    import csv
    import zipfile
    t, pp = _opts()
    imgs = [m for m in S.list_images() if m.get("has_prob")]
    if not imgs:
        return jsonify(ok=False, error="nothing predicted yet"), 404
    buf = io.BytesIO()
    summary = io.StringIO()
    sw = csv.writer(summary)
    sw.writerow(["image", "id", "height_px", "width_px", "megapixels",
                 "crack_px", "crack_area_fraction", "n_regions",
                 "corrected_crack_px", "corrected_not_px"])
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for m in imgs:
            iid = m["id"]
            mask = P.effective_mask(iid, threshold=t, postprocess=pp)
            if mask is None:
                continue
            stem = os.path.splitext(m.get("filename") or iid)[0]
            z.writestr(f"{stem}/{stem}_crack_mask.png", _mask_png_bytes(mask))
            z.writestr(f"{stem}/{stem}_overlay.png", _overlay_png_bytes(iid, mask))
            z.writestr(f"{stem}/{stem}_stats.csv", _stats_csv_bytes(iid, mask))
            from skimage.measure import label
            sw.writerow([m.get("filename"), iid, mask.shape[0], mask.shape[1],
                         round(mask.size / 1e6, 3), int(mask.sum()),
                         round(float(mask.mean()), 6),
                         int(label(mask, connectivity=2).max()),
                         m.get("corrected_crack_px", 0), m.get("corrected_not_px", 0)])
            del mask
        z.writestr("summary.csv", summary.getvalue())
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name="txm_crack_export.zip")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8800"))
    print(f"\n  TXM Crack Detection")
    print(f"  open  http://127.0.0.1:{port}\n", flush=True)
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
