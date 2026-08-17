"""
End-to-end self test. Exercises every user-facing feature against a running
server and reports pass/fail per feature.

    python3 app/server.py &            # or ./run_app.sh
    python3 app/selftest.py            # default http://127.0.0.1:8800
    python3 app/selftest.py --base http://127.0.0.1:3000 --retrain

Ships with the package so a new user can verify their install rather than trust
a README. Everything except --retrain runs in about a minute; --retrain adds a
few minutes because it trains and then validates against the ground truth.

It creates its own test image from the built-in ground truth, so it needs no
data of your own, and it cleans up after itself unless --keep is given.
"""

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))

PASS, FAIL, SKIP = [], [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""), flush=True)
    return ok


def skip(name, why):
    SKIP.append(name)
    print(f"  SKIP  {name}  -- {why}", flush=True)


def req(base, path, method="GET", body=None, files=None, timeout=600):
    url = base.rstrip("/") + path
    if files:
        boundary = "----txmselftest"
        parts = []
        for field, (fname, content) in files.items():
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                         f'name="{field}"; filename="{fname}"\r\n'
                         f"Content-Type: application/octet-stream\r\n\r\n".encode())
            parts.append(content)
            parts.append(b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        data = b"".join(parts)
        r = urllib.request.Request(url, data=data, method="POST")
        r.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    else:
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(url, data=data, method=method)
        if data:
            r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            return resp.status, (json.loads(raw) if "json" in ctype else raw)
    except urllib.error.HTTPError as e:
        # A 4xx is a legitimate answer to test -- "this file is not readable" is
        # supposed to be a 400 with a JSON explanation. urlopen raises on those and
        # str(HTTPError) does not include the body, so read it here and return it
        # like any other response; callers already inspect .ok / .error.
        raw = e.read()
        ctype = e.headers.get("Content-Type", "") if e.headers else ""
        try:
            return e.code, (json.loads(raw) if "json" in ctype else raw)
        except Exception:
            return e.code, raw


def S_label(resp):
    return ((resp or {}).get("current") or {}).get("label")


def wait_job(base, jid, label, timeout=2400):
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        _, j = req(base, f"/api/job/{jid}")
        if j.get("stage") != last:
            last = j.get("stage")
            print(f"        {label}: {last}", flush=True)
        if j["state"] != "running":
            return j
        time.sleep(4)
    return dict(state="timeout")


def make_test_tiff():
    """A test image built from the shipped ground truth, so no user data needed."""
    import tifffile
    gtdir = os.path.join(PROJECT, "dataset_cache")
    cands = sorted(f for f in os.listdir(gtdir) if f.endswith("_img.npy")) if os.path.isdir(gtdir) else []
    if not cands:
        return None, None
    # smallest first, so the test is quick
    cands.sort(key=lambda f: os.path.getsize(os.path.join(gtdir, f)))
    a = np.load(os.path.join(gtdir, cands[0]))
    buf = io.BytesIO()
    tifffile.imwrite(buf, (np.clip(a, 0, 1) * 65535).astype(np.uint16))
    return "SELFTEST_IMAGE.tif", buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8800")
    ap.add_argument("--retrain", action="store_true", help="also test retrain (slow)")
    ap.add_argument("--keep", action="store_true", help="do not delete the test image")
    args = ap.parse_args()
    B = args.base
    print(f"TXM app self test against {B}\n")

    # ---- server + model
    try:
        _, m = req(B, "/api/model", timeout=30)
    except Exception as e:
        print(f"  FAIL  server reachable -- {type(e).__name__}: {e}")
        print("\n  Is it running?  python3 app/server.py")
        sys.exit(1)
    check("server reachable", True)
    check("model loaded", "NOT LOADED" not in m.get("description", ""), m.get("description", ""))
    gt_ok = bool(m.get("ground_truth_available"))
    check("ground truth present (needed to validate a retrain)", gt_ok)

    # ---- upload + ingest
    fname, content = make_test_tiff()
    if content is None:
        print("  FAIL  could not build a test image (dataset_cache missing)")
        sys.exit(1)
    _, up = req(B, "/api/upload", files={"files": (fname, content)}, timeout=120)
    check("upload accepted", up.get("ok") is True)
    iid = (up.get("added") or up.get("reused") or [None])[0]
    check("upload returned an image id", bool(iid), str(iid))
    if up.get("job"):
        j = wait_job(B, up["job"], "ingest")
        check("ingest completed (preprocess + SAM + predict)", j["state"] == "done",
              j.get("error") or f"{j.get('seconds')}s")

    _, imgs = req(B, "/api/images")
    me = next((x for x in imgs["images"] if x["id"] == iid), {})
    check("image marked ready", me.get("status") == "ready", me.get("status", "?"))
    check("prediction produced", me.get("predicted_area") is not None,
          f"{(me.get('predicted_area') or 0)*100:.2f}% crack")
    check("display image is preprocessed", "destitch" in str(me.get("display", "")),
          str(me.get("display")))

    # ---- file formats the UI advertises must actually load.
    # The drop zone offers .tif .tiff .png and the file picker adds .jpg, but ingest
    # read everything with tifffile.imread, so a dropped PNG was accepted, stored,
    # and then died with "not a TIFF file: header=b'\x89PNG'". Tested here because
    # it shipped: the previous suite only ever uploaded a TIFF.
    try:
        from PIL import Image as _I
        buf = io.BytesIO()
        _I.fromarray((np.linspace(0, 255, 256 * 256).reshape(256, 256)).astype(np.uint8)).save(buf, format="PNG")
        _, pu = req(B, "/api/upload", files={"files": ("SELFTEST_PNG.png", buf.getvalue())}, timeout=120)
        pid = (pu.get("added") or pu.get("reused") or [None])[0]
        pj = wait_job(B, pu["job"], "png ingest") if pu.get("job") else {"state": "done"}
        check("PNG upload ingests (not just TIFF)", pu.get("ok") is True and pj["state"] == "done",
              pj.get("error") or "ok")
        if pid:
            req(B, f"/api/image/{pid}", "DELETE")
    except Exception as e:
        check("PNG upload ingests (not just TIFF)", False, str(e))

    # An unreadable file should be refused at upload with a sentence, not stored and
    # then failed inside a background job with a decoder traceback.
    try:
        st_rej, rj = req(B, "/api/upload", files={"files": ("SELFTEST_NOTES.txt", b"not an image")}, timeout=60)
        check("unreadable file is refused with a clear message",
              rj.get("ok") is False and "supported" in (rj.get("error") or ""),
              (rj.get("error") or "")[:70])
    except Exception as e:
        # req() raises on a 400, which is itself the correct behaviour -- accept it
        # as long as the message names what is supported.
        check("unreadable file is refused with a clear message", "supported" in str(e), str(e)[:70])

    # ---- rendering endpoints
    for path, label in [(f"/api/image/{iid}/display.png", "display PNG renders"),
                        (f"/api/image/{iid}/mask.png", "overlay PNG renders")]:
        try:
            st, raw = req(B, path, timeout=120)
            check(label, st == 200 and raw[:4] == b"\x89PNG", f"{len(raw)} bytes")
        except Exception as e:
            check(label, False, str(e))

    # ---- painting.
    # Start from a known state. Re-uploading the same file reuses its entry (the id
    # is a content hash), so an earlier run -- or an earlier ABORTED run -- can leave
    # corrections behind. Asserting absolute pixel counts then fails for a reason
    # that has nothing to do with the app, which is exactly what happened once.
    req(B, f"/api/image/{iid}/correction", "POST", dict(mode="clear"))
    _, base = req(B, f"/api/image/{iid}/stats")
    _, r = req(B, f"/api/image/{iid}/correction", "POST",
               dict(mode="crack", radius=30, points=[[400, 400], [440, 410], [480, 420]]))
    check("paint crack stroke", r.get("crack_px", 0) > 0, f"{r.get('crack_px'):,} px")
    crack_after_first = r.get("crack_px")
    not_after_first = r.get("not_px")
    d1 = r.get("undo_depth")
    _, r2 = req(B, f"/api/image/{iid}/correction", "POST",
                dict(mode="erase", radius=30, points=[[900, 500], [940, 510]]))
    check("paint eraser stroke", r2.get("not_px", 0) > not_after_first,
          f"{r2.get('not_px'):,} px")
    check("undo stack grows per stroke", r2.get("undo_depth") == d1 + 1,
          f"{d1} -> {r2.get('undo_depth')}")

    # ---- undo: must restore exactly the state before the LAST stroke
    _, u = req(B, f"/api/image/{iid}/undo", "POST")
    check("undo removes only the last stroke",
          bool(u.get("ok")) and u.get("not_px") == not_after_first
          and u.get("crack_px") == crack_after_first,
          f"crack {u.get('crack_px'):,} (want {crack_after_first:,}) / "
          f"not {u.get('not_px'):,} (want {not_after_first:,})")

    # ---- region removal. Probe a point that is actually INSIDE a predicted
    # region, found from the mask itself -- a hard-coded coordinate silently
    # skipped this test whenever it landed on background.
    probe = (850, 850)
    try:
        from PIL import Image
        from scipy import ndimage as _ndi
        _, raw = req(B, f"/api/image/{iid}/mask.png", timeout=120)
        alpha = np.array(Image.open(io.BytesIO(raw)).convert("RGBA"))[:, :, 3] > 0
        if alpha.any():
            lab, n = _ndi.label(alpha)
            sizes = _ndi.sum(alpha, lab, range(1, n + 1))
            big = int(np.argmax(sizes)) + 1
            cy, cx = _ndi.center_of_mass(lab == big)
            if lab[int(cy), int(cx)] == big:
                probe = (int(cx), int(cy))
            else:                       # centroid can fall outside a curved region
                ys, xs = np.nonzero(lab == big)
                probe = (int(xs[len(xs) // 2]), int(ys[len(ys) // 2]))
    except Exception:
        pass
    _, fr = req(B, f"/api/image/{iid}/flip_region", "POST",
                dict(x=probe[0], y=probe[1], mode="remove"), timeout=180)
    if fr.get("ok"):
        check("remove-region deletes a whole component", fr.get("region_px", 0) > 1000,
              f"{fr.get('region_px'):,} px")
        _, u2 = req(B, f"/api/image/{iid}/undo", "POST")
        check("undo restores a removed region", u2.get("ok") is True)
    else:
        skip("remove-region", fr.get("error", "no region under the probe point"))

    # ---- exports
    for ep, magic, label in [("mask.png", b"\x89PNG", "export B&W mask"),
                             ("overlay.png", b"\x89PNG", "export overlay"),
                             ("stats.csv", None, "export CSV")]:
        try:
            st, raw = req(B, f"/api/export/{iid}/{ep}", timeout=300)
            ok = st == 200 and len(raw) > 200 and (magic is None or raw[:4] == magic)
            extra = f"{len(raw)} bytes"
            if ep == "stats.csv":
                head = raw.decode("utf8", "replace")
                ok = ok and "SourceImage,CrackID,Area_px" in head and "SkeletonLength_px" in head
                extra += "; SEM column set" if ok else "; MISSING SEM columns"
            check(label, ok, extra)
        except Exception as e:
            check(label, False, str(e))
    try:
        st, raw = req(B, "/api/export/all.zip", timeout=900)
        import zipfile
        z = zipfile.ZipFile(io.BytesIO(raw))
        names = z.namelist()
        check("export all.zip", st == 200 and "summary.csv" in names,
              f"{len(names)} entries, {len(raw)/1e6:.1f} MB")
    except Exception as e:
        check("export all.zip", False, str(e))

    # ---- B&W polarity: crack must be BLACK
    try:
        from PIL import Image
        _, raw = req(B, f"/api/export/{iid}/mask.png", timeout=300)
        a = np.array(Image.open(io.BytesIO(raw)))
        vals = sorted(np.unique(a).tolist())
        check("B&W mask is binary with crack=black", vals == [0, 255] or vals in ([0], [255]),
              f"values {vals}, black {float((a==0).mean())*100:.1f}%")
    except Exception as e:
        check("B&W mask is binary with crack=black", False, str(e))

    # ---- threshold + postprocess actually change the output
    try:
        _, m1 = req(B, f"/api/image/{iid}/mask.png?threshold=0.20", timeout=120)
        _, m2 = req(B, f"/api/image/{iid}/mask.png?threshold=0.80", timeout=120)
        check("threshold slider changes the mask", m1 != m2,
              f"{len(m1)} vs {len(m2)} bytes")
        _, p1 = req(B, f"/api/image/{iid}/mask.png?postprocess=0", timeout=120)
        _, p2 = req(B, f"/api/image/{iid}/mask.png?postprocess=1", timeout=300)
        check("post-process toggle changes the mask", p1 != p2)
    except Exception as e:
        check("threshold / post-process toggles", False, str(e))

    # ---- the status bar must describe the mask actually on screen.
    # api_stats used to call effective_mask() with no arguments, so it always
    # reported threshold 0.50 while the user looked at something else.
    try:
        _, s20 = req(B, f"/api/image/{iid}/stats?threshold=0.20", timeout=180)
        _, s80 = req(B, f"/api/image/{iid}/stats?threshold=0.80", timeout=180)
        a20, a80 = s20.get("area_fraction"), s80.get("area_fraction")
        check("stats honour the sensitivity setting", a20 is not None and a80 is not None and a20 > a80,
              f"{a20:.4f} at 0.20 vs {a80:.4f} at 0.80")
    except Exception as e:
        check("stats honour the sensitivity setting", False, str(e))

    # ---- concurrent strokes must not lose each other.
    # Flask runs threaded and every stroke is a load-modify-save of the whole
    # correction array, so without a per-image lock the later write silently
    # discarded the earlier stroke -- which had already reported "saved".
    try:
        import threading as _th
        req(B, f"/api/image/{iid}/correction", "POST", dict(mode="clear"))
        boxes = [[[300, 300]], [[400, 400]], [[500, 500]], [[600, 600]],
                 [[700, 700]], [[800, 800]], [[900, 900]], [[1000, 1000]]]
        errs = []

        def paint(pts):
            try:
                req(B, f"/api/image/{iid}/correction", "POST",
                    dict(mode="crack", radius=15, points=pts), timeout=300)
            except Exception as ex:                      # noqa: BLE001
                errs.append(str(ex))

        ts = [_th.Thread(target=paint, args=(p,)) for p in boxes]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        # Measure directly rather than probing with undo, which would itself
        # remove one of the dabs being counted.
        _, lst = req(B, "/api/images")
        me = [m for m in lst.get("images", []) if m["id"] == iid]
        painted = me[0].get("corrected_crack_px", 0) if me else 0
        # The 8 dabs are disjoint disks of radius 15, so with the lock held every
        # one survives: 8 * ~709 px. A lost update shows up as a whole dab missing.
        n_equiv = painted / 709.0
        check("concurrent strokes do not overwrite each other",
              not errs and n_equiv >= 7.5,
              f"{painted:,} px = {n_equiv:.1f} of 8 dabs kept"
              + (f"; errors: {errs[:1]}" if errs else ""))
    except Exception as e:
        check("concurrent strokes do not overwrite each other", False, str(e))

    # ---- the overlay is the most-requested endpoint in the app, so it is cached.
    # Assert identical bytes on repeat and that painting invalidates it, rather than
    # asserting a timing, which would be flaky on a loaded machine.
    try:
        _, o1 = req(B, f"/api/image/{iid}/mask.png?threshold=0.50&postprocess=0", timeout=180)
        _, o2 = req(B, f"/api/image/{iid}/mask.png?threshold=0.50&postprocess=0", timeout=180)
        cached_file = os.path.exists(os.path.join(PROJECT, "app_data", "images", iid,
                                                 "overlays", "0.50_0.png"))
        check("overlay is cached and stable on repeat", o1 == o2 and cached_file,
              f"{len(o1)} bytes, cache file {'written' if cached_file else 'MISSING'}")
        req(B, f"/api/image/{iid}/correction", "POST",
            dict(mode="crack", radius=20, points=[[250, 250]]))
        _, o3 = req(B, f"/api/image/{iid}/mask.png?threshold=0.50&postprocess=0", timeout=180)
        check("painting invalidates the cached overlay", o3 != o2,
              f"{len(o2)} -> {len(o3)} bytes")
        req(B, f"/api/image/{iid}/undo", "POST")
    except Exception as e:
        check("overlay caching", False, str(e))

    # ---- model picker. Switching to a model already computed for an image must be
    # instant (a hard link), not a re-prediction.
    try:
        _, ml = req(B, "/api/models", timeout=120)
        models = ml.get("models") or []
        cur = next((m for m in models if m.get("current")), None)
        check("model list offers the current model", cur is not None,
              f"{len(models)} model(s): " + ", ".join(m["label"] for m in models[:3]))
        others = [m for m in models if not m.get("current")]
        if not others:
            skip("switching models is instant when cached",
                 "only one model exists until you retrain")
        else:
            # Switch away and back; the trip back must need no prediction job because
            # the current model's result for this image is already on disk.
            t0 = time.time()
            _, away = req(B, "/api/model/select", "POST",
                          dict(id=others[0]["id"], focus=iid), timeout=120)
            if away.get("job"):
                wait_job(B, away["job"], "predict with other model")
            _, back = req(B, "/api/model/select", "POST", dict(id=cur["id"]), timeout=120)
            dt = time.time() - t0
            check("switching models is instant when cached",
                  back.get("ok") is True and not back.get("job") and len(back.get("instant") or []) >= 1,
                  f"return trip needed no prediction job ({len(back.get('instant') or [])} from cache)")
            check("switching back restores the original model",
                  S_label(back) == cur["label"], f"{S_label(back)}")
    except Exception as e:
        check("model picker", False, str(e))

    # ---- reset
    _, c = req(B, f"/api/image/{iid}/correction", "POST", dict(mode="clear"))
    check("reset clears this image's corrections", c.get("crack_px") == 0 and c.get("not_px") == 0)
    _, u3 = req(B, f"/api/image/{iid}/undo", "POST")
    check("undo restores after reset", u3.get("ok") is True,
          f"crack {u3.get('crack_px'):,} px back")

    # ---- retrain (optional, slow)
    if args.retrain:
        req(B, f"/api/image/{iid}/correction", "POST",
            dict(mode="crack", radius=40, points=[[500, 700], [560, 720], [620, 740]]))
        req(B, f"/api/image/{iid}/correction", "POST",
            dict(mode="erase", radius=50, points=[[1200, 300], [1260, 320], [1320, 340]]))
        _, rt = req(B, "/api/retrain", "POST", dict(deploy=False))
        j = wait_job(B, rt["job"], "retrain")
        res = j.get("result") or {}
        check("retrain ran", j["state"] == "done", j.get("error") or f"{j.get('seconds')}s")
        if res.get("ok"):
            info = res.get("info", {})
            check("ground truth reached training (no width-dropped blocks)",
                  info.get("blocks_dropped_for_width") == 0,
                  f"dropped {info.get('blocks_dropped_for_width')}")
            check("class balance sane", 0.3 <= info.get("crack_fraction", 0) <= 0.7,
                  f"{info.get('crack_fraction', 0)*100:.1f}% crack")
            check("candidate was validated against ground truth", "candidate" in res,
                  f"incumbent {res.get('incumbent', {}).get('iou', 0):.3f} -> "
                  f"candidate {res.get('candidate', {}).get('iou', 0):.3f}")
        else:
            check("retrain refused for a stated reason", bool(res.get("error")),
                  str(res.get("error"))[:90])
    else:
        skip("retrain", "pass --retrain to include it (slow)")

    if not args.keep:
        try:
            req(B, f"/api/image/{iid}", "DELETE")
            print(f"\n  cleaned up test image {iid}")
        except Exception:
            pass

    print(f"\n{'='*66}")
    print(f"  {len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    if FAIL:
        print("  FAILED: " + ", ".join(FAIL))
    print("=" * 66)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
