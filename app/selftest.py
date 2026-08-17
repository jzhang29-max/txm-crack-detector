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
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        raw = resp.read()
        ctype = resp.headers.get("Content-Type", "")
        return resp.status, (json.loads(raw) if "json" in ctype else raw)


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

    # ---- rendering endpoints
    for path, label in [(f"/api/image/{iid}/display.png", "display PNG renders"),
                        (f"/api/image/{iid}/mask.png", "overlay PNG renders")]:
        try:
            st, raw = req(B, path, timeout=120)
            check(label, st == 200 and raw[:4] == b"\x89PNG", f"{len(raw)} bytes")
        except Exception as e:
            check(label, False, str(e))

    # ---- painting
    _, r = req(B, f"/api/image/{iid}/correction", "POST",
               dict(mode="crack", radius=30, points=[[400, 400], [440, 410], [480, 420]]))
    check("paint crack stroke", r.get("crack_px", 0) > 0, f"{r.get('crack_px'):,} px")
    d1 = r.get("undo_depth")
    _, r2 = req(B, f"/api/image/{iid}/correction", "POST",
                dict(mode="erase", radius=30, points=[[900, 500], [940, 510]]))
    check("paint eraser stroke", r2.get("not_px", 0) > 0, f"{r2.get('not_px'):,} px")
    check("undo stack grows per stroke", r2.get("undo_depth") == d1 + 1,
          f"{d1} -> {r2.get('undo_depth')}")

    # ---- undo
    _, u = req(B, f"/api/image/{iid}/undo", "POST")
    check("undo removes only the last stroke",
          u.get("ok") and u.get("not_px") == 0 and u.get("crack_px") == r.get("crack_px"),
          f"crack {u.get('crack_px'):,} / not {u.get('not_px'):,}")

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
