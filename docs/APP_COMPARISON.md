# TXM app vs SEM app — what each does better, and the shared layout

Two sibling projects, two correction apps, built months apart. This is the
comparison and what was moved in each direction as a result.

## Verdict

**Neither was strictly better.** SEM had the better *packaging and correction
ergonomics*; TXM had the better *outputs, controls and safety*. Both gaps were
real and both have now been closed by copying the winner in each category rather
than defending either original.

| capability | SEM (before) | TXM (before) | now |
|---|---|---|---|
| one-command launch with venv bootstrap | ✅ `run_app.sh` | ❌ manual `pip install` | both ✅ |
| slim distributable repo | ✅ `make_package.sh` | ❌ | SEM ✅, TXM documented |
| **click whole region to remove it** | ✅ `flip_region` | ❌ brush only | both ✅ |
| drag-and-drop upload | ✅ | ✅ | both ✅ |
| ⌘Z undo | ✅ | ✅ (added) | both ✅ |
| job progress bar | ✅ | ✅ | both ✅ |
| **B&W mask / overlay / CSV export** | ❌ none in-app | ✅ | TXM ✅ |
| **all-images .zip with summary CSV** | ❌ | ✅ | TXM ✅ |
| **live threshold slider** | ❌ fixed | ✅ | TXM ✅ |
| **post-processing toggle** | ❌ always on | ✅ off by default | TXM ✅ |
| **model registry + rollback** | ❌ | ✅ | TXM ✅ |
| **retrain validation gate** | partial | ✅ refuses regressions | TXM ✅ |
| self-contained data directory | ❌ reads `original/` | ✅ `app_data/` | TXM ✅ |
| automatic preprocessing on upload | ❌ | ✅ destitch+flatfield | TXM ✅ |

## What TXM took from SEM

**1. `run_app.sh`.** Creates a virtualenv on first run, installs dependencies only
when `requirements.txt` changes (stamp file), creates the directories, warns if
PyTorch is missing instead of crashing on first predict, and execs the server.
TXM previously required the user to know to `pip install -r requirements.txt` and
to set `KMP_DUPLICATE_LIB_OK=TRUE` — which is not optional, because scikit-learn
and torch each vendor an OpenMP runtime and loading both aborts on macOS without
it. That is exactly the kind of thing a launcher should hide.

**2. Click-to-remove-region.** SEM's `flip_region` docstring explains why it
exists: "brushing over every one of its pixels by hand — important when that
region is huge (e.g. one connected component spanning most of the image's dark
background, which is exactly what made painting-only correction impractical for
large regions)". Measured on a TXM image after porting it: one click removed a
791,590-pixel region and took the frame from 30.12% to 2.72% predicted crack.
Brushing that by hand at a 24 px radius would have been several hundred strokes.
It pushes a single undo entry, so ⌘Z restores the whole region — verified.

## What SEM should take from TXM

**1. Exports.** SEM has no in-app download at all: masks and overlays are written
by separate scripts run from a terminal. TXM offers per-image B&W mask, burned-in
overlay, and a CSV carrying one row per crack region (area, bbox, major/minor
axis, eccentricity, orientation, centroid), plus an all-images `.zip` with a
cross-image `summary.csv`. For a user whose next step is measuring crack growth,
that region table is the actual product.

**2. Threshold and post-processing as live controls.** SEM fixes both. TXM exposes
the threshold as a slider and post-processing as a toggle, defaulting **off** —
because post-processing measurably deletes thin crack: on one ground-truth image
it costs 0.084 IoU and 0.072 recall, and recall against hand-painted strokes
drops from ~0.87 at a raw threshold to 0.14–0.40 after it. A fixed pipeline hides
that; a toggle lets you see it.

**3. The retrain gate, and refusing rather than warning.** TXM validates a
candidate on ground truth and will not deploy a regression. Two bugs found by
actually running it are worth recording, because both were silent:
  - The ground-truth blocks are 17-dimensional while corrections are
    273-dimensional (they carry SAM features). The gather step kept only the wider
    ones, so **the ground truth never reached training** — producing a 100%-crack
    training set and a model scoring IoU 0.003. Fixed by embedding the ground
    truth too.
  - A degenerate class balance only printed a warning. It now refuses to train,
    because spending five minutes to produce an impossible model and then
    reporting it as "a regression" buries the real cause.

**4. A model registry with history and one-click rollback**, so a bad retrain is
reversible without finding files by hand.

## The shared layout

Both apps now present the same thing in the same order:

```
┌ sidebar ─────────────┐┌ main ────────────────────────────────────────────┐
│ title                ││ TOOLS      Add crack | Eraser | Remove region    │
│ drag-and-drop zone   ││ VIEW       Brush · Zoom · Fit · Overlay          │
│ MODEL                ││ OPERATING  Post-process · Threshold              │
│   description        ││ EDIT       Undo ⌘Z · Clear corrections           │
│   Retrain            ││ DOWNLOAD   B&W mask · Overlay · CSV · All .zip   │
│   Re-overlay all     ││─────────────────────────────────────────────────│
│   Roll back          ││ progress bar                                     │
│ IMAGES               ││ canvas (image · overlay · paint layer)           │
│   per-image list     ││─────────────────────────────────────────────────│
│   with area + label  ││ status line                                      │
│   counts             ││                                                  │
└──────────────────────┘└──────────────────────────────────────────────────┘
```

Left column: what you are working *on* and *with*. Top bar: grouped
left-to-right by when you reach for it — tools, then view, then operating point,
then edits, then outputs. Bottom: one status line that always says what just
happened, including how many undo steps remain.

Shared conventions:
- red overlay = predicted crack; green brush = force-crack; blue brush = force-not-crack
- crack is **black** in exported B&W masks
- long operations return a job id and stream progress; the browser never blocks
- the status line reports counts, not just "done"
