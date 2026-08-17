# TXM Crack Detection — Quickstart

Detects cracks in transmission X-ray microscopy images. Drag images in, look at
what the model found, fix what it got wrong, press Retrain. That is the whole loop.

## Install

```bash
git clone <this-repo>
cd TXM_Crack_Detection_Pipeline
python3 -m pip install -r requirements.txt
```

Python 3.9+. Apple Silicon, CUDA and CPU-only all work; a GPU makes the SAM step
~10x faster but nothing requires one.

## Run

```bash
python3 app/server.py
```

Open **http://127.0.0.1:8800**.

On the very first prediction it downloads Meta's SAM ViT-H (~2.4 GB) from
HuggingFace and caches it in `~/.cache/huggingface`. That happens once. Set
`PORT=9000` to use a different port.

## Using it

1. **Drag TXM images in** (`.tif`, `.tiff`, `.png`). Each one is automatically
   destitched, flat-fielded, embedded with SAM, and predicted. Budget ~20 s per
   2.9 MP image on a GPU; the progress bar names the stage.
2. **Look.** Red overlay is predicted crack. The image you see is the
   destitched + flat-fielded version, because real cracks are thin and faint and
   are often invisible in raw; the *model* is fed raw, which is what it was
   trained on. Both corrections preserve geometry, so the mask registers exactly.
3. **Correct.** *Add crack* paints crack, *Eraser* removes it. Brush size and
   zoom are on the toolbar. Strokes save on mouse-up.
4. **Retrain on my corrections.** Trains a new model on every correction you have
   painted across every image, validates it, and deploys it only if it does not
   regress. Then **Re-overlay all** re-predicts your images with it.
5. **Export mask** downloads the final mask as a PNG (crack = black).

Nothing here needs a config file edited or a script run in the right order.

## What the model is

A mean-probability ensemble of two models:

- **17 hand-crafted features → MLP.** Intensity, Gaussian-smoothed intensity at
  σ=2…64, gradient magnitude, Laplacian, and local-standard-deviation texture.
- **SAM + those 17 → MLP.** Meta's Segment Anything ViT-H image embedding (256
  channels) concatenated with the same 17 features.

Measured under leave-one-image-out on the 4 Ilastik ground-truth images, with
false positives measured on 6 specimens confirmed crack-free by the specimen owner:

| approach | mean IoU | pixel-weighted IoU | recall | crack-free FP |
|---|---|---|---|---|
| 17 features alone | 0.744 | 0.721 | 0.891 | 7.43% |
| SAM + 17 (hybrid alone) | 0.795 | 0.719 | 0.894 | 0.14% |
| **ensemble of the two (default)** | **0.821** | **0.777** | **0.914** | **0.11%** |

The hybrid alone only *ties* the simple model once you weight by pixel count,
because it loses badly on the largest image — which is 73% of all labelled
pixels. Averaging wins on every image, on both weightings, with recall going up
rather than being traded away. That is why the default is the ensemble.

Zero-shot SAM, used the way SAM is designed to be used (prompt it, read out
masks), scores 0.23–0.36 here and is not usable. See `SAM_COMPARISON.md` for the
full study, including 33 verified citations.

## Things worth knowing before you trust a number

- **Ground truth is 4 images, all one specimen group, all wide-open cracks**
  (median crack width 65 px). With n=4 an exact sign test cannot go below
  p=0.125, so nothing here can be statistically significant. Treat differences
  under ~0.015 IoU as indistinguishable from reseeding — that is the measured
  run-to-run noise.
- **Post-processing is off by default and is under suspicion.** The
  shape-validation and minimum-size filter measurably removes thin crack: on one
  ground-truth image it costs 0.084 IoU and 0.072 recall, and hand-painted stroke
  recall drops from ~0.87 at a raw threshold to 0.14–0.40 after it. Toggle it on
  if you want the old behaviour.
- **Retrain refuses to deploy a regression.** A candidate must hold IoU within
  0.01. Every regression this project has had passed a single-metric check —
  an over-aggressive filter and a good one both reduce predicted area, and only
  recall against ground truth separates them.
- If you retrain and it says *not deployed*, the model file is still saved so you
  can inspect it. **Roll back** restores the previous model.

## Layout

```
app/server.py          the web app
app/core/model.py      the deployed model, one predict() call
app/core/pipeline.py   ingest + retrain, including the validation gate
app/core/store.py      per-image storage and the model registry
app/static/index.html  the whole frontend
code/                  the research pipeline: features, destitch, flatfield, SAM harness
dataset_cache/         the 4 ground-truth images (needed to validate a retrain)
models/                shipped model weights
app_data/              your uploads, embeddings and retrained models (gitignored)
```

Your data lives in `app_data/` inside the checkout. Nothing points at an absolute
path outside it, so moving or deleting your original files cannot break the app.
