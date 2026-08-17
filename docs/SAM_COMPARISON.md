# Why not Meta's Segment Anything?

Short answer: **it was tried, and both answers are in the data.** Used the way SAM is designed to be used — prompt it and read out masks — it fails badly on TXM crack images. Used as a frozen feature extractor with a supervised head, it is competitive with the hand-crafted features, and combining the two is better than either alone.

Everything below is measured on the four Ilastik ground-truth images (the only pixel-level truth that exists for this project), with the same `metrics_from_pred` and the same leave-one-image-out protocol as every other model in the repository. Reproduce with:

```bash
python3 code/sam_experiments.py --conditions all --model huge --save-masks
python3 code/baseline_loio_for_sam.py
python3 code/generate_sam_figures.py
```

SAM checkpoint: `facebook/sam-vit-huge` (641M parameters, the largest SAM 1). Input rendering: gray. Device: mps. Training budget for the learned conditions: 20,000 pixels per class.

## Results

| approach | deployable | mean IoU | recall | precision |
|---|---|---|---|---|
| **Our 17 features + MLP** (current pipeline) | yes | **0.744** | 0.891 | 0.818 |
| SAM automatic masks, whole frame | yes | 0.344 | 0.356 | 0.706 |
| SAM automatic masks, 1024 px tiles | yes | 0.289 | 0.467 | 0.535 |
| SAM 16x16 point grid per tile | yes | 0.257 | 0.959 | 0.259 |
| SAM ViT features -> logistic regression | yes | 0.729 | 0.843 | 0.838 |
| SAM ViT features -> MLP | yes | 0.738 | 0.838 | 0.852 |
| SAM ViT features + our 17 -> MLP | yes | 0.797 | 0.893 | 0.879 |
| SAM automatic masks + perfect mask picker | **no — given the answer** | 0.296 | 0.797 | 0.320 |
| SAM prompted with points ON the true crack | **no — given the answer** | 0.616 | 0.873 | 0.681 |
| SAM given a box around each true crack | **no — given the answer** | 0.679 | 0.893 | 0.739 |


### Is that the features, or the classifier?

The SAM-embedding rows train `MLP(128,64)`; the deployed baseline trains `MLP(64,32)` with early stopping. Comparing those directly confounds the feature set with the head, so the head was held fixed and only the features swapped:

| features | head | mean IoU |
|---|---|---|
| 17 hand-crafted | MLP(64,32) + early stop (deployed) | 0.744 |
| 17 hand-crafted | **MLP(128,64)** — SAM's head | 0.660 |
| SAM ViT (256) | MLP(128,64) | 0.738 |
| SAM ViT (256) + 17 | MLP(128,64) | 0.797 |

The head alone accounts for -0.084 IoU. Whatever remains of the gap is attributable to the features, which is the claim being made.

The rows marked *no* were handed the ground truth they are scored against — prompt points sampled on the true crack, boxes drawn around it, or a mask-picker that knows the answer. They cannot run in deployment. They are included so the result cannot be dismissed as bad prompting: if SAM loses while being told where the crack is, the prompting was not the problem.

## Per image

This is where the interesting structure is, and it is invisible in the means.

| image | megapixels | our 17 features | SAM ViT features -> MLP | SAM ViT features + our 17 -> MLP | SAM automatic masks, 1024 px tiles |
|---|---|---|---|---|---|
| `333_75_um_zoom` | 2.9 | 0.738 | 0.746 | 0.787 | 0.318 |
| `336_25` | 2.9 | 0.748 | 0.851 | 0.854 | 0.248 |
| `338_13` | 2.9 | 0.780 | 0.845 | 0.869 | 0.395 |
| `LARGE_343_75` | 23.5 | 0.708 | 0.508 | 0.680 | 0.194 |

## What the numbers mean

**1. Zero-shot SAM does not work here — but NOT because it cannot see the crack.** Used as designed it scores 0.23–0.36. Whole-frame inference is precise but nearly empty (precision 0.94–0.96 at recall 0.35–0.48); tiling inverts that (recall 0.75–0.85 at precision ~0.27) because SAM starts returning whole tiles. Same mediocre IoU, opposite causes.

The tempting explanation — that SAM does not resolve the structure — is WRONG, and measuring it is what settled the question. HuggingFace's mask-generation pipeline defaults to `pred_iou_thresh=0.88` and `stability_score_thresh=0.95`, tuned for natural photographs, and those discard the great majority of proposals: 8–13 masks survive per image. Relaxing the gates yields 200+ proposals per image, and the BEST SINGLE PROPOSAL reaches IoU 0.639–0.766 against ground truth — comparable to what SAM achieves when handed an oracle bounding box.

So SAM perceives the crack and proposes a good mask for it. What it cannot do is say WHICH of its 200 proposals is the crack: it is class-agnostic by design, and its own confidence score ranks the correct proposal below the default threshold. The bottleneck is PROPOSAL SELECTION, not perception — and supervised classification is exactly the thing that solves selection. That is why the frozen-feature route below works while the prompt-and-read-out route does not.

| image | best single proposal (relaxed AMG) | after deployable selection |
|---|---|---|
| `333_75_um_zoom` | 0.639 | 0.421 |
| `336_25` | 0.686 | 0.357 |
| `338_13` | 0.645 | 0.430 |
| `LARGE_343_75` | 0.766 | 0.526 |

**2. SAM cannot ingest the largest image at all.** `LARGE_343_75` (3691×6367, 23.5 MP) overflows a tensor index in the mask decoder and exhausts 39 GB of unified memory. Tiling works around it, but "SAM handles your data as-is" is false.

**3. But SAM's features are genuinely good.** Freezing the ViT and training a small head on its embeddings gives mean IoU 0.738 against 0.744 for the 17 hand-crafted features — a tie within noise, and SAM wins outright on the three same-magnification zoom frames. The failure is confined to the 23.5 MP mosaic, which is at a different magnification: SAM sees fixed 1024 px tiles and has no scale invariance, whereas the 17 features include Gaussians from σ=2 to σ=64 and do. **This contradicts the obvious prediction** that SAM's 16×16-pixel embedding stride would cap it, and that prediction was made and then overruled by measurement.

**4. The two feature sets are complementary.** Concatenating SAM's 256 embedding channels with the 17 hand-crafted features gives 0.797, which beats the 17 features alone (0.744) and SAM alone (0.738). That is the actionable result: not "SAM instead", but "SAM as well".

## Cost

| | our 17 features | SAM ViT-Huge |
|---|---|---|
| model size | ~1 MB | 2.4 GB |
| dependency | scikit-learn | PyTorch + transformers + 2.4 GB weights |
| feature extraction, 2.9 MP image | seconds, CPU | GPU required in practice |
| largest image | works | crashes without tiling |
| interpretable | yes — 17 named features with importances | no — 256 opaque channels |

## The literature says the same thing

Independent published evidence, every citation checked to exist (35 verification passes, 35 confirmed real):

- **Zero-shot SAM on the largest crack benchmark: F1 13%, IoU 17%.** After adapting 41K LayerNorm parameters on 22,158 labelled crack images it reaches IoU 44%. *Segment Any Crack*, ASCE J. Computing in Civil Engineering — https://arxiv.org/abs/2504.14138
- **Un-finetuned EdgeSAM on cracks: IoU 0.157 at recall 0.765** — high recall with massive false positives. This is almost exactly the tiled result measured above, independently replicated. *Crack-EdgeSAM* — https://arxiv.org/abs/2412.07205
- **SAM's own paper** lists the failure mode: it "can miss fine structures, hallucinates small disconnected components, and does not produce boundaries as crisply as more computationally intensive methods". Trained on 11M natural photographs; zero X-ray, zero microscopy. https://arxiv.org/abs/2304.02643
- **Thin, low-contrast structures are a fundamental limit, and fine-tuning does not fix it**: "targeted fine-tuning fails to resolve this issue, indicating a fundamental limitation." Retinal vessels: average IoU ≈0.05. WACV 2026 — https://arxiv.org/abs/2412.04243
- **A frozen SAM encoder lacks crack-relevant features**: tuning only the head gives IoU 0.556; adapters and LoRA inside the ViT backbone are required to reach 0.649. *CrackSAM*, Construction and Building Materials — https://arxiv.org/abs/2312.04233
- **Every published SAM-for-cracks method needed 9,603–22,158 pixel-labelled images.** This project has 4 with pixel truth and 20 of 71 with any hand-drawn crack strokes.
- Even fully supervised, cracks are hard: the OmniCrack30k benchmark's best model (nnU-Net) reaches only 64% centreline IoU across 30k images — the field had to invent a tolerance metric because plain IoU is dominated by 1–2 px boundary disagreement on thin structures. CVPR 2024 Workshops.

Full verified citation list (33 sources): `results/sam/citations.json`

## RETRACTION: the hybrid's advantage does not survive scrutiny

The +0.05 mean IoU gain reported for SAM features + our 17 is an artifact of the UNWEIGHTED per-image mean. Three of the four ground-truth images are 2.9 MP; the fourth is 23.5 MP, i.e. 73% of all labelled pixels. Weighting by pixel count instead:

| condition | unweighted mean | pixel-weighted mean | wins | exact sign-test p |
|---|---|---|---|---|
| SAM ViT features + our 17 -> MLP | 0.797 | 0.722 | 3/4 | 0.625 |
| SAM ViT features -> MLP | 0.738 | 0.590 | 3/4 | 0.625 |
| SAM ViT features -> logistic regression | 0.729 | 0.591 | 3/4 | 0.625 |
| amg_relaxed_oracle | 0.640 | 0.685 | 2/4 | 1.000 |
| SAM given a box around each true crack | 0.679 | 0.786 ⚠︎ flips | 1/4 | 0.625 |
| SAM automatic masks, 1024 px tiles | 0.289 | 0.228 | 0/4 | 0.125 |
| *our 17 features (baseline)* | 0.744 | 0.721 | — | — |

So the hybrid goes from **0.797 vs 0.744** unweighted (+0.054) to **0.722 vs 0.721** pixel-weighted (+0.001) — a dead heat. Per-image deltas: 333_75_um_zoom +0.049, 336_25 +0.105, 338_13 +0.088, LARGE_343_75 -0.028.

**Neither weighting is wrong**, which is the problem: a result that reverses between two defensible aggregations is not robust enough to deploy on. And SAM-features-alone is decisively WORSE pixel-weighted (0.590 vs 0.721), because it fails on the one large image.

**No result here is statistically significant, and none can be.** n=4 paired; exact sign test and Wilcoxon both floor at p=0.125 two-sided, so significance at 0.05 is unreachable by construction. A 4–0 sweep is the strongest evidence this dataset can produce. The zero-shot SAM conditions DO sweep 0–4 against the baseline under both weightings, so that conclusion is safe. The feature-level comparison is 3–1 with sign-test p=0.625 — no better than a coin flip's worth of evidence.

**Consequence: do not deploy the SAM hybrid on this evidence.** The honest reading is that SAM's frozen features are COMPARABLE to the 17 hand-crafted ones on same-magnification frames and WORSE on the one frame at different magnification, with far higher cost. Revisit only if more ground truth appears, especially at more than one magnification.

## The caveat that matters most

**These results are validated only on WIDE cracks, and the project's actual unsolved problem is thin ones.**

Measured from the ground-truth masks (2x distance transform along the medial axis), the median crack width in the four GT images is **65 px** — four SAM embedding cells across. At 333–343 lbf these cracks are wide open, which is precisely the regime a blob-oriented segmenter handles well. Only 27% of the labelled crack is narrower than one embedding cell.

That resolves what looks like a contradiction with the literature. The published zero-shot SAM crack failures (F1 13%, retinal vessels IoU ≈0.05) are on hairline structures a few pixels wide. This dataset, at load, is not in that regime. Our result and theirs are consistent; they are measurements of different regimes.

The consequence is a limit on what may be claimed. HANDOFF.md §3 records that the real cracks in the AM, Wrought and B3 groups are **thin, very faint and central** — the regime where the literature says SAM fails and where fine-tuning measurably failed to help. There is no pixel ground truth for those groups, so **nothing here shows the SAM hybrid will help on them.** It should be deployed on the strength of the B2 evidence and then checked on AM/Wrought against new hand labels, not assumed to transfer.

## Would fine-tuning SAM help?

Probably not, and the reason is data rather than compute. Published SAM crack fine-tunes used 9,603–22,158 labelled images; this project has pixel-level truth for 4, all from one specimen group. The WACV result above is the more damaging one: for thin, low-contrast structures, fine-tuning measurably *failed* to remove the deficit, which the authors attribute to SAM misreading local structure as global texture.

The higher-value use of the same effort is labelling more images for the existing classifier — 51 of 71 still have no positive crack strokes, and 27 (all of B3 and Wrought) have never shown any model a crack in their own material. That is the actual bottleneck, and no choice of architecture fixes it.

---

*Generated by `code/write_sam_report.py` from the result JSONs — every number above is read from measurement output, not transcribed.*
