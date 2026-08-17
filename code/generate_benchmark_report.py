"""
Comprehensive ML benchmark report for the TXM crack pixel classifier, in the
same spirit as the "MeltpoolNet" benchmarking paper (Akbari et al., Additive
Manufacturing 55 (2022) 102817): multiple algorithms trained/evaluated under
one consistent protocol, accuracy bar charts, ROC curves, confusion
matrices, a Random-Forest feature-importance study, and a sample-size-vs-
accuracy learning curve. Every number here is computed fresh from the real
cached TXM data (dataset_cache/) -- nothing is hand-entered or simulated.

Protocol (must match train_pixel_hgb.py / train_pixel_rf.py so results are
comparable to this project's earlier model-selection results):
  - Leave-one-image-out (LOIO) across the 4 ground-truth images, 4 folds.
  - Per fold: train on up to N_PER_CLASS_PER_IMAGE crack + N_PER_CLASS_PER_IMAGE
    background pixels sampled from EACH of the 3 non-held-out images.
  - Predict on every pixel of the held-out image, threshold at 0.5.
  - IoU / Dice / Precision / Recall against that image's real ground truth.

Produces, in ../benchmark_figures/:
  fig_a_model_comparison.png    grouped bar: IoU/Dice/Precision/Recall, 3 base algorithms
  fig_b_area_fraction_parity.png predicted vs. actual crack-area-fraction per held-out image
  fig_c_roc_curves.png          pooled ROC curve per algorithm (AUC in legend)
  fig_d_confusion_matrix.png    pixel-level confusion matrix, one panel per algorithm
  fig_e_feature_importance.png  Random Forest feature importance, all 17 features
  fig_f_learning_curve.png      IoU vs. bootstrap sample size (HistGradientBoosting)
  fig_g_decision_boundary.png   2-feature decision-boundary visualization
  benchmark_summary.json        every number behind every figure, for the record
  benchmark_summary.md          the same, as a readable table

Usage:
    python3 generate_benchmark_report.py [--quick]

--quick uses tiny sample sizes and skips the learning curve, for a fast
correctness smoke-test before committing to the full ~15-minute real run.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import auc, roc_curve
from skimage.measure import label as sklabel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from txm_features import FEATURE_NAMES, N_FEATURES
from apply_pixel_model import postprocess_mask

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(PROJECT_DIR, "dataset_cache")
OUT_DIR = os.path.join(PROJECT_DIR, "benchmark_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- palette
# Fixed categorical hues from the dataviz skill's validated palette
# (references/palette.md) -- assigned once per role and never reassigned
# within a figure family, so the same model always has the same color
# across every figure in this report.
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948",
)
INK, MUTED, GRID = "#0b0b0b", "#898781", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
    "grid.color": "#e1e0d9", "grid.linewidth": 0.8, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif",
})

with open(os.path.join(CACHE_DIR, "manifest.json")) as f:
    IMAGES = json.load(f)["images"]

SEED = 0
MODEL_SPECS = {
    "RandomForest": (RandomForestClassifier,
                      dict(n_estimators=300, max_depth=None, class_weight="balanced", n_jobs=-1, random_state=0),
                      BLUE),
    "ExtraTrees": (ExtraTreesClassifier,
                    dict(n_estimators=400, max_depth=None, class_weight="balanced", n_jobs=-1, random_state=0),
                    ORANGE),
    "HistGradientBoosting": (HistGradientBoostingClassifier,
                               dict(max_iter=300, max_depth=8, learning_rate=0.1, class_weight="balanced", random_state=0),
                               AQUA),
}
PRODUCTION_MODEL = "HistGradientBoosting"


# ------------------------------------------------------------- core utils
def sample_pixels(feats, gt, n_per_class, rng):
    flat_feats = feats.reshape(-1, feats.shape[-1])
    flat_gt = gt.reshape(-1)
    crack_idx = np.flatnonzero(flat_gt)
    bg_idx = np.flatnonzero(~flat_gt)
    n_c = min(n_per_class, len(crack_idx))
    n_b = min(n_per_class, len(bg_idx))
    c_s = rng.choice(crack_idx, size=n_c, replace=False)
    b_s = rng.choice(bg_idx, size=n_b, replace=False)
    idx = np.concatenate([c_s, b_s])
    X = flat_feats[idx]
    y = np.concatenate([np.ones(n_c, dtype=bool), np.zeros(n_b, dtype=bool)])
    return X, y


def metrics_from_pred(pred, gt):
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    p_sum, g_sum = int(pred.sum()), int(gt.sum())
    return dict(
        iou=inter / union if union else float("nan"),
        dice=2 * inter / (p_sum + g_sum) if (p_sum + g_sum) else float("nan"),
        precision=inter / p_sum if p_sum else 0.0,
        recall=inter / g_sum if g_sum else 0.0,
    )


def run_loio(model_cls, params, n_per_class, want_fi=False, pool_cap=150000, on_fold=None):
    """LOIO across the 4 cached images. `on_fold(name, proba_2d, gt_2d)` is
    called right after prediction, while the full-resolution arrays are
    still in memory -- lets the caller do extra per-fold work (postprocess,
    confusion-matrix accumulation) with zero extra model fits."""
    fold_results = []
    pooled_probs, pooled_true = [], []
    fi_list = []
    for i, held in enumerate(IMAGES):
        rng = np.random.RandomState(SEED)
        X_list, y_list = [], []
        for j, img in enumerate(IMAGES):
            if j == i:
                continue
            feats = np.load(img["feat_path"])
            gt = np.load(img["gt_path"])
            X, y = sample_pixels(feats, gt, n_per_class, rng)
            X_list.append(X)
            y_list.append(y)
            del feats, gt
        X_train = np.concatenate(X_list)
        y_train = np.concatenate(y_list)
        del X_list, y_list

        clf = model_cls(**params)
        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_s = time.time() - t0
        n_train = len(y_train)
        del X_train, y_train

        feats_h = np.load(held["feat_path"])
        gt_h = np.load(held["gt_path"])
        flat_h = feats_h.reshape(-1, feats_h.shape[-1])
        gt_flat = gt_h.reshape(-1)
        t0 = time.time()
        proba_flat = clf.predict_proba(flat_h)[:, 1]
        pred_s = time.time() - t0
        pred_flat = proba_flat >= 0.5

        m = metrics_from_pred(pred_flat, gt_flat)
        m.update(image=held["name"], fit_seconds=fit_s, predict_seconds=pred_s, n_train_pixels=n_train)
        fold_results.append(m)
        print(f"    [{held['name']}] IoU={m['iou']:.4f} Dice={m['dice']:.4f} "
              f"(fit {fit_s:.1f}s, predict {pred_s:.1f}s, n_train={n_train})")

        sub_n = min(pool_cap, len(gt_flat))
        sub_idx = rng.choice(len(gt_flat), size=sub_n, replace=False)
        pooled_probs.append(proba_flat[sub_idx])
        pooled_true.append(gt_flat[sub_idx])

        if want_fi and hasattr(clf, "feature_importances_"):
            fi_list.append(clf.feature_importances_)

        if on_fold is not None:
            on_fold(held["name"], proba_flat.reshape(gt_h.shape), gt_h)

        del feats_h, gt_h, flat_h, gt_flat, proba_flat, pred_flat

    mean_fi = np.mean(fi_list, axis=0) if fi_list else None
    return fold_results, np.concatenate(pooled_probs), np.concatenate(pooled_true), mean_fi


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke test: tiny samples, skip learning curve")
    args = ap.parse_args()

    n_per_class = 300 if args.quick else 30000
    summary = {"protocol": {
        "n_images": len(IMAGES),
        "n_per_class_per_image": n_per_class,
        "images": [img["name"] for img in IMAGES],
        "n_features": N_FEATURES,
        "feature_names": FEATURE_NAMES,
    }, "models": {}}

    # confusion-matrix + parity-plot accumulators, filled in by on_fold callbacks
    confmat = {name: np.zeros((2, 2), dtype=np.int64) for name in MODEL_SPECS}
    parity_rows = []  # (model, image, actual_frac, predicted_frac)

    def make_on_fold(model_name):
        def _on_fold(image_name, proba_2d, gt_2d):
            final_mask = postprocess_mask(proba_2d)
            actual_frac = float(gt_2d.mean())
            pred_frac = float(final_mask.mean())
            parity_rows.append(dict(model=model_name, image=image_name,
                                     actual_fraction=actual_frac, predicted_fraction=pred_frac))
            gt_flat, pred_flat = gt_2d.reshape(-1), final_mask.reshape(-1)
            tp = int(np.logical_and(pred_flat, gt_flat).sum())
            tn = int(np.logical_and(~pred_flat, ~gt_flat).sum())
            fp = int(np.logical_and(pred_flat, ~gt_flat).sum())
            fn = int(np.logical_and(~pred_flat, gt_flat).sum())
            confmat[model_name] += np.array([[tn, fp], [fn, tp]], dtype=np.int64)
        return _on_fold

    # ---------------- Round 1: base-algorithm bake-off (LOIO) ----------------
    pooled = {}
    fi_by_model = {}
    for name, (cls, params, color) in MODEL_SPECS.items():
        print(f"\n=== {name} (LOIO, n_per_class={n_per_class}) ===")
        folds, probs, trues, fi = run_loio(cls, params, n_per_class, want_fi=True, on_fold=make_on_fold(name))
        mean_m = {k: float(np.mean([f[k] for f in folds])) for k in ("iou", "dice", "precision", "recall")}
        std_m = {k: float(np.std([f[k] for f in folds])) for k in ("iou", "dice", "precision", "recall")}
        pooled[name] = (probs, trues)
        fi_by_model[name] = fi
        summary["models"][name] = dict(folds=folds, mean=mean_m, std=std_m,
                                        mean_fit_seconds=float(np.mean([f["fit_seconds"] for f in folds])),
                                        mean_predict_seconds=float(np.mean([f["predict_seconds"] for f in folds])))
        print(f"  mean IoU={mean_m['iou']:.4f}  Dice={mean_m['dice']:.4f}  "
              f"Precision={mean_m['precision']:.4f}  Recall={mean_m['recall']:.4f}")

    # ---------------- Fig A: grouped bar, IoU/Dice/Precision/Recall ----------------
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_order = ["iou", "dice", "precision", "recall"]
    metric_labels = ["IoU", "Dice", "Precision", "Recall"]
    model_names = list(MODEL_SPECS.keys())
    x = np.arange(len(metrics_order))
    width = 0.25
    for k, name in enumerate(model_names):
        color = MODEL_SPECS[name][2]
        means = [summary["models"][name]["mean"][m] for m in metrics_order]
        stds = [summary["models"][name]["std"][m] for m in metrics_order]
        ax.bar(x + (k - 1) * width, means, width, yerr=stds, capsize=3, label=name, color=color,
               edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (mean over 4 leave-one-image-out folds)")
    ax.set_title("(a) Base-algorithm comparison, leave-one-image-out cross-validation")
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, "fig_a_model_comparison.png")

    # ---------------- Fig B: predicted vs. actual crack-area-fraction (parity) ----------------
    fig, ax = plt.subplots(figsize=(6, 6))
    lims = [0, 0.4]
    ax.plot(lims, lims, "--", color=MUTED, linewidth=1.3, label="perfect prediction (y = x)")
    markers = {img["name"]: m for img, m in zip(IMAGES, ["o", "s", "^", "D"])}
    for row in parity_rows:
        color = MODEL_SPECS[row["model"]][2]
        ax.scatter(row["actual_fraction"], row["predicted_fraction"], s=90, color=color,
                   marker=markers[row["image"]], edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual crack area fraction (ground truth)")
    ax.set_ylabel("Predicted crack area fraction (final post-processed mask)")
    ax.set_title("(b) Predicted vs. actual crack coverage per held-out image")
    model_handles = [Line2D([0], [0], marker="o", linestyle="", color=MODEL_SPECS[n][2], label=n, markersize=9)
                      for n in model_names]
    image_handles = [Line2D([0], [0], marker=markers[img["name"]], linestyle="", color=MUTED,
                              label=img["name"], markersize=9) for img in IMAGES]
    leg1 = ax.legend(handles=model_handles, loc="upper left", frameon=False, title="Model (color)")
    ax.add_artist(leg1)
    ax.legend(handles=image_handles, loc="lower right", frameon=False, title="Image (marker shape)")
    savefig(fig, "fig_b_area_fraction_parity.png")

    # ---------------- Fig C: pooled ROC curves ----------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1.2, label="chance (AUC = 0.50)")
    roc_summary = {}
    for name in model_names:
        probs, trues = pooled[name]
        fpr, tpr, _ = roc_curve(trues, probs)
        roc_auc = auc(fpr, tpr)
        roc_summary[name] = float(roc_auc)
        ax.plot(fpr, tpr, color=MODEL_SPECS[name][2], linewidth=2.2, label=f"{name} (AUC = {roc_auc:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("(c) ROC curves, pooled across 4 held-out images")
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, "fig_c_roc_curves.png")
    summary["roc_auc"] = roc_summary

    # ---------------- Fig D: confusion matrices (production-pipeline masks) ----------------
    # sharey=True + a real wspace, and the "Ground truth" y-label/ticks only
    # on the leftmost panel -- otherwise each subplot's own ylabel crowds
    # into the narrow gap next to its neighbor and the labels collide.
    fig, axes = plt.subplots(1, len(model_names), figsize=(4.6 * len(model_names), 4.6), sharey=True)
    fig.subplots_adjust(wspace=0.15)
    cmap_conf = LinearSegmentedColormap.from_list("blue_ramp", ["#fcfcfb", BLUE])
    confmat_summary = {}
    for k, (ax, name) in enumerate(zip(axes, model_names)):
        cm = confmat[name]
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        confmat_summary[name] = cm.tolist()
        im = ax.imshow(cm_norm, cmap=cmap_conf, vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["background", "crack"])
        if k == 0:
            ax.set_yticklabels(["background", "crack"])
            ax.set_ylabel("Ground truth")
        ax.set_xlabel("Predicted")
        ax.set_title(name)
        ax.grid(False)
        for r in range(2):
            for c in range(2):
                txt_color = "white" if cm_norm[r, c] > 0.5 else INK
                ax.text(c, r, f"{cm[r, c]:,}\n({cm_norm[r, c]*100:.1f}%)", ha="center", va="center",
                        color=txt_color, fontsize=10)
    fig.suptitle("(d) Pixel-level confusion matrix, final post-processed mask (all 4 held-out images pooled)")
    savefig(fig, "fig_d_confusion_matrix.png")
    summary["confusion_matrix"] = confmat_summary

    # ---------------- Fig E: Random Forest feature importance ----------------
    rf_fi = fi_by_model["RandomForest"]
    order = np.argsort(rf_fi)[::-1]
    group_of = []
    for fname in FEATURE_NAMES:
        if fname == "intensity":
            group_of.append("intensity")
        elif fname.startswith("smooth"):
            group_of.append("smoothed intensity")
        elif fname.startswith("gradmag"):
            group_of.append("gradient magnitude")
        elif fname.startswith("laplacian"):
            group_of.append("Laplacian")
        else:
            group_of.append("texture (local std)")
    group_color = {"intensity": VIOLET, "smoothed intensity": BLUE, "gradient magnitude": ORANGE,
                    "Laplacian": AQUA, "texture (local std)": YELLOW}
    fig, ax = plt.subplots(figsize=(7, 6))
    y_pos = np.arange(N_FEATURES)
    ax.barh(y_pos, rf_fi[order], color=[group_color[group_of[i]] for i in order], edgecolor="white", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([FEATURE_NAMES[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Relative importance (Random Forest, mean decrease in impurity)")
    ax.set_title("(e) Feature importance across all 17 pixel features")
    handles = [Line2D([0], [0], marker="s", linestyle="", color=c, label=g, markersize=10)
                for g, c in group_color.items()]
    ax.legend(handles=handles, frameon=False, loc="lower right", title="Feature group")
    savefig(fig, "fig_e_feature_importance.png")
    summary["feature_importance_random_forest"] = {FEATURE_NAMES[i]: float(rf_fi[i]) for i in range(N_FEATURES)}

    # ---------------- Fig F: learning curve, IoU vs. bootstrap sample size ----------------
    if not args.quick:
        sizes = [5000, 15000, 30000, 60000, 100000]
        lc_cls, lc_params, lc_color = MODEL_SPECS[PRODUCTION_MODEL]
        lc_means, lc_stds = [], []
        print(f"\n=== Learning curve: {PRODUCTION_MODEL} IoU vs. bootstrap sample size ===")
        for n in sizes:
            print(f"  n_per_class_per_image={n}")
            folds, _, _, _ = run_loio(lc_cls, lc_params, n)
            ious = [f["iou"] for f in folds]
            lc_means.append(float(np.mean(ious)))
            lc_stds.append(float(np.std(ious)))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(sizes, lc_means, yerr=lc_stds, marker="o", markersize=7, color=lc_color,
                    linewidth=2, capsize=4)
        ax.set_xscale("log")
        ax.set_xlabel("Bootstrap sample size (pixels per class per training image)")
        ax.set_ylabel("Mean IoU (4-fold leave-one-image-out)")
        ax.set_title(f"(f) {PRODUCTION_MODEL}: accuracy vs. training sample size")
        savefig(fig, "fig_f_learning_curve.png")
        summary["learning_curve"] = {"sample_sizes": sizes, "mean_iou": lc_means, "std_iou": lc_stds,
                                       "model": PRODUCTION_MODEL}
    else:
        print("\n(--quick: skipping learning curve)")

    # ---------------- Fig G: 2-feature decision-boundary visualization ----------------
    top2_idx = np.argsort(rf_fi)[::-1][:2]
    f1_name, f2_name = FEATURE_NAMES[top2_idx[0]], FEATURE_NAMES[top2_idx[1]]
    print(f"\n=== Decision boundary in top-2-feature space: {f1_name} vs {f2_name} ===")
    rng = np.random.RandomState(SEED)
    X2_list, y2_list = [], []
    for img in IMAGES:
        feats = np.load(img["feat_path"])
        gt = np.load(img["gt_path"])
        X, y = sample_pixels(feats, gt, 15000, rng)
        X2_list.append(X[:, top2_idx])
        y2_list.append(y)
        del feats, gt
    X2 = np.concatenate(X2_list)
    y2 = np.concatenate(y2_list)
    lc_cls, lc_params, _ = MODEL_SPECS[PRODUCTION_MODEL]
    clf2 = lc_cls(**lc_params)
    clf2.fit(X2, y2)

    x_min, x_max = np.percentile(X2[:, 0], [0.5, 99.5])
    y_min, y_max = np.percentile(X2[:, 1], [0.5, 99.5])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
    grid_proba = clf2.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

    diverging = LinearSegmentedColormap.from_list("blue_gray_red", [BLUE, "#f0efec", RED])
    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(xx, yy, grid_proba, levels=np.linspace(0, 1, 21), cmap=diverging, alpha=0.85)
    ax.contour(xx, yy, grid_proba, levels=[0.5], colors=INK, linewidths=1.5, linestyles="--")
    plot_n = min(4000, len(y2))
    plot_idx = rng.choice(len(y2), size=plot_n, replace=False)
    ax.scatter(X2[plot_idx, 0][~y2[plot_idx]], X2[plot_idx, 1][~y2[plot_idx]],
               s=8, color=BLUE, alpha=0.5, label="background (true)", edgecolor="none")
    ax.scatter(X2[plot_idx, 0][y2[plot_idx]], X2[plot_idx, 1][y2[plot_idx]],
               s=8, color=RED, alpha=0.5, label="crack (true)", edgecolor="none")
    # Clamp to the grid's own extent -- otherwise a handful of scatter
    # outliers beyond the 99.5th percentile silently expand the axes and
    # leave the decision-region background looking like it only covers a
    # small corner of the plot.
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    fig.colorbar(cf, ax=ax, label="Predicted crack probability")
    ax.set_xlabel(f1_name)
    ax.set_ylabel(f2_name)
    ax.set_title(f"(g) {PRODUCTION_MODEL} decision regions, top-2 features by RF importance")
    ax.legend(frameon=False, loc="upper right")
    savefig(fig, "fig_g_decision_boundary.png")
    summary["decision_boundary"] = {"feature_1": f1_name, "feature_2": f2_name}

    # ---------------------------------------------------------------- save summary
    json_path = os.path.join(OUT_DIR, "benchmark_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved {json_path}")

    md_lines = ["# TXM Crack Classifier -- Benchmark Report", "",
                f"Protocol: leave-one-image-out cross-validation across {len(IMAGES)} images "
                f"({', '.join(i['name'] for i in IMAGES)}), {n_per_class} pixels/class/image bootstrap sample.", "",
                "## Model comparison (mean over 4 folds)", "",
                "| Model | IoU | Dice | Precision | Recall | Fit time (s) | Predict time (s) | ROC AUC |",
                "|---|---|---|---|---|---|---|---|"]
    for name in model_names:
        m = summary["models"][name]["mean"]
        md_lines.append(
            f"| {name} | {m['iou']:.4f} | {m['dice']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{summary['models'][name]['mean_fit_seconds']:.1f} | "
            f"{summary['models'][name]['mean_predict_seconds']:.1f} | {roc_summary[name]:.4f} |"
        )
    md_lines += ["", "## Figures", "",
                 "- `fig_a_model_comparison.png` -- accuracy bar chart",
                 "- `fig_b_area_fraction_parity.png` -- predicted vs actual crack coverage",
                 "- `fig_c_roc_curves.png` -- ROC curves",
                 "- `fig_d_confusion_matrix.png` -- confusion matrices",
                 "- `fig_e_feature_importance.png` -- Random Forest feature importance",
                 "- `fig_f_learning_curve.png` -- IoU vs training sample size" + (" (skipped, --quick)" if args.quick else ""),
                 "- `fig_g_decision_boundary.png` -- decision boundary, top-2 features"]
    md_path = os.path.join(OUT_DIR, "benchmark_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
