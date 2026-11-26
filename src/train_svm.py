"""
CS 6140 Project - Step 4: Support vector machine (scikit-learn)

Compares linear, polynomial and RBF kernels and tunes C (and gamma for RBF)
with stratified 5-fold CV, picks a decision threshold on training data, then
evaluates once on the 2016-2019 test set. Results go to results/svm/<tag>/.

Kernel SVM training grows roughly with the square of the number of samples,
so the SVM is tuned on 10k and trained on 20k trials from the 2005-2015 period.

Usage:
  python src/train_svm.py
  python src/train_svm.py --data-dir data/processed_with_enrollment --tag with_enrollment
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             balanced_accuracy_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.svm import SVC

SEED = 42
TUNE_SIZE = 10000
TRAIN_SIZE = 20000
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc",
           "balanced_accuracy": "balanced_accuracy"}

PARAM_GRID = [
    {"kernel": ["linear"], "C": [0.01, 0.1, 1], "class_weight": [None, "balanced"]},
    {"kernel": ["rbf"], "C": [0.1, 1, 10], "gamma": ["scale", 0.001],
     "class_weight": ["balanced"]},
    {"kernel": ["poly"], "degree": [2], "C": [0.1, 1], "class_weight": ["balanced"]},
]


def metrics_at(y, score, threshold):
    pred = (score >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "f1": f1_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
    }


def best_f1_threshold(y, score):
    prec, rec, thr = precision_recall_curve(y, score)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return thr[np.argmax(f1[:-1])]


def setting_name(p):
    name = p["kernel"]
    if p["kernel"] == "rbf":
        name += f", gamma={p['gamma']}"
    if p["kernel"] == "poly":
        name += f", degree={p['degree']}"
    return name + f", weight={p['class_weight'] or 'none'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    out = os.path.join("results", "svm", args.tag)
    os.makedirs(out, exist_ok=True)

    d = np.load(os.path.join(args.data_dir, "data.npz"), allow_pickle=True)
    X_all, y_all, X_test, y_test = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    with open(os.path.join(args.data_dir, "feature_names.json")) as f:
        names = json.load(f)

    # stratified samples from the training period (tune set is part of train set)
    X_train, _, y_train, _ = train_test_split(X_all, y_all, train_size=TRAIN_SIZE,
                                              stratify=y_all, random_state=SEED)
    X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=TUNE_SIZE,
                                            stratify=y_train, random_state=SEED)
    print(f"Tune {X_tune.shape}, train {X_train.shape}, test {X_test.shape}, "
          f"{y_train.mean():.1%} / {y_test.mean():.1%} terminated")

    # 1. choose kernel, C and gamma with 5-fold CV
    t0 = time.time()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(SVC(cache_size=1000), PARAM_GRID, scoring=SCORING,
                          refit="pr_auc", cv=cv, n_jobs=-1)
    search.fit(X_tune, y_tune)
    print(f"Grid search done in {time.time() - t0:.0f}s")

    res = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame({
        "setting": [setting_name(p) for p in res["params"]],
        "C": [p["C"] for p in res["params"]],
        **{f"cv_{m}": res[f"mean_test_{m}"] for m in SCORING},
        **{f"cv_{m}_std": res[f"std_test_{m}"] for m in SCORING},
        "fit_time_s": res["mean_fit_time"],
    }).sort_values("cv_pr_auc", ascending=False)
    table.to_csv(os.path.join(out, "cv_results.csv"), index=False)
    best = {k: v for k, v in search.best_params_.items()}
    best["class_weight"] = best["class_weight"] or "none"
    print("Best setting:", best, f"(CV PR-AUC {search.best_score_:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, m, title in zip(axes, ["pr_auc", "roc_auc"], ["PR-AUC", "ROC-AUC"]):
        for name, g in table.groupby("setting"):
            g = g.sort_values("C")
            ax.errorbar(g["C"], g[f"cv_{m}"], yerr=g[f"cv_{m}_std"], marker="o",
                        capsize=3, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("C")
        ax.set_title(f"5-fold CV {title}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cv_vs_C.png"), dpi=150)
    plt.close(fig)

    # 2. threshold on SVM scores from out-of-fold predictions on the train set
    t0 = time.time()
    model = clone(search.best_estimator_)
    oof = cross_val_predict(model, X_train, y_train, cv=cv,
                            method="decision_function", n_jobs=-1)
    thr = best_f1_threshold(y_train, oof)
    print(f"F1-best threshold (train, out-of-fold): {thr:.3f}  ({time.time() - t0:.0f}s)")

    # 3. train on the 20k train set and test on the full test period
    t0 = time.time()
    model.fit(X_train, y_train)
    score = model.decision_function(X_test)
    print(f"Final fit + test predictions in {time.time() - t0:.0f}s, "
          f"{model.n_support_.sum():,} support vectors")
    m0, mthr = metrics_at(y_test, score, 0.0), metrics_at(y_test, score, thr)

    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump({"best_params": best,
                   "cv_best": {m: float(table.iloc[0][f"cv_{m}"]) for m in SCORING},
                   "n_support_vectors": int(model.n_support_.sum()),
                   "test_at_0": m0, "test_at_tuned_threshold": mthr,
                   "random_pr_auc": float(y_test.mean())}, f, indent=2, default=float)
    # SVM gives scores (distance to the boundary), not probabilities.
    # ROC/PR only need the ranking, so these work for the final comparison.
    np.save(os.path.join(out, "test_scores.npy"), score)

    if best["kernel"] == "linear":
        coefs = pd.DataFrame({"feature": names, "coefficient": model.coef_.ravel()})
        coefs = coefs.reindex(coefs["coefficient"].abs().sort_values(ascending=False).index)
        coefs.to_csv(os.path.join(out, "coefficients.csv"), index=False)

    # ROC + PR, with logistic regression on the same plot if it has been run
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    curves = [("SVM", score)]
    lr_path = os.path.join("results", "logreg", args.tag, "test_probabilities.npy")
    if os.path.exists(lr_path):
        curves.append(("Logistic regression", np.load(lr_path)))
    for label, s in curves:
        fpr, tpr, _ = roc_curve(y_test, s)
        axes[0].plot(fpr, tpr, label=f"{label} (AUC = {roc_auc_score(y_test, s):.3f})")
        prec, rec, _ = precision_recall_curve(y_test, s)
        axes[1].plot(rec, prec, label=f"{label} (AP = {average_precision_score(y_test, s):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].axhline(y_test.mean(), color="k", ls="--", lw=1, label="random")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test)")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "roc_pr_test.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, (score >= thr).astype(int), ax=ax,
                                            display_labels=["Completed", "Terminated"],
                                            cmap="Blues", values_format=",d", colorbar=False)
    ax.set_title(f"SVM test set, threshold {thr:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "confusion_matrix_test.png"), dpi=150)
    plt.close(fig)

    print("\nCV results (top 5):")
    print(table.head(5)[["setting", "C", "cv_pr_auc", "cv_roc_auc"]].to_string(
        index=False, float_format="%.4f"))
    print("\nTEST SET (2016-2019)")
    print(f"{'':18s} {'thr 0':>9s} {'thr ' + format(thr, '.2f'):>9s}")
    for k in ["roc_auc", "pr_auc", "f1", "balanced_accuracy", "precision", "recall"]:
        print(f"{k:18s} {m0[k]:9.4f} {mthr[k]:9.4f}")
    print(f"(random PR-AUC = {y_test.mean():.4f})")
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
