"""
CS 6140 Project - Step 3b: Logistic regression (scikit-learn)

Tunes C, L1/L2 and class weighting with stratified 5-fold CV on the
2005-2015 trials, picks a decision threshold on training data, then
evaluates once on the 2016-2019 test set. Results go to results/logreg/<tag>/.

Usage:
  python src/train_logreg.py
  python src/train_logreg.py --data-dir data/processed_with_enrollment --tag with_enrollment
"""
import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             balanced_accuracy_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.base import clone
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     train_test_split)

SEED = 42
# Tuning on all 124k trials was too slow, so we tune on 30k of them
# and then train the best model on all the trials.
TUNE_SIZE = 30000
C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc",
           "f1": "f1", "balanced_accuracy": "balanced_accuracy"}

# Newer scikit-learn versions (1.8+) use l1_ratio instead of penalty.
NEW_SKLEARN = tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 8)


def param_grid():
    l1 = {"l1_ratio": [1.0]} if NEW_SKLEARN else {"penalty": ["l1"]}
    l2 = {"l1_ratio": [0.0]} if NEW_SKLEARN else {"penalty": ["l2"]}
    weights = [None, "balanced"]
    # L2 is fast so we try all C values with and without class weights.
    # L1 is slow so we try fewer C values for it.
    # We use the saga solver for L1 because liblinear gave wrong intercepts.
    return [{"C": [0.01, 0.1, 1], "class_weight": [None], "solver": ["saga"], **l1},
            {"C": C_VALUES, "class_weight": weights, "solver": ["lbfgs"], **l2}]


def penalty_name(params):
    if "l1_ratio" in params:
        return "L1" if params["l1_ratio"] == 1.0 else "L2"
    return params["penalty"].upper()


def metrics_at(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        "f1": f1_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
    }


def best_f1_threshold(y, prob):
    prec, rec, thr = precision_recall_curve(y, prob)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return thr[np.argmax(f1[:-1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    out = os.path.join("results", "logreg", args.tag)
    os.makedirs(out, exist_ok=True)

    d = np.load(os.path.join(args.data_dir, "data.npz"), allow_pickle=True)
    X_train, y_train, X_test, y_test = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    with open(os.path.join(args.data_dir, "feature_names.json")) as f:
        names = json.load(f)
    print(f"Train {X_train.shape}, test {X_test.shape}, "
          f"{y_train.mean():.1%} / {y_test.mean():.1%} terminated")

    # 1. find the best settings using 5-fold cross validation
    X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=TUNE_SIZE,
                                            stratify=y_train, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(LogisticRegression(max_iter=3000, random_state=SEED),
                          param_grid(), scoring=SCORING, refit="pr_auc", cv=cv, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        search.fit(X_tune, y_tune)

    res = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame({
        "penalty": [penalty_name(p) for p in res["params"]],
        "C": [p["C"] for p in res["params"]],
        "class_weight": [p["class_weight"] or "none" for p in res["params"]],
        **{f"cv_{m}": res[f"mean_test_{m}"] for m in SCORING},
        **{f"cv_{m}_std": res[f"std_test_{m}"] for m in SCORING},
    }).sort_values("cv_pr_auc", ascending=False)
    table.to_csv(os.path.join(out, "cv_results.csv"), index=False)

    best = {"penalty": penalty_name(search.best_params_), "C": search.best_params_["C"],
            "class_weight": search.best_params_["class_weight"] or "none"}
    print("Best setting:", best, f"(CV PR-AUC {search.best_score_:.4f}, on {TUNE_SIZE:,} trials)")

    # plot how the CV score changes with C
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m, title in zip(axes, ["pr_auc", "roc_auc", "f1"],
                            ["PR-AUC", "ROC-AUC", "F1 at threshold 0.5"]):
        for (pen, cw), g in table.groupby(["penalty", "class_weight"]):
            g = g.sort_values("C")
            ax.errorbar(g["C"], g[f"cv_{m}"], yerr=g[f"cv_{m}_std"], marker="o",
                        capsize=3, label=f"{pen}, weight={cw}")
        ax.set_xscale("log")
        ax.set_xlabel("C")
        ax.set_title(f"5-fold CV {title}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cv_vs_C.png"), dpi=150)
    plt.close(fig)

    # 2. pick the threshold that gives the best F1 on the training data
    best_model = clone(search.best_estimator_)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        oof = cross_val_predict(best_model, X_train, y_train, cv=cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
    thr = best_f1_threshold(y_train, oof)
    print(f"F1-best threshold (train, out-of-fold): {thr:.3f}")

    # 3. train the best model on all training trials and test it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = best_model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    m05, mthr = metrics_at(y_test, prob, 0.5), metrics_at(y_test, prob, thr)

    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump({"best_params": best,
                   "cv_best": {m: float(table.iloc[0][f"cv_{m}"]) for m in SCORING},
                   "test_at_0.5": m05, "test_at_tuned_threshold": mthr,
                   "random_pr_auc": float(y_test.mean())}, f, indent=2, default=float)
    np.save(os.path.join(out, "test_probabilities.npy"), prob)   # saved to compare models later

    coefs = pd.DataFrame({"feature": names, "coefficient": model.coef_.ravel()})
    coefs["odds_ratio"] = np.exp(coefs["coefficient"])
    coefs = coefs.reindex(coefs["coefficient"].abs().sort_values(ascending=False).index)
    coefs.to_csv(os.path.join(out, "coefficients.csv"), index=False)

    # save the plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fpr, tpr, _ = roc_curve(y_test, prob)
    axes[0].plot(fpr, tpr, label=f"AUC = {m05['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test)")
    prec, rec, _ = precision_recall_curve(y_test, prob)
    axes[1].plot(rec, prec, label=f"AP = {m05['pr_auc']:.3f}")
    axes[1].axhline(y_test.mean(), color="k", ls="--", lw=1, label="random")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test)")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "roc_pr_test.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, (prob >= thr).astype(int), ax=ax,
                                            display_labels=["Completed", "Terminated"],
                                            cmap="Blues", values_format=",d", colorbar=False)
    ax.set_title(f"Test set, threshold {thr:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "confusion_matrix_test.png"), dpi=150)
    plt.close(fig)

    top = coefs.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["feature"], top["coefficient"],
            color=["tab:red" if c > 0 else "tab:blue" for c in top["coefficient"]])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Coefficient (red = higher termination risk)")
    ax.set_title("Top 20 logistic regression coefficients")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "top_coefficients.png"), dpi=150)
    plt.close(fig)

    print("\nTEST SET (2016-2019)")
    print(f"{'':18s} {'thr 0.5':>9s} {'thr ' + format(thr, '.2f'):>9s}")
    for k in ["roc_auc", "pr_auc", "f1", "balanced_accuracy", "precision", "recall"]:
        print(f"{k:18s} {m05[k]:9.4f} {mthr[k]:9.4f}")
    print(f"(random PR-AUC = {y_test.mean():.4f})")
    print("\nTop 10 coefficients:")
    print(coefs.head(10).to_string(index=False, float_format="%.3f"))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
