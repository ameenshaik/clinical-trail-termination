"""
CS 6140 Project - Step 3a: Logistic regression from scratch (NumPy)

Picks lambda on a validation split of the 2005-2015 trials, retrains on all
of them, evaluates once on the 2016-2019 test set, and checks the weights
against scikit-learn. Results go to results/logreg_scratch/.

Usage:
  python src/logreg_scratch.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = "data/processed"
OUT_DIR = "results/logreg_scratch"

LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
LEARNING_RATE = 0.1
BATCH_SIZE = 256
EPOCHS = 15
SEED = 42


def sigmoid(z):
    # clip so np.exp does not overflow for very negative z
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1), dtype=X.dtype), X])


def bce(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p)))


def train_sgd(X, y, lam, X_val=None, y_val=None, epochs=EPOCHS, lr=LEARNING_RATE,
              batch_size=BATCH_SIZE, seed=SEED):
    """Minibatch SGD for L2-regularized logistic regression. X already has the bias column."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d)
    history = {"train_bce": [], "val_bce": []}

    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            Xb, yb = X[idx], y[idx]
            p = sigmoid(Xb @ w)
            grad = -(Xb.T @ (yb - p)) / len(idx)
            reg = lam * w
            reg[0] = 0.0                      # do not shrink the bias
            w -= lr * (grad + reg)

        history["train_bce"].append(bce(y, sigmoid(X @ w)))
        if X_val is not None:
            history["val_bce"].append(bce(y_val, sigmoid(X_val @ w)))

    return w, history


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d = np.load(os.path.join(DATA_DIR, "data.npz"), allow_pickle=True)
    X_train = add_bias(d["X_train"].astype(np.float64))
    X_test = add_bias(d["X_test"].astype(np.float64))
    y_train = d["y_train"].astype(np.float64)
    y_test = d["y_test"].astype(np.float64)
    with open(os.path.join(DATA_DIR, "feature_names.json")) as f:
        names = ["bias"] + json.load(f)

    #  1. train / validation split inside the training period 
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED)

    #  2. try each lambda
    runs = {}
    for lam in LAMBDAS:
        w, hist = train_sgd(X_tr, y_tr, lam, X_val, y_val)
        val_prob = sigmoid(X_val @ w)
        runs[lam] = {
            "history": hist,
            "val_bce": hist["val_bce"][-1],
            "val_roc_auc": roc_auc_score(y_val, val_prob),
            "weight_norm": float(np.linalg.norm(w[1:])),
        }
        print(f"lambda={lam:<7g} val BCE={runs[lam]['val_bce']:.4f}  "
              f"val ROC-AUC={runs[lam]['val_roc_auc']:.4f}  ||w||={runs[lam]['weight_norm']:.2f}")

    best_lam = min(runs, key=lambda l: runs[l]["val_bce"])
    print(f"\nBest lambda by validation BCE: {best_lam}")

    # loss curves for every lambda
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for lam, r in runs.items():
        ep = range(1, EPOCHS + 1)
        axes[0].plot(ep, r["history"]["train_bce"], label=f"lambda={lam:g}")
        axes[1].plot(ep, r["history"]["val_bce"], label=f"lambda={lam:g}")
    axes[0].set_title("Train BCE")
    axes[1].set_title("Validation BCE")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean BCE")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "loss_curves.png"), dpi=150)
    plt.close(fig)

    #  3. retrain on the full training period, test once 
    w, _ = train_sgd(X_train, y_train, best_lam)
    test_prob = sigmoid(X_test @ w)
    test_pred = (test_prob >= 0.5).astype(int)

    error_rate = float(np.mean(test_pred != y_test))   
    baseline_error = float(y_test.mean())              
    results = {
        "best_lambda": best_lam,
        "lambdas": {str(l): {k: v for k, v in r.items() if k != "history"} for l, r in runs.items()},
        "test_error_rate_at_0.5": error_rate,
        "baseline_error_always_completed": baseline_error,
        "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
        "test_pr_auc": float(average_precision_score(y_test, test_prob)),
        "test_recall_at_0.5": float(test_pred[y_test == 1].mean()),
    }

    #  4. sanity check against scikit-learn 
    
    if best_lam > 0:
        C = 1.0 / (best_lam * X_train.shape[0])
        sk = LogisticRegression(C=C, max_iter=3000)
    else:
        sk = LogisticRegression(C=1e6, max_iter=3000)   # (almost) no regularization
    sk.fit(X_train[:, 1:], y_train)
    sk_w = np.concatenate([sk.intercept_, sk.coef_.ravel()])
    sk_prob = sk.predict_proba(X_test[:, 1:])[:, 1]
    results["sklearn_check"] = {
        "C_used": C if best_lam > 0 else 1e6,
        "sklearn_test_roc_auc": float(roc_auc_score(y_test, sk_prob)),
        "weight_correlation_ours_vs_sklearn": float(np.corrcoef(w, sk_w)[0, 1]),
    }

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    order = np.argsort(-np.abs(w[1:]))[:10] + 1
    print("\nTEST SET (2016-2019)")
    print(f"Error rate at 0.5      : {error_rate:.4f}  (always 'Completed' gets {baseline_error:.4f})")
    print(f"Recall at 0.5          : {results['test_recall_at_0.5']:.4f}")
    print(f"ROC-AUC                : {results['test_roc_auc']:.4f}")
    print(f"PR-AUC                 : {results['test_pr_auc']:.4f}  (random = {y_test.mean():.4f})")
    print(f"sklearn ROC-AUC (same C): {results['sklearn_check']['sklearn_test_roc_auc']:.4f}")
    print(f"Weight correlation     : {results['sklearn_check']['weight_correlation_ours_vs_sklearn']:.4f}")
    print("\nLargest weights (ours):")
    for i in order:
        print(f"  {names[i]:45s} {w[i]:+.3f}")
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
