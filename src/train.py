"""Train several candidate models, pick the best by cross-validation,
evaluate it on a held-out test set, and save the model plus a metrics report."""
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import Config
from data import load_data, split_data
from models import build_candidates


def cross_validate_candidates(candidates, X_train, y_train, cfg):
    """Return {model_name: (mean_score, std_score)} using stratified k-fold CV."""
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_seed)
    results = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=cfg.selection_metric)
        results[name] = (float(np.mean(scores)), float(np.std(scores)))
    return results


def select_best(cv_results):
    return max(cv_results, key=lambda name: cv_results[name][0])


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        # In this dataset label 0 = malignant. Missing a malignant tumour is the
        # costly error, so report how many malignant cases were caught.
        "malignant_recall": round(float(tn / (tn + fp)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def run(cfg: Config) -> dict:
    X, y, feature_names, target_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, cfg.test_size, cfg.random_seed)

    candidates = build_candidates(cfg.random_seed)
    cv_results = cross_validate_candidates(candidates, X_train, y_train, cfg)

    print(f"\nCross-validation ({cfg.cv_folds}-fold, metric = {cfg.selection_metric}):")
    for name, (mean, std) in sorted(cv_results.items(), key=lambda kv: -kv[1][0]):
        print(f"  {name:<22} {mean:.4f} ± {std:.4f}")

    best_name = select_best(cv_results)
    best_model = candidates[best_name].fit(X_train, y_train)
    test_metrics = evaluate(best_model, X_test, y_test)

    os.makedirs(cfg.output_dir, exist_ok=True)
    model_path = os.path.join(cfg.output_dir, "tumor_classifier.joblib")
    joblib.dump({"model": best_model, "feature_names": feature_names,
                 "target_names": target_names}, model_path)

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "config": cfg.__dict__,
        "cv_results": {k: {"mean": round(m, 4), "std": round(s, 4)} for k, (m, s) in cv_results.items()},
        "selected_model": best_name,
        "test_metrics": test_metrics,
    }
    with open(os.path.join(cfg.output_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSelected model: {best_name}")
    print("Held-out test metrics:")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "malignant_recall"):
        print(f"  {k:<17} {test_metrics[k]:.4f}")
    cm = test_metrics["confusion_matrix"]
    print(f"  confusion         TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    print(f"\nSaved model   -> {model_path}")
    print(f"Saved metrics -> {os.path.join(cfg.output_dir, 'metrics.json')}")
    return report


if __name__ == "__main__":
    run(Config.from_env())
