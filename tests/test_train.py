import json
import os

import pytest

from config import Config
from data import load_data
from predict import load_bundle, predict
from train import run, select_best


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    cfg = Config(cv_folds=3, output_dir=str(out))
    return out, run(cfg)


def test_select_best_picks_highest_mean():
    results = {"a": (0.90, 0.01), "b": (0.97, 0.02), "c": (0.95, 0.00)}
    assert select_best(results) == "b"


def test_artifacts_written(trained):
    out, _ = trained
    assert os.path.exists(out / "tumor_classifier.joblib")
    with open(out / "metrics.json") as f:
        report = json.load(f)
    assert set(report["cv_results"]) == {"logistic_regression", "svm_rbf", "gradient_boosting"}


def test_model_meets_quality_bar(trained):
    _, report = trained
    m = report["test_metrics"]
    assert m["roc_auc"] >= 0.95
    assert m["malignant_recall"] >= 0.90


def test_saved_model_predicts(trained):
    out, _ = trained
    bundle = load_bundle(str(out))
    X, _, _, _ = load_data()
    preds = predict(bundle, X[:3])
    assert len(preds) == 3
    assert all(label in ("malignant", "benign") for label, _ in preds)
    assert all(0.5 <= conf <= 1.0 for _, conf in preds)
