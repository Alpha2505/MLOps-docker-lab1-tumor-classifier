"""Load the saved model and classify a few samples as a smoke test.

Usage (inside the container, after training):
    python predict.py            # predicts 5 samples
    python predict.py 10         # predicts 10 samples
"""
import os
import sys

import joblib
import numpy as np

from data import load_data


def load_bundle(output_dir="artifacts"):
    path = os.path.join(output_dir, "tumor_classifier.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model at {path}. Run train.py first.")
    return joblib.load(path)


def predict(bundle, X):
    model = bundle["model"]
    labels = [bundle["target_names"][i] for i in model.predict(X)]
    confidence = model.predict_proba(X).max(axis=1)
    return [(label, float(c)) for label, c in zip(labels, confidence)]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    bundle = load_bundle(os.getenv("OUTPUT_DIR", "artifacts"))
    X, y, _, target_names = load_data()
    idx = np.random.default_rng(0).choice(len(X), size=n, replace=False)
    for i, (label, conf) in zip(idx, predict(bundle, X[idx])):
        actual = target_names[y[i]]
        mark = "OK" if label == actual else "WRONG"
        print(f"sample {i:>3}: predicted={label:<9} ({conf:.1%})  actual={actual:<9} {mark}")
