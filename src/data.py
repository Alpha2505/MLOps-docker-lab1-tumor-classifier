"""Data loading and splitting for the Breast Cancer Wisconsin (Diagnostic) dataset."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data():
    """Return features X, labels y (1 = benign, 0 = malignant), and feature/target names."""
    ds = load_breast_cancer()
    return ds.data, ds.target, list(ds.feature_names), list(ds.target_names)


def split_data(X, y, test_size, seed):
    """Stratified split so both sets keep the same malignant/benign ratio."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
