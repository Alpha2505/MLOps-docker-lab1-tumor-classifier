import numpy as np

from data import load_data, split_data


def test_dataset_shape():
    X, y, features, targets = load_data()
    assert X.shape == (569, 30)
    assert len(features) == 30
    assert targets == ["malignant", "benign"]


def test_split_is_stratified():
    X, y, _, _ = load_data()
    _, _, y_train, y_test = split_data(X, y, test_size=0.2, seed=42)
    assert abs(np.mean(y_train) - np.mean(y_test)) < 0.02


def test_split_is_reproducible():
    X, y, _, _ = load_data()
    a = split_data(X, y, 0.2, 7)[3]
    b = split_data(X, y, 0.2, 7)[3]
    assert np.array_equal(a, b)
