"""Candidate models. Each is a Pipeline so scaling is fitted only on training folds
and is saved together with the model (no train/serve skew)."""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_candidates(seed: int) -> dict:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=seed)),
        ]),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=seed)),
        ]),
        "gradient_boosting": Pipeline([
            ("clf", GradientBoostingClassifier(random_state=seed)),
        ]),
    }
