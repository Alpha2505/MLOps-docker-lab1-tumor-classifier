"""Runtime configuration, read from environment variables.

Every setting can be overridden at `docker run` time with -e, e.g.
    docker run -e CV_FOLDS=10 -e SELECTION_METRIC=f1 lab1-tumor-classifier
"""
import os
from dataclasses import dataclass

VALID_METRICS = {"roc_auc", "f1", "accuracy", "recall", "precision"}


@dataclass(frozen=True)
class Config:
    test_size: float = 0.2
    random_seed: int = 42
    cv_folds: int = 5
    selection_metric: str = "roc_auc"
    output_dir: str = "artifacts"

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls(
            test_size=float(os.getenv("TEST_SIZE", cls.test_size)),
            random_seed=int(os.getenv("RANDOM_SEED", cls.random_seed)),
            cv_folds=int(os.getenv("CV_FOLDS", cls.cv_folds)),
            selection_metric=os.getenv("SELECTION_METRIC", cls.selection_metric),
            output_dir=os.getenv("OUTPUT_DIR", cls.output_dir),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not 0.05 <= self.test_size <= 0.5:
            raise ValueError(f"TEST_SIZE must be between 0.05 and 0.5, got {self.test_size}")
        if self.cv_folds < 2:
            raise ValueError(f"CV_FOLDS must be at least 2, got {self.cv_folds}")
        if self.selection_metric not in VALID_METRICS:
            raise ValueError(
                f"SELECTION_METRIC must be one of {sorted(VALID_METRICS)}, got {self.selection_metric!r}"
            )
