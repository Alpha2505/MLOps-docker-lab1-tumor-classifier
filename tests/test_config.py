import pytest

from config import Config


def test_defaults_are_valid():
    Config().validate()


def test_reads_env_overrides(monkeypatch):
    monkeypatch.setenv("CV_FOLDS", "3")
    monkeypatch.setenv("SELECTION_METRIC", "f1")
    cfg = Config.from_env()
    assert cfg.cv_folds == 3
    assert cfg.selection_metric == "f1"


@pytest.mark.parametrize("var,value", [
    ("TEST_SIZE", "0.9"),
    ("CV_FOLDS", "1"),
    ("SELECTION_METRIC", "not_a_metric"),
])
def test_rejects_bad_values(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(ValueError):
        Config.from_env()
