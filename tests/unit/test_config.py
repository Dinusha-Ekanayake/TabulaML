import importlib

import tabulaml.config as config


def test_cloud_mode_lowers_limits(monkeypatch):
    monkeypatch.setenv("TABULAML_CLOUD", "1")
    cloud = importlib.reload(config)
    assert cloud.CLOUD_MODE and cloud.MAX_CELLS == 400_000 and cloud.SVM_MAX_ROWS == 5_000
    monkeypatch.setenv("TABULAML_CLOUD", "0")
    local = importlib.reload(config)
    assert not local.CLOUD_MODE and local.MAX_CELLS == 1_000_000
