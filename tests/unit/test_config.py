import importlib

import tabulaml.config as config


def test_cloud_mode_lowers_limits(monkeypatch):
    monkeypatch.setenv("TABULAML_CLOUD", "1")
    cloud = importlib.reload(config)
    assert cloud.CLOUD_MODE and cloud.MAX_CELLS == 400_000 and cloud.SVM_MAX_ROWS == 5_000
    monkeypatch.setenv("TABULAML_CLOUD", "0")
    local = importlib.reload(config)
    assert not local.CLOUD_MODE and local.MAX_CELLS == 1_000_000


def test_package_version_matches_pyproject():
    import tomllib
    from pathlib import Path

    import tabulaml

    pyproject = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    from tabulaml.config import APP_VERSION

    assert tabulaml.__version__ == pyproject["project"]["version"] == APP_VERSION
