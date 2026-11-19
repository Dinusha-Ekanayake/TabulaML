"""Evidence that ONE unchanged code path handles very different tabular datasets.

recommend (Automatic mode) -> train [rf, lgbm, logreg/ridge] -> predict a batch, on public
sklearn datasets, the bundled Sri Lankan samples and seeded random "chaos" frames with random
column types, cardinalities and missingness.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine

from tabulaml.core.types import ProblemType
from tabulaml.data.profiler import profile_dataset
from tabulaml.data.samples import list_samples, load_sample
from tabulaml.data.validation import detect_leakage, validate_target
from tabulaml.inference.predictor import Predictor, make_input_spec
from tabulaml.preprocessing.cleaning import clean_frame
from tabulaml.preprocessing.recommender import recommend_config
from tabulaml.training.trainer import train_models

CLS, REG = ProblemType.CLASSIFICATION, ProblemType.REGRESSION
pytestmark = pytest.mark.usefixtures("fast_models")


def chaos_frame(seed: int) -> tuple[pd.DataFrame, str, ProblemType]:
    """Random mixed-type frame; the target depends on a few random columns."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(150, 400))
    df = pd.DataFrame(index=range(n))
    signal = np.zeros(n)
    for j in range(int(rng.integers(4, 10))):
        kind = rng.choice(["num", "skewed", "cat", "highcat", "bool", "date", "id", "const"])
        name = f"{kind}_{j}"
        if kind in ("num", "skewed"):
            values = rng.normal(size=n) if kind == "num" else rng.lognormal(0, 1.2, n)
            signal += rng.normal() * (values - values.mean()) / (values.std() + 1e-9)
            df[name] = values
        elif kind in ("cat", "highcat"):
            levels = [f"L{k}" for k in range(int(rng.integers(2, 6 if kind == "cat" else 80)))]
            values = rng.choice(levels, n)
            effect = {lev: rng.normal() for lev in levels}
            signal += np.array([effect[v] for v in values])
            df[name] = [v.lower() if rng.random() < 0.1 else v for v in values]
        elif kind == "bool":
            df[name] = rng.choice(["yes", "No", "Y"], n)
        elif kind == "date":
            df[name] = (
                pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
            ).astype(str)
        elif kind == "id":
            df[name] = [f"X{seed}{i:05d}" for i in range(n)]
        else:
            df[name] = "same"
        missing = rng.random(n) < rng.uniform(0, 0.4)
        df.loc[missing & (rng.random() < 0.7), name] = np.nan
    df["anchor"] = rng.normal(size=n)  # guarantees at least one usable feature
    signal += df["anchor"].to_numpy()
    if seed % 2 == 0:
        df["target"] = np.where(
            signal + rng.normal(scale=0.5, size=n) > np.median(signal), "pos", "neg"
        )
        return df, "target", CLS
    df["target"] = signal + rng.normal(scale=0.5, size=n)
    return df, "target", REG


def _sklearn(loader, ptype):
    frame = loader(as_frame=True).frame.rename(columns={"target": "y"})
    return frame, "y", ptype


def _cases():
    cases = [
        ("breast_cancer", *_sklearn(load_breast_cancer, CLS)),
        ("wine_multiclass", *_sklearn(load_wine, CLS)),
        ("diabetes", *_sklearn(load_diabetes, REG)),
    ]
    for card in list_samples():
        df, card = load_sample(card.key)
        cases.append((card.key, df, card.target, card.problem_type))
    cases += [(f"chaos_{s}", *chaos_frame(s)) for s in range(6)]
    return cases


@pytest.mark.parametrize(("name", "df", "target", "ptype"), _cases(), ids=[c[0] for c in _cases()])
def test_same_code_trains_and_predicts_on_any_dataset(name, df, target, ptype):
    assert not [i for i in validate_target(df, target, ptype, 3) if i.level == "error"], name
    clean = clean_frame(df)
    leaks = detect_leakage(clean, target, ptype)
    cfg = recommend_config(profile_dataset(clean), target, ptype, leaks, y=df[target])
    cfg = cfg.__class__(**{**cfg.__dict__, "cv_folds": 3})
    linear = "logreg" if ptype == CLS else "ridge"
    run = train_models(df, cfg, ["rf", "lgbm", linear])
    errors = {r.key: r.error for r in run.results if r.status != "ok"}
    assert not errors, (name, errors)
    best = run.best()
    spec = make_input_spec(df[list(run.feature_columns)], cfg)
    scored = Predictor(best.pipeline, cfg, run.class_labels, spec).predict_batch(df.head(25))
    assert scored["prediction"].notna().all(), name
