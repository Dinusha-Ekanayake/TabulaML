import io

import joblib
import numpy as np
import pandas as pd
import pytest

from tabulaml.data.profiler import profile_dataset
from tabulaml.data.samples import load_sample
from tabulaml.data.validation import detect_leakage
from tabulaml.inference.persistence import bundle_bytes, load_bundle, save_bundle
from tabulaml.inference.predictor import Predictor, make_input_spec
from tabulaml.preprocessing.cleaning import clean_frame
from tabulaml.preprocessing.recommender import recommend_config
from tabulaml.training.trainer import train_models


def _trained(key, model="rf"):
    df, card = load_sample(key)
    clean = clean_frame(df)
    leaks = detect_leakage(clean, card.target, card.problem_type)
    cfg = recommend_config(
        profile_dataset(clean), card.target, card.problem_type, leaks, y=df[card.target]
    )
    cfg = cfg.__class__(**{**cfg.__dict__, "cv_folds": 2})
    run = train_models(df, cfg, [model])
    result = run.result_for(model)
    spec = make_input_spec(df[list(run.feature_columns)], cfg)
    return df, run, result, Predictor(result.pipeline, cfg, run.class_labels, spec), spec


@pytest.fixture(scope="module")
def tea():
    return _trained("lk_tea_auction_catalogue", "logreg")


def test_input_spec_has_defaults(tea):
    _, run, _, predictor, _ = tea
    spec = {s["name"]: s for s in predictor.input_spec()}
    assert set(spec) == set(run.feature_columns)
    assert spec["fx_usd"]["type"] == "numeric" and spec["fx_usd"]["default"] > 200
    assert "nuwara_eliya" in spec["region"]["options"]
    assert spec["tier"]["type"] == "categorical"


def test_single_prediction_with_unseen_category_and_missing_value(tea):
    _, _, _, predictor, spec = tea
    record = {s["name"]: s["default"] for s in spec}
    record["region"] = "Atlantis"
    record["fx_usd"] = None
    out = predictor.predict_one(record)
    assert out["prediction"] in {"dust", "high_grown", "low_grown", "off_grade"}
    assert sum(out["probabilities"].values()) == pytest.approx(1.0)


def test_batch_with_shuffled_and_extra_columns(tea):
    df, run, _, predictor, _ = tea
    batch = df.head(20).copy()
    batch = batch[list(reversed(batch.columns))].assign(junk=1)
    issues = {i.code for i in predictor.validate_input(batch)}
    assert "extra_columns" in issues
    out = predictor.predict_batch(batch)
    assert {"prediction", "proba_dust", "proba_low_grown"} <= set(out.columns)
    assert len(out) == 20 and "prediction" not in batch.columns  # input not mutated


def test_batch_missing_column_is_an_error(tea):
    df, _, _, predictor, _ = tea
    bad = df.head(5).drop(columns=["fx_usd"])
    assert "missing_columns" in {i.code for i in predictor.validate_input(bad)}
    with pytest.raises(ValueError, match="fx_usd"):
        predictor.predict_batch(bad)
    assert "empty_input" in {i.code for i in predictor.validate_input(df.head(0))}


def test_save_load_roundtrip_gives_identical_predictions(tea, tmp_path):
    df, run, result, predictor, spec = tea
    path = save_bundle(tmp_path, result, run, spec)
    loaded = load_bundle(path)
    a = predictor.predict_batch(df.head(30))["prediction"]
    b = loaded.predict_batch(df.head(30))["prediction"]
    pd.testing.assert_series_equal(a, b)
    from_bytes = load_bundle(bundle_bytes(result, run, spec))
    assert from_bytes.class_labels == ("dust", "high_grown", "low_grown", "off_grade")


def test_load_rejects_foreign_pickles(tmp_path):
    buffer = io.BytesIO()
    joblib.dump({"hello": 1}, buffer)
    with pytest.raises(ValueError, match="Not a TabulaML"):
        load_bundle(buffer.getvalue())


def test_regression_predictor_returns_floats_on_original_scale():
    df, _, _, predictor, spec = _trained("lk_colombo_dengue")
    out = predictor.predict_one({s["name"]: s["default"] for s in spec})
    assert isinstance(out["prediction"], float) and out["probabilities"] is None
    assert 0 < out["prediction"] < 2000  # weekly dengue cases, on the original scale
    assert np.isfinite(predictor.predict_batch(df.head(10))["prediction"].astype(float)).all()


def test_pickle_and_joblib_bundles_load_the_same_model(tea):
    df, run, result, predictor, spec = tea
    as_joblib = bundle_bytes(result, run, spec, fmt="joblib")
    as_pickle = bundle_bytes(result, run, spec, fmt="pickle")
    assert as_joblib[:2] != as_pickle[:2] or len(as_joblib) != len(
        as_pickle
    )  # really different files
    a = load_bundle(as_joblib).predict_batch(df.head(20))["prediction"]
    b = load_bundle(as_pickle).predict_batch(df.head(20))["prediction"]
    pd.testing.assert_series_equal(a, b)
    with pytest.raises(ValueError, match="Unknown bundle format"):
        bundle_bytes(result, run, spec, fmt="zip")


def test_pickle_bundle_from_disk(tea, tmp_path):
    _, run, result, _, spec = tea
    path = tmp_path / "model.pkl"
    path.write_bytes(bundle_bytes(result, run, spec, fmt="pickle"))
    assert load_bundle(path).class_labels == run.class_labels
