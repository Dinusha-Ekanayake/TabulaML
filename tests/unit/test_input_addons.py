"""E4: JSON / Parquet upload, apostrophe thousands marks and small-data warnings."""

import io
import json

import numpy as np
import pandas as pd
import pytest

from tabulaml.core.types import ProblemType
from tabulaml.data.loader import DataLoadError, read_tabular_file
from tabulaml.preprocessing.cleaning import coerce_numeric
from tabulaml.workflow import Roles, auto_configure

FRAME = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "y": [0.5, 1.5, 2.5]})


@pytest.mark.parametrize(
    "payload",
    [
        json.dumps(FRAME.to_dict("records")),
        FRAME.to_json(orient="records", lines=True),
        json.dumps(FRAME.to_dict("list")),
        json.dumps({"data": FRAME.to_dict("records")}),
    ],
)
def test_json_shapes(payload):
    df = read_tabular_file(io.BytesIO(payload.encode()), "data.json")
    assert list(df.columns) == ["a", "b", "y"] and len(df) == 3


def test_bad_json_and_unknown_type():
    with pytest.raises(DataLoadError, match="JSON"):
        read_tabular_file(io.BytesIO(b"{not json"), "data.json")
    with pytest.raises(DataLoadError, match="Parquet"):
        read_tabular_file(io.BytesIO(b"garbage"), "data.parquet")
    with pytest.raises(DataLoadError, match="Unsupported"):
        read_tabular_file(io.BytesIO(b"abc"), "data.pdf")


def test_parquet_round_trip():
    buffer = io.BytesIO()
    FRAME.to_parquet(buffer)
    df = read_tabular_file(io.BytesIO(buffer.getvalue()), "data.parquet")
    pd.testing.assert_frame_equal(df, FRAME)


def test_thousands_marks():
    swiss = coerce_numeric(pd.Series(["1'234.5", "1’234.5", "12'345'678"]))
    assert swiss.tolist() == [1234.5, 1234.5, 12345678.0]
    spaced = coerce_numeric(pd.Series(["1 234,5", "1 234,5"]), decimal=",")
    assert spaced.tolist() == [1234.5, 1234.5]
    assert np.isnan(coerce_numeric(pd.Series(["12'34"]))[0])  # not a thousands group


def test_small_dataset_warnings():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=60), "z": rng.normal(size=60)})
    df["y"] = df["x"] * 2 + rng.normal(size=60)
    auto = auto_configure(df, Roles(target="y", problem_type=ProblemType.REGRESSION, cv_folds=3))
    codes = {i.code for i in auto.split_issues}
    assert {"small_dataset", "tiny_test_set"} <= codes
