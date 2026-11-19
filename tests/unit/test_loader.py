import io

import pandas as pd
import pytest

from tabulaml.data import loader
from tabulaml.data.loader import DataLoadError, list_excel_sheets, read_tabular_file

DF = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def test_reads_comma_csv():
    out = read_tabular_file(io.BytesIO(DF.to_csv(index=False).encode()), "d.csv")
    assert out.shape == (3, 2)


def test_sniffs_semicolon_delimiter():
    out = read_tabular_file(io.BytesIO(DF.to_csv(index=False, sep=";").encode()), "d.csv")
    assert list(out.columns) == ["a", "b"]


def test_falls_back_to_latin1():
    raw = "name,city\nA,Colombo\nB,Kandy\nC,café\n".encode("latin-1")
    out = read_tabular_file(io.BytesIO(raw), "d.csv")
    assert out["city"].iloc[2] == "café"


def test_xlsx_roundtrip_and_sheets(tmp_path):
    path = tmp_path / "d.xlsx"
    with pd.ExcelWriter(path) as writer:
        DF.to_excel(writer, sheet_name="first", index=False)
        DF.assign(c=1).to_excel(writer, sheet_name="second", index=False)
    assert list_excel_sheets(path) == ["first", "second"]
    assert read_tabular_file(path, "d.xlsx").shape == (3, 2)
    assert read_tabular_file(path, "d.xlsx", sheet="second").shape == (3, 3)


def test_duplicate_column_names_are_made_unique():
    out = read_tabular_file(io.BytesIO(b"a,a,b\n1,2,3\n4,5,6\n"), "d.csv")
    assert list(out.columns) == ["a", "a_1", "b"]


def test_empty_file_raises():
    with pytest.raises(DataLoadError):
        read_tabular_file(io.BytesIO(b""), "d.csv")


def test_header_only_raises():
    with pytest.raises(DataLoadError):
        read_tabular_file(io.BytesIO(b"a,b\n"), "d.csv")


def test_single_column_raises():
    with pytest.raises(DataLoadError):
        read_tabular_file(io.BytesIO(b"a\n1\n2\n"), "d.csv")


def test_unsupported_extension_raises():
    with pytest.raises(DataLoadError):
        read_tabular_file(io.BytesIO(b"{}"), "d.json")


def test_too_many_cells_raises(monkeypatch):
    monkeypatch.setattr(loader, "MAX_CELLS", 4)
    with pytest.raises(DataLoadError):
        read_tabular_file(io.BytesIO(DF.to_csv(index=False).encode()), "d.csv")


def test_too_large_file_raises(monkeypatch):
    monkeypatch.setattr(loader, "MAX_FILE_MB", 0)
    with pytest.raises(DataLoadError):
        read_tabular_file(io.BytesIO(b"a,b\n1,2\n"), "d.csv")
