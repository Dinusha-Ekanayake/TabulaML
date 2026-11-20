"""Robust reading of user-supplied tabular files (CSV / XLSX / XLS / JSON / Parquet)."""

from __future__ import annotations

import csv
import io
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from tabulaml.config import MAX_CELLS, MAX_FILE_MB

logger = logging.getLogger(__name__)

CSV_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1")
EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")


class DataLoadError(ValueError):
    """Raised when a file cannot be turned into a usable DataFrame."""


def _to_bytes(file_obj: Any) -> bytes:
    if isinstance(file_obj, (str, Path)):
        return Path(file_obj).read_bytes()
    if hasattr(file_obj, "getvalue"):
        return file_obj.getvalue()
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    return file_obj.read()


def _dedupe_columns(columns: list[Any]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for col in columns:
        name = "unnamed" if pd.isna(col) else (str(col).strip() or "unnamed")
        count = seen.get(name, 0)
        seen[name] = count + 1
        out.append(name if count == 0 else f"{name}_{count}")
    return out


def _sniff_delimiter(text: str) -> str:
    try:
        return csv.Sniffer().sniff(text[:10_000], delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _read_csv_body(raw: bytes, sep: str, encoding: str, names: list[str]) -> pd.DataFrame:
    """Read with explicit names. Rows wider than the header are an error, never silently
    shifted into the index (misaligning every column) or truncated (losing data)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.ParserWarning)
        try:
            return pd.read_csv(
                io.BytesIO(raw), sep=sep, encoding=encoding, header=0, names=names, index_col=False
            )
        except pd.errors.ParserWarning as exc:
            raise DataLoadError(
                "Some rows have more values than there are column names. "
                "Check the delimiter and quoting of the file."
            ) from exc


def _read_csv(raw: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            # Read the raw header first: pandas silently mangles duplicate names ("a.1").
            sep = _sniff_delimiter(raw.decode(encoding))
            header = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=encoding, header=None, nrows=1)
            names = _dedupe_columns(list(header.iloc[0]))
            return _read_csv_body(raw, sep, encoding, names)
        except UnicodeDecodeError as exc:
            last_error = exc
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DataLoadError(f"Could not parse CSV: {exc}") from exc
    raise DataLoadError(f"Could not decode file with {CSV_ENCODINGS}: {last_error}")


def list_excel_sheets(file_obj: Any) -> list[str]:
    try:
        return list(pd.ExcelFile(io.BytesIO(_to_bytes(file_obj))).sheet_names)
    except Exception as exc:  # noqa: BLE001 - surface any reader failure uniformly
        raise DataLoadError(f"Could not read Excel file: {exc}") from exc


def _read_excel(raw: bytes, sheet: str | None) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(raw), sheet_name=sheet or 0)
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Could not read Excel file: {exc}") from exc


def _read_json(raw: bytes) -> pd.DataFrame:
    """Records (``[{...}, ...]``), JSON lines, or a ``{"column": [...]}`` object."""
    text = raw.decode("utf-8-sig", errors="replace").strip()
    try:
        if text.startswith("["):
            return pd.DataFrame(json.loads(text))
        lines = text.splitlines()
        if len(lines) > 1 and lines[0].startswith("{") and lines[0].rstrip().endswith("}"):
            return pd.read_json(io.StringIO(text), lines=True)  # JSON lines
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return pd.DataFrame(data["data"])  # common API style: {"data": [...]}
        return pd.DataFrame(data)
    except (ValueError, TypeError) as exc:
        raise DataLoadError(
            "Could not read JSON. Use a list of records, JSON lines, or {column: [values]}."
        ) from exc


def _read_parquet(raw: bytes) -> pd.DataFrame:
    try:
        return pd.read_parquet(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001 - surface any reader failure uniformly
        raise DataLoadError(f"Could not read Parquet file: {exc}") from exc


def _check_limits(raw: bytes, df: pd.DataFrame) -> None:
    if df.empty:
        raise DataLoadError("The file contains no rows.")
    if df.shape[1] < 2:
        raise DataLoadError("The file needs at least two columns (features and a target).")
    if df.size > MAX_CELLS:
        raise DataLoadError(f"Dataset has {df.size:,} cells; the limit is {MAX_CELLS:,}.")


def read_tabular_file(file_obj: Any, filename: str, sheet: str | None = None) -> pd.DataFrame:
    """Read an upload. Duplicate column names are made unique, not rejected."""
    raw = _to_bytes(file_obj)
    if not raw:
        raise DataLoadError("The file is empty.")
    if len(raw) > MAX_FILE_MB * 1024 * 1024:
        raise DataLoadError(f"File is larger than {MAX_FILE_MB} MB.")
    suffix = Path(filename).suffix.lower()
    if suffix in EXCEL_SUFFIXES:
        df = _read_excel(raw, sheet)
    elif suffix in (".csv", ".txt", ".tsv", ""):
        df = _read_csv(raw)
    elif suffix in (".json", ".jsonl"):
        df = _read_json(raw)
    elif suffix == ".parquet":
        df = _read_parquet(raw)
    else:
        raise DataLoadError(
            f"Unsupported file type '{suffix}'. Use CSV, TXT, XLSX, XLS, JSON or Parquet."
        )
    original = list(df.columns)
    df.columns = _dedupe_columns(original)
    if [str(c) for c in original] != list(df.columns):
        logger.warning("Column names were made unique/non-empty on load.")
    _check_limits(raw, df)
    return df
