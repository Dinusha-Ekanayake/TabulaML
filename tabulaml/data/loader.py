"""Robust reading of user-supplied tabular files (CSV / XLSX / XLS)."""

from __future__ import annotations

import csv
import io
import logging
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
        name = str(col).strip() or "unnamed"
        count = seen.get(name, 0)
        seen[name] = count + 1
        out.append(name if count == 0 else f"{name}_{count}")
    return out


def _sniff_delimiter(text: str) -> str:
    try:
        return csv.Sniffer().sniff(text[:10_000], delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _read_csv(raw: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            # Read the raw header first: pandas silently mangles duplicate names ("a.1").
            sep = _sniff_delimiter(raw.decode(encoding))
            header = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=encoding, header=None, nrows=1)
            names = _dedupe_columns(list(header.iloc[0]))
            return pd.read_csv(io.BytesIO(raw), sep=sep, encoding=encoding, header=0, names=names)
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


def _check_limits(raw: bytes, df: pd.DataFrame) -> None:
    if df.empty:
        raise DataLoadError("The file contains no rows.")
    if df.shape[1] < 2:
        raise DataLoadError("The file needs at least two columns (features and a target).")
    if df.size > MAX_CELLS:
        raise DataLoadError(f"Dataset has {df.size:,} cells; the limit is {MAX_CELLS:,}.")


def read_tabular_file(file_obj: Any, filename: str, sheet: str | None = None) -> pd.DataFrame:
    """Read a CSV/Excel upload. Duplicate column names are made unique, not rejected."""
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
    else:
        raise DataLoadError(f"Unsupported file type '{suffix}'. Use CSV, XLSX or XLS.")
    original = list(df.columns)
    df.columns = _dedupe_columns(original)
    if [str(c) for c in original] != list(df.columns):
        logger.warning("Column names were made unique/non-empty on load.")
    _check_limits(raw, df)
    return df
