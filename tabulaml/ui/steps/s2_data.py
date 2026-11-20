"""Step 2: upload a dataset or load a Sri Lankan sample, then choose the target."""

from __future__ import annotations

import streamlit as st

from tabulaml.config import MAX_FILE_MB
from tabulaml.core.types import ProblemType, ValidationScheme
from tabulaml.data.loader import DataLoadError, list_excel_sheets, read_tabular_file
from tabulaml.data.samples import list_samples, load_sample
from tabulaml.data.validation import suggest_problem_type, validate_target
from tabulaml.preprocessing.cleaning import clean_target
from tabulaml.ui.components import dataset_card_view, issues_view, nav_buttons
from tabulaml.ui.state import AppState, get_state, reset_from
from tabulaml.ui.steps.s2_forecast import forecast_section
from tabulaml.workflow import suggest_validation

PREVIEW_ROWS = 50
EXCEL = (".xlsx", ".xlsm", ".xls")


def _upload_tab() -> None:
    file = st.file_uploader(
        f"CSV, Excel, JSON or Parquet file (up to {MAX_FILE_MB} MB)",
        type=["csv", "txt", "xlsx", "xls", "json", "jsonl", "parquet"],
        key="upload_file",
    )
    if file is None:
        return
    sheet = None
    if file.name.lower().endswith(EXCEL):
        try:
            sheet = st.selectbox("Sheet", list_excel_sheets(file), key="upload_sheet")
        except DataLoadError as exc:
            st.error(str(exc))
            return
    if st.button("Load this file", key="load_upload", type="primary"):
        try:
            df = read_tabular_file(file, file.name, sheet)
        except DataLoadError as exc:
            st.error(str(exc))
            return
        reset_from(
            2, raw_df=df, source="upload", filename=file.name, card=None, target=str(df.columns[-1])
        )
        st.rerun()


def _sample_tab(state: AppState) -> None:
    cards = list_samples(state.problem_type)
    if state.forecasting:  # time series samples only
        cards = [c for c in cards if c.validation == ValidationScheme.TIME_ORDERED.value]
    if not cards:
        st.info("No bundled sample for this problem type.")
        return
    titles = {c.title: c for c in cards}
    title = st.selectbox("Sample dataset", list(titles), key="sample_choice")
    card = titles[title]
    dataset_card_view(card)
    if st.button("Load this sample", key="load_sample", type="primary"):
        df, card = load_sample(card.key)
        reset_from(
            2,
            raw_df=df,
            source="sample",
            filename=f"{card.key}.csv",
            card=card,
            target=card.target,
            validation=ValidationScheme(card.validation),
            group_column=card.group_column,
            order_column=card.order_column,
        )
        st.rerun()


def _target_section(state: AppState) -> bool:
    df = state.raw_df
    st.subheader("Loaded data")
    st.caption(
        f"{state.filename}: {len(df):,} rows x {df.shape[1]} columns, "
        f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB in memory"
    )
    st.dataframe(df.head(PREVIEW_ROWS), width="stretch", hide_index=True)
    columns = list(map(str, df.columns))
    index = columns.index(state.target) if state.target in columns else len(columns) - 1
    target = st.selectbox("Target column (what to predict)", columns, index=index, key="target_col")
    if target != state.target:
        reset_from(3, target=target)
        st.rerun()
    # Minimal check here; step 3 re-checks with the real folds, test share and scheme.
    issues = validate_target(df, target, state.problem_type, cv_folds=2)
    suggested = suggest_problem_type(df[target])
    if suggested != state.problem_type:
        st.info(
            f"'{target}' looks more like a {suggested.value} target. You can go back to "
            "step 1 to change the problem type.",
            icon="ℹ️",
        )
    issues_view(issues)
    ok = not any(i.level == "error" for i in issues)
    if state.forecasting:
        return ok
    return _extra_targets(get_state(), columns) and ok


def _extra_targets(state: AppState, columns: list[str]) -> bool:
    """Optional: predict several columns of the same kind at once (multi-output)."""
    kind = "categories" if state.problem_type == ProblemType.CLASSIFICATION else "numbers"
    several = st.toggle(
        "Predict several columns at once",
        value=bool(state.extra_targets),
        key="multi_toggle",
        help=f"All targets must be {kind}. One model copy is trained per target, and the "
        "other targets are never used as inputs. With yes/no targets this is multi-label "
        "classification.",
    )
    options = [c for c in columns if c != state.target]
    chosen = (
        st.multiselect(
            "Additional target columns",
            options,
            default=[c for c in state.extra_targets if c in options],
            key="extra_targets",
        )
        if several
        else []
    )
    if tuple(chosen) != state.extra_targets:
        reset_from(3, extra_targets=tuple(chosen))
    ok = True
    for name in chosen:
        problems = validate_target(state.raw_df, name, state.problem_type, cv_folds=2)
        for issue in problems:
            if issue.level == "error":
                st.error(f"'{name}': {issue.message}", icon="⛔")
                ok = False
    if several and not chosen:
        st.caption("Choose at least one more target, or switch the option off.")
    return ok


def render(state: AppState) -> None:
    st.header("2. Choose your data")
    upload, sample = st.tabs(["Upload my dataset", "Use a Sri Lankan sample dataset"])
    with upload:
        _upload_tab()
    with sample:
        _sample_tab(state)
    ok = _target_section(state) if state.raw_df is not None else False
    if ok:
        ok = (forecast_section if state.forecasting else _roles_section)(get_state())
    nav_buttons(back=1, next_step=3, next_enabled=ok)


SCHEME_HELP = {
    ValidationScheme.RANDOM: "Random split: rows are independent (one row per customer, lot...).",
    ValidationScheme.GROUPED: "Grouped: several rows belong to the same thing (a customer, a "
    "weekly sale). A whole group goes to training OR testing, never both.",
    ValidationScheme.TIME_ORDERED: "Time-ordered: rows happen over time. The model trains on "
    "the past and is tested on later rows, like real forecasting.",
}


def _role_select(label: str, columns: list[str], current: str | None, key: str) -> str | None:
    index = columns.index(current) if current in columns else 0
    return st.selectbox(label, columns, index=index, key=key)


def _roles_section(state: AppState) -> bool:
    """Column roles: how to test the model, number format and the positive class."""
    df = state.raw_df
    columns = [str(c) for c in df.columns if str(c) not in (state.target, *state.extra_targets)]
    st.subheader("How should the model be tested?")
    if state.source == "upload" and state.validation == ValidationScheme.RANDOM:
        scheme, _, reason = suggest_validation(df, state.target)
        if scheme != ValidationScheme.RANDOM:
            st.info(f"Suggestion: grouped validation. {reason}", icon=":material/lightbulb:")
    options = list(ValidationScheme)
    scheme = st.radio(
        "Validation",
        options,
        index=options.index(state.validation),
        format_func=lambda s: s.value.replace("_", "-").capitalize(),
        horizontal=True,
        key="role_scheme",
    )
    st.caption(SCHEME_HELP[scheme])
    group = order = None
    if scheme == ValidationScheme.GROUPED:
        group = _role_select("Group column", columns, state.group_column, "role_group")
    if scheme == ValidationScheme.TIME_ORDERED:
        order = _role_select("Time / order column", columns, state.order_column, "role_order")
    left, right = st.columns(2)
    decimal = left.radio(
        "Number format",
        [".", ","],
        index=[".", ","].index(state.decimal),
        format_func=lambda d: "1,234.5 (decimal point)" if d == "." else "1.234,5 (decimal comma)",
        key="role_decimal",
        horizontal=True,
    )
    positive = _positive_class(state, right)
    changes = {
        "validation": scheme,
        "group_column": group,
        "order_column": order,
        "decimal": decimal,
        "positive_class": positive,
    }
    if any(getattr(state, k) != v for k, v in changes.items()):
        reset_from(3, **changes)
    return True


def _positive_class(state: AppState, col) -> str | None:  # noqa: ANN001 - Streamlit column
    if state.problem_type != ProblemType.CLASSIFICATION or state.extra_targets:
        return None
    classes = clean_target(state.raw_df[state.target], regression=False).dropna()
    counts = classes.value_counts()
    if len(counts) != 2:
        return None
    options = list(counts.index[::-1])  # minority first: usually the class of interest
    current = state.positive_class if state.positive_class in options else options[0]
    return col.selectbox(
        "Positive class (the outcome you care about)",
        options,
        index=options.index(current),
        key="role_positive",
        help="Precision, recall, F1 and the ROC/PR curves are about this class.",
    )
